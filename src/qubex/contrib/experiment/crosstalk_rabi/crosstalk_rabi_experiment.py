"""Crosstalk Rabi experiment helpers and result persistence utilities."""

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonpickle
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from qxpulse import FlatTop, PulseSchedule

import qubex as qx
import qubex.visualization as viz
from qubex.analysis.fitting import FitResult, FitStatus
from qubex.experiment.experiment_constants import DEFAULT_RABI_TIME_RANGE, HPI_DURATION
from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    RabiData,
    SweepData,
)
from qubex.experiment.models.result import Result

from .crosstalk_rabi_constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CROSSTALK_RABI_TIME_RANGE,
    HIGH_INDEX,
    LOW_INDEX,
    SAVE_DESCRIPTION_TEMPLATE,
    SAVE_FILENAME,
)
from .crosstalk_rabi_result import CrosstalkRabiPairSummary, build_pair_summary


@dataclass(kw_only=True)
class CrosstalkRabiData(RabiData):
    """RabiData with explicit drive-target context for crosstalk experiments."""

    drive_target: str

    @property
    def measure_target(self) -> str:
        """Return the measured target label."""
        pair = self.target.split("-")
        return pair[0]

    @property
    def is_jj(self) -> bool:
        """Return whether the drive and measured target are identical."""
        return self.drive_target == self.measure_target

    @property
    def is_kj(self) -> bool:
        """Return whether the drive and measured target are different."""
        return self.drive_target != self.measure_target


def _next_failed_fit_plot_name(
    *,
    drive_target: str,
    measure_target: str,
    images_dir: Path | str = "images",
) -> str:
    images_path = Path(images_dir)
    images_path.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    prefix = f"CrosstalkRabiExperiment_{drive_target}-{measure_target}_{date}_"
    idx = 0
    while (images_path / f"{prefix}{idx}.png").exists():
        idx += 1
    return f"{prefix}{idx}"


def _save_failed_fit_plot(
    *,
    rabi_data: CrosstalkRabiData,
    drive_target: str,
    measure_target: str,
    kind: str,
    fit_result: FitResult,
    images_dir: Path | str = "images",
) -> Path:
    time_range = np.asarray(rabi_data.time_range)
    fig = viz.make_figure()
    fig.add_trace(
        go.Scatter(
            mode="markers+lines",
            x=time_range,
            y=np.asarray(rabi_data.data).real,
            name="I",
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="markers+lines",
            x=time_range,
            y=np.asarray(rabi_data.data).imag,
            name="Q",
        )
    )
    fig.update_layout(
        title=(
            "Failed crosstalk Rabi fit "
            f"({kind}) : drive={drive_target}, measure={measure_target}, data={rabi_data.target}"
        ),
        xaxis_title="Drive duration (ns)",
        yaxis_title="Signal (arb. units)",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        text=f"status={fit_result.status.value}<br>{fit_result.message or ''}",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
    )

    image_name = _next_failed_fit_plot_name(
        drive_target=drive_target,
        measure_target=measure_target,
        images_dir=images_dir,
    )
    viz.save_figure(fig, name=image_name, images_dir=images_dir)
    image_path = Path(images_dir) / f"{image_name}.png"
    print(f"Saved failed crosstalk Rabi plot to {image_path}")
    return image_path


def _save_config_result(
    config_result: Result,
    save_dir: Path | str = DEFAULT_CONFIG_DIR,
    base_name: str = SAVE_FILENAME,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{base_name}.json"

    config_save_data: dict[str, Any] = {}
    if file_path.exists():
        with file_path.open("r") as f:
            loaded = jsonpickle.decode(f.read())  # noqa: S301
        if isinstance(loaded, dict):
            config_save_data = dict(loaded)

    existing_reference_points = config_save_data.get("reference_points")
    if not isinstance(existing_reference_points, dict):
        existing_reference_points = {}

    new_reference_points = config_result.data["reference_points"]
    if not isinstance(new_reference_points, dict):
        raise TypeError("config_result.data['reference_points'] must be a mapping.")

    merged_reference_points = dict(existing_reference_points)
    merged_reference_points.update(new_reference_points)

    config_save_data["reference_points"] = merged_reference_points
    file_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = jsonpickle.encode(config_save_data, unpicklable=True)
    if not isinstance(encoded, str):
        raise TypeError("jsonpickle.encode() must return a string.")

    with file_path.open("w") as f:
        f.write(encoded)

    print(f"Saved crosstalk Rabi config result to {file_path}")


def _target_index(target: str) -> int:
    if not target.startswith("Q"):
        raise ValueError(f"Target must start with 'Q': {target}")
    return int(target[1:])


def is_high(target: str) -> bool:
    """Return whether the qubit belongs to the high-frequency group."""
    return _target_index(target) % 4 in HIGH_INDEX


def is_low(target: str) -> bool:
    """Return whether the qubit belongs to the low-frequency group."""
    return _target_index(target) % 4 in LOW_INDEX


def _validate_qubit_pairs(drive_target: str, measure_target: str) -> None:
    if drive_target == measure_target:
        raise ValueError("Drive and measure targets must be different.")


def _validate_frequency_group(
    drive_target: str,
    measure_target: str,
) -> None:
    """Validate that both targets belong to the same high/low group."""
    if not (is_high(drive_target) == is_high(measure_target)):
        raise ValueError(
            "crosstalk_rabi_experiment currently supports only drive_target "
            "and measure_target in the same high/low group. "
            f"Got drive_target={drive_target}, measure_target={measure_target}."
        )


def _validate_target_frequency_difference(
    ex: qx.Experiment,
    drive_target: str,
    measure_target: str,
) -> None:
    """Validate that the target frequency difference is below the supported range."""
    frequency_diff = abs(
        ex.targets[drive_target].frequency - ex.targets[measure_target].frequency
    )
    if frequency_diff >= 0.2:
        raise ValueError(
            "crosstalk_rabi_experiment requires the frequency difference between "
            "drive_target and measure_target to be smaller than 0.2 GHz. "
            f"Got drive_target={drive_target}, measure_target={measure_target}, "
            f"difference={frequency_diff:.6f} GHz."
        )


def _crosstalk_rabi_experiment(
    ex: qx.Experiment,
    *,
    drive_target: str,
    measure_target: str,
    crosstalk_rabi_time_range: NDArray | None = None,
    drive_amplitude: float | None = None,
    ramp_time: int = HPI_DURATION,
    plot_rabi_jj: bool = True,
    plot_rabi_kj: bool = True,
) -> Result:

    if crosstalk_rabi_time_range is None:
        crosstalk_rabi_time_range = np.asarray(DEFAULT_CROSSTALK_RABI_TIME_RANGE)

    if drive_amplitude is None:
        drive_amplitude = 1.2 * ex.params.control_amplitude[drive_target]

    reference_points = ex.obtain_reference_points(
        targets=[drive_target, measure_target], n_shots=1024
    )["iq"]

    results_rabi_jj = ex.obtain_rabi_params(
        targets=[drive_target, measure_target],
        time_range=np.asarray(DEFAULT_RABI_TIME_RANGE),
        amplitudes={
            drive_target: drive_amplitude,
            measure_target: ex.params.control_amplitude.get(measure_target, 0.1),
        },  # for drive target only, corresponding to the crosstalk Rabi exp
        plot=plot_rabi_jj,
    )
    rabi_data_jj = CrosstalkRabiData(
        target=f"{drive_target}-{drive_target}",
        data=results_rabi_jj.data[drive_target].data,
        time_range=np.asarray(DEFAULT_RABI_TIME_RANGE),
        rabi_param=results_rabi_jj.data[drive_target].rabi_param,
        drive_target=drive_target,
    )

    def rabi_sequence(T: int):
        with PulseSchedule([drive_target, measure_target]) as ps:
            ps.add(
                drive_target,
                FlatTop(
                    duration=T + 2 * ramp_time,
                    amplitude=drive_amplitude,
                    tau=ramp_time,
                ),
            )
            ps.barrier()
        return ps

    result_rabi_kj: ExperimentResult[SweepData] = ex.sweep_parameter(
        sequence=rabi_sequence,
        sweep_range=crosstalk_rabi_time_range,
        frequencies={
            drive_target: ex.targets[measure_target].frequency,
            measure_target: ex.targets[measure_target].frequency,
        },
        plot=plot_rabi_kj,
    )
    effective_time_range = crosstalk_rabi_time_range + ramp_time
    rabi_data_kj = CrosstalkRabiData(
        target=f"{measure_target}-{drive_target}",
        data=result_rabi_kj.data[measure_target].data,
        time_range=effective_time_range,
        rabi_param=results_rabi_jj.data[measure_target].rabi_param,
        drive_target=drive_target,
    )

    return Result(
        data={
            "drive_target": drive_target,
            "measure_target": measure_target,
            "effective_time_range": effective_time_range,
            "rabi_data_jj": rabi_data_jj,
            "rabi_data_kj": rabi_data_kj,
            "reference_points": reference_points,
        }
    )


def _measure_crosstalk_rabi_experiment(
    ex: qx.Experiment,
    *,
    drive_target: str,
    measure_target: str,
    crosstalk_rabi_time_range: NDArray | None = None,
    drive_amplitude: float | None = None,
    plot_rabi: bool = True,
    plot_fit: bool = True,
) -> Result:
    """Run the measurement, fit, and persistence flow for one crosstalk pair."""
    if crosstalk_rabi_time_range is None:
        crosstalk_rabi_time_range = np.asarray(DEFAULT_CROSSTALK_RABI_TIME_RANGE)

    if drive_amplitude is None:
        drive_amplitude = 1.2 * ex.params.control_amplitude[drive_target]

    result = _crosstalk_rabi_experiment(
        ex=ex,
        drive_target=drive_target,
        measure_target=measure_target,
        crosstalk_rabi_time_range=crosstalk_rabi_time_range,
        drive_amplitude=drive_amplitude,
        plot_rabi_jj=plot_rabi,
        plot_rabi_kj=plot_rabi,
    )
    reference_points = result.data["reference_points"]
    rabi_data_jj: CrosstalkRabiData = result.data["rabi_data_jj"]
    rabi_data_kj: CrosstalkRabiData = result.data["rabi_data_kj"]

    fit_result_jj: FitResult = rabi_data_jj.fit(
        is_damped=True,
        reference_point=reference_points[drive_target],
        plot=plot_fit,
    )
    fit_result_kj: FitResult = rabi_data_kj.fit(
        is_damped=True,
        reference_point=reference_points[measure_target],
        plot=plot_fit,
    )

    summary = build_pair_summary(
        drive_target=drive_target,
        measure_target=measure_target,
        fit_result_jj=fit_result_jj,
        fit_result_kj=fit_result_kj,
    )

    if fit_result_jj.status == FitStatus.SUCCESS:
        result_jj = ExperimentResult(data={measure_target: rabi_data_jj})
        result_jj.save(
            name=f"{SAVE_FILENAME}",
            description=SAVE_DESCRIPTION_TEMPLATE(drive_target, measure_target),
        )
    else:
        _save_failed_fit_plot(
            rabi_data=rabi_data_jj,
            drive_target=drive_target,
            measure_target=measure_target,
            kind="jj",
            fit_result=fit_result_jj,
        )
    if fit_result_kj.status == FitStatus.SUCCESS:
        result_kj = ExperimentResult(data={measure_target: rabi_data_kj})
        result_kj.save(
            name=f"{SAVE_FILENAME}",
            description=SAVE_DESCRIPTION_TEMPLATE(drive_target, measure_target),
        )
    else:
        _save_failed_fit_plot(
            rabi_data=rabi_data_kj,
            drive_target=drive_target,
            measure_target=measure_target,
            kind="kj",
            fit_result=fit_result_kj,
        )

    config_result = Result(
        data={
            "drive_target": drive_target,
            "measure_target": measure_target,
            "reference_points": reference_points,
            "pair_summary": summary,
            "status": summary.status,
        }
    )

    _save_config_result(config_result)
    return config_result


def measure_crosstalk_rabi_experiment(
    ex: qx.Experiment,
    *,
    drive_target: str,
    measure_target: str,
    crosstalk_rabi_time_range: NDArray | None = None,
    drive_amplitude: float | None = None,
    plot_rabi: bool = True,
    plot_fit: bool = True,
) -> Result:
    """Run the crosstalk Rabi experiment for targets in the same frequency group."""
    if crosstalk_rabi_time_range is None:
        crosstalk_rabi_time_range = np.asarray(DEFAULT_CROSSTALK_RABI_TIME_RANGE)

    try:
        _validate_qubit_pairs(
            drive_target,
            measure_target,
        )

        _validate_frequency_group(
            drive_target,
            measure_target,
        )
        _validate_target_frequency_difference(
            ex,
            drive_target,
            measure_target,
        )
    except ValueError as exc:
        summary = CrosstalkRabiPairSummary(
            drive_target=drive_target,
            measure_target=measure_target,
            status="skipped",
            warning=str(exc),
        )
        warnings.warn(str(exc), stacklevel=2)
        return Result(
            data={
                "drive_target": drive_target,
                "measure_target": measure_target,
                "pair_summary": summary,
                "warning": str(exc),
                "status": "skipped",
            }
        )
    return _measure_crosstalk_rabi_experiment(
        ex=ex,
        drive_target=drive_target,
        measure_target=measure_target,
        crosstalk_rabi_time_range=crosstalk_rabi_time_range,
        drive_amplitude=drive_amplitude,
        plot_rabi=plot_rabi,
        plot_fit=plot_fit,
    )
