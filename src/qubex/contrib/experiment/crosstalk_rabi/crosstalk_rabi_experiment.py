"""Crosstalk Rabi experiment helpers and result persistence utilities."""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qxpulse import FlatTop, PulseSchedule

import qubex as qx
from qubex.analysis.fitting import FitResult, FitStatus
from qubex.experiment.experiment_constants import DEFAULT_RABI_TIME_RANGE, HPI_DURATION
from qubex.experiment.models.experiment_result import (
    ExperimentResult,
    RabiData,
    SweepData,
)
from qubex.experiment.models.result import Result

from .crosstalk_rabi_constants import (
    DEFAULT_CROSSTALK_RABI_TIME_RANGE,
    DEFAULT_CROSSTALK_RATIO,
    DEFAULT_SAMPLING_PERIOD,
    HIGH_INDEX,
    LOW_INDEX,
    QUBITS_WITH_CONTROL_LINE_AMP_IN_64Q,
    QUBITS_WITH_CONTROL_LINE_AMP_IN_144Q,
    SAVE_DESCRIPTION_TEMPLATE,
    SAVE_FILENAME,
)
from .crosstalk_rabi_record import CrosstalkRabiRecord
from .crosstalk_rabi_result import CrosstalkRabiPairSummary, build_pair_summary


class FrequencyType:
    """Enum-like class for frequency group labels."""

    HIGH = "High"
    LOW = "Low"


@dataclass(kw_only=True)
class CrosstalkRabiData(RabiData):
    """RabiData with explicit drive-target context for crosstalk experiments."""

    drive_target: str
    drive_amplitude_hw: float
    reference_point: complex | None = None

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


def _invalid_qubit_pairs(drive_target: str, measure_target: str) -> None:
    if drive_target == measure_target:
        raise ValueError("Drive and measure targets must be different.")


def _get_frequency_group(
    drive_target: str,
    measure_target: str,
) -> dict[str, str]:
    """Validate that both targets belong to the same high/low group."""
    drive_target_type = (
        FrequencyType.HIGH if is_high(drive_target) else FrequencyType.LOW
    )
    measure_target_type = (
        FrequencyType.HIGH if is_high(measure_target) else FrequencyType.LOW
    )
    return {
        "drive_target": drive_target_type,
        "measure_target": measure_target_type,
    }


def _invalid_qubit_with_amp(ex: qx.Experiment, target: str) -> None:

    _message = (
        f"crosstalk_rabi_experiment currently does not support target {target} on {ex.chip_id} due to amplitude control issues. "
        "Please refer to the experiment documentation for details."
    )

    if "144" in ex.chip_id:
        if target in QUBITS_WITH_CONTROL_LINE_AMP_IN_144Q:
            raise ValueError(_message)
    elif "64" in ex.chip_id:
        if target in QUBITS_WITH_CONTROL_LINE_AMP_IN_64Q:
            raise ValueError(_message)
    else:
        raise ValueError(
            f"Unsupported chip_id {ex.chip_id} for crosstalk_rabi_experiment."
        )


def _invalid_target_frequency_difference(
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
        )


def _invalid_message_for_test(
    drive_target: str,
    measure_target: str,
) -> None:
    if True:
        raise ValueError(
            "crosstalk_rabi_experiment currently supports only drive_target "
            "and measure_target in the same high/low group. "
            f"Got drive_target={drive_target}, measure_target={measure_target}."
        )


def _crosstalk_rabi_experiment(
    ex: qx.Experiment,
    *,
    drive_target: str,
    measure_target: str,
    crosstalk_rabi_time_range: NDArray | None = None,
    amplitude_for_drive_target_rabi: float | None = None,
    amplitude_for_measure_target_rabi: float | None = None,
    ramp_time: int = HPI_DURATION,
    plot_rabi_jj: bool = True,
    plot_rabi_kj: bool = True,
) -> Result:

    if crosstalk_rabi_time_range is None:
        max_rabi_freq = ex.calc_rabi_rate(target=drive_target, control_amplitude=1.0)
        crosstalk_rabi_freq = DEFAULT_CROSSTALK_RATIO * max_rabi_freq
        T = int(1 / crosstalk_rabi_freq)
        dt = max(
            DEFAULT_SAMPLING_PERIOD,
            int(T / 10) // DEFAULT_SAMPLING_PERIOD * DEFAULT_SAMPLING_PERIOD,
        )
        crosstalk_rabi_time_range = np.arange(0, int(3 * T), dt)

    if amplitude_for_drive_target_rabi is None:
        amplitude_for_drive_target_rabi = ex.params.control_amplitude.get(drive_target)
    if amplitude_for_measure_target_rabi is None:
        amplitude_for_measure_target_rabi = ex.params.control_amplitude.get(
            measure_target
        )

    reference_points = ex.obtain_reference_points(
        targets=[drive_target, measure_target], n_shots=1024
    )["iq"]

    results_rabi_jj = ex.obtain_rabi_params(
        targets=[drive_target, measure_target],
        time_range=np.asarray(DEFAULT_RABI_TIME_RANGE),
        amplitudes={
            drive_target: amplitude_for_drive_target_rabi,  # for crosstalk evaluation
            measure_target: ex.params.control_amplitude.get(
                measure_target
            ),  # for normalization.
        },
        plot=plot_rabi_jj,
    )
    rabi_data_jj = CrosstalkRabiData(
        target=f"{drive_target}-{drive_target}",
        data=results_rabi_jj.data[drive_target].data,
        reference_point=reference_points[drive_target],
        time_range=np.asarray(DEFAULT_RABI_TIME_RANGE),
        rabi_param=results_rabi_jj.data[drive_target].rabi_param,
        drive_target=drive_target,
        drive_amplitude_hw=ex.params.control_amplitude.get(drive_target),
    )

    def crosstalk_rabi_sequence(T: int):
        with PulseSchedule([drive_target, measure_target]) as ps:
            ps.add(
                drive_target,
                FlatTop(
                    duration=T + 2 * ramp_time,
                    amplitude=amplitude_for_measure_target_rabi,
                    tau=ramp_time,
                ),
            )
            ps.barrier()
        return ps

    result_rabi_kj: ExperimentResult[SweepData] = ex.sweep_parameter(
        sequence=crosstalk_rabi_sequence,
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
        reference_point=reference_points[measure_target],
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

    jj_result_file: str | None = None
    if fit_result_jj.status == FitStatus.SUCCESS:
        result_jj = ExperimentResult(data={measure_target: rabi_data_jj})
        saved_jj = result_jj.save(
            name=f"{SAVE_FILENAME}",
            description=SAVE_DESCRIPTION_TEMPLATE(drive_target, measure_target),
        )
        jj_result_file = saved_jj.file_name

    kj_result_file: str | None = None
    if fit_result_kj.status == FitStatus.SUCCESS:
        result_kj = ExperimentResult(data={measure_target: rabi_data_kj})
        saved_kj = result_kj.save(
            name=f"{SAVE_FILENAME}",
            description=SAVE_DESCRIPTION_TEMPLATE(drive_target, measure_target),
        )
        kj_result_file = saved_kj.file_name

    pair_record = CrosstalkRabiRecord(
        drive_target=drive_target,
        measure_target=measure_target,
        pair_summary=summary,
        jj_result_file=jj_result_file,
        kj_result_file=kj_result_file,
    )
    saved_pair_record = pair_record.save()


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
        _invalid_qubit_pairs(
            drive_target,
            measure_target,
        )
    except ValueError as exc:
        summary = CrosstalkRabiPairSummary(
            drive_target=drive_target,
            measure_target=measure_target,
            status="not_crosstalk_pair",
            warning=str(exc),
        )
        pair_record = CrosstalkRabiRecord(
            drive_target=drive_target,
            measure_target=measure_target,
            pair_summary=summary,
        )
        saved_pair_record = pair_record.save()
        warnings.warn(str(exc), stacklevel=2)
        return Result(
            data={
                "drive_target": drive_target,
                "measure_target": measure_target,
                "pair_summary": summary,
                "pair_record": pair_record,
                "pair_record_file": saved_pair_record.file_name,
                "warning": str(exc),
                "status": "not_crosstalk_pair",
            }
        )
    frequency_group = _get_frequency_group(
        drive_target,
        measure_target,
    )

    if frequency_group["drive_target"] == frequency_group["measure_target"]:
        try:
            _invalid_target_frequency_difference(
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
            pair_record = CrosstalkRabiRecord(
                drive_target=drive_target,
                measure_target=measure_target,
                pair_summary=summary,
            )
            saved_pair_record = pair_record.save()
            warnings.warn(str(exc), stacklevel=2)
            return Result(
                data={
                    "drive_target": drive_target,
                    "measure_target": measure_target,
                    "pair_summary": summary,
                    "pair_record": pair_record,
                    "pair_record_file": saved_pair_record.file_name,
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
    else:
        try:
            _invalid_message_for_test(
                drive_target=drive_target,
                measure_target=measure_target,
            )
        except ValueError as exc:
            summary = CrosstalkRabiPairSummary(
                drive_target=drive_target,
                measure_target=measure_target,
                status="skipped",
                warning=str(exc),
            )
            pair_record = CrosstalkRabiRecord(
                drive_target=drive_target,
                measure_target=measure_target,
                pair_summary=summary,
            )
            saved_pair_record = pair_record.save()
            warnings.warn(str(exc), stacklevel=2)
            return Result(
                data={
                    "drive_target": drive_target,
                    "measure_target": measure_target,
                    "pair_summary": summary,
                    "pair_record": pair_record,
                    "pair_record_file": saved_pair_record.file_name,
                    "warning": str(exc),
                    "status": "skipped",
                }
            )
