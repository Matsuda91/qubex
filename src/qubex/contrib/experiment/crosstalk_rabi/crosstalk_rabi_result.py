"""Pair-level summaries for crosstalk Rabi measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qubex.analysis.fitting import FitResult, FitStatus
from qubex.experiment.models.experiment_record import ExperimentRecord
from qubex.experiment.models.experiment_result import ExperimentResult

from .crosstalk_rabi_constants import DEFAULT_DATA_DIR, SAVE_FILENAME
from .crosstalk_rabi_experiment import CrosstalkRabiData


@dataclass
class CrosstalkRabiPairSummary:
    """Summarize one measured drive/measure pair for matrix updates."""

    drive_target: str
    measure_target: str
    status: str
    jj_frequency: float | None = None
    kj_frequency: float | None = None
    ratio: float = np.nan
    jj_fit_status: FitStatus | None = None
    kj_fit_status: FitStatus | None = None
    warning: str | None = None


def fit_status_name(fit_result: FitResult) -> str:
    """Return a stable string representation of the fit status."""
    status = fit_result.status
    return status.name.lower() if hasattr(status, "name") else str(status).lower()


def fit_frequency(fit_result: FitResult) -> float | None:
    """Extract the fitted Rabi frequency when available."""
    frequency = fit_result.data.get("frequency")
    if frequency is None:
        return None
    return float(frequency)


def build_pair_summary(
    *,
    drive_target: str,
    measure_target: str,
    fit_result_jj: FitResult,
    fit_result_kj: FitResult,
    warning: str | None = None,
) -> CrosstalkRabiPairSummary:
    """Build a pair summary from the two fit results."""
    jj_frequency = fit_frequency(fit_result_jj)
    kj_frequency = fit_frequency(fit_result_kj)
    ratio = np.nan
    status = "fit_failed"
    if jj_frequency is not None and kj_frequency is not None and jj_frequency != 0:
        ratio = kj_frequency / jj_frequency
        if (
            fit_result_jj.status == FitStatus.SUCCESS
            and fit_result_kj.status == FitStatus.SUCCESS
        ):
            status = "measured"

    return CrosstalkRabiPairSummary(
        drive_target=drive_target,
        measure_target=measure_target,
        status=status,
        jj_frequency=jj_frequency,
        kj_frequency=kj_frequency,
        ratio=float(ratio),
        jj_fit_status=fit_result_jj.status,
        kj_fit_status=fit_result_kj.status,
        warning=warning,
    )


def find_crosstalk_rabi_experiment_jsons(
    *,
    drive_target: str,
    measure_target: str,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> ExperimentResult[CrosstalkRabiData]:
    """Load the latest saved ExperimentResult for one crosstalk-Rabi pair."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    matched_records: list[
        tuple[str, float, str, ExperimentResult[CrosstalkRabiData]]
    ] = []
    for path in sorted(data_path.glob(f"*_{SAVE_FILENAME}_*.json")):
        record = ExperimentRecord.load(path.name, data_dir=str(data_path))
        experiment_result = record.data
        if not isinstance(experiment_result, ExperimentResult):
            continue
        if len(experiment_result.data) != 1:
            continue

        target_data = next(iter(experiment_result.data.values()))
        if isinstance(target_data, CrosstalkRabiData):
            if (
                target_data.drive_target == drive_target
                and target_data.measure_target == measure_target
            ):
                matched_records.append(
                    (
                        record.created_at,
                        path.stat().st_mtime,
                        path.name,
                        experiment_result,
                    )
                )
                continue

        description = record.description
        if (
            f"drive_target={drive_target}" in description
            and f"measure_target={measure_target}" in description
        ):
            matched_records.append(
                (
                    record.created_at,
                    path.stat().st_mtime,
                    path.name,
                    experiment_result,
                )
            )

    if not matched_records:
        raise FileNotFoundError(
            "No crosstalk-Rabi ExperimentResult JSON found for "
            f"drive_target={drive_target}, measure_target={measure_target}."
        )

    matched_records.sort(key=lambda item: (item[0], item[1], item[2]))
    return matched_records[-1][3]
