"""Pair-level summaries for crosstalk Rabi measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qubex.analysis.fitting import FitResult, FitStatus

if TYPE_CHECKING:
    pass


@dataclass
class CrosstalkRabiPairSummary:
    """Summarize one measured drive/measure pair for matrix updates."""

    drive_target: str
    measure_target: str
    status: str
    jj_frequency: float | None = None
    kj_frequency: float | None = None
    amplitude_hw_for_jj: float | None = None
    amplitude_hw_for_kj: float | None = None
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


def _calc_ratio(
    jj_frequency: float,
    kj_frequency: float,
    amplitude_hw_for_jj: float | None = None,
    amplitude_hw_for_kj: float | None = None,
) -> float:
    if amplitude_hw_for_jj is None or amplitude_hw_for_kj is None:
        Warning("Amplitude for JJ or KJ is None.")
        return np.nan
    r = amplitude_hw_for_jj / amplitude_hw_for_kj
    return (kj_frequency / jj_frequency) * r


def build_pair_summary(
    *,
    drive_target: str,
    measure_target: str,
    fit_result_jj: FitResult,
    fit_result_kj: FitResult,
    amplitude_for_drive_target_rabi: float | None = None,
    amplitude_for_measure_target_rabi: float | None = None,
    warning: str | None = None,
) -> CrosstalkRabiPairSummary:
    """Build a pair summary from the two fit results."""
    jj_frequency = fit_frequency(fit_result_jj)
    kj_frequency = fit_frequency(fit_result_kj)
    ratio = np.nan
    status = "fit_failed"
    if (
        fit_result_jj.status == FitStatus.SUCCESS
        and fit_result_kj.status == FitStatus.SUCCESS
        and jj_frequency is not None
        and kj_frequency is not None
        and jj_frequency != 0
        and amplitude_for_measure_target_rabi != 0
    ):
        ratio = _calc_ratio(
            jj_frequency=jj_frequency,
            kj_frequency=kj_frequency,
            amplitude_hw_for_jj=amplitude_for_drive_target_rabi,
            amplitude_hw_for_kj=amplitude_for_measure_target_rabi,
        )
        status = "measured"

    return CrosstalkRabiPairSummary(
        drive_target=drive_target,
        measure_target=measure_target,
        status=status,
        jj_frequency=jj_frequency,
        kj_frequency=kj_frequency,
        amplitude_hw_for_jj=amplitude_for_drive_target_rabi,
        amplitude_hw_for_kj=amplitude_for_measure_target_rabi,
        ratio=float(ratio),
        jj_fit_status=fit_result_jj.status,
        kj_fit_status=fit_result_kj.status,
        warning=warning,
    )
