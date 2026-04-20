"""Pair-level summaries for crosstalk Rabi measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubex.analysis.fitting import FitResult, FitStatus


@dataclass
class CrosstalkRabiPairSummary:
    """Summarize one measured drive/measure pair for matrix updates."""

    drive_target: str
    measure_target: str
    status: str
    jj_frequency: float | None = None
    kj_frequency: float | None = None
    ratio: float = np.nan
    jj_fit_status: str | None = None
    kj_fit_status: str | None = None
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
