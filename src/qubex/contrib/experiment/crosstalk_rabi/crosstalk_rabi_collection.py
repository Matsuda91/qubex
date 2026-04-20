"""Unified access to crosstalk-Rabi matrices and saved pair results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from qubex.experiment.models.experiment_record import ExperimentRecord
from qubex.experiment.models.experiment_result import ExperimentResult

from .crosstalk_rabi_constants import DEFAULT_DATA_DIR, SAVE_FILENAME
from .crosstalk_rabi_matrix import (
    STATUS_LABELS,
    CrosstalkRabiMatrix,
    _build_target_index,
    _canonical_target,
)
from .crosstalk_rabi_result import CrosstalkRabiPairSummary


@dataclass
class CrosstalkRabiCollection:
    """Provide one entry point for matrix views and saved pair-level Rabi data."""

    matrix: CrosstalkRabiMatrix
    matrix_path: Path | None = None
    data_dir: Path = Path(DEFAULT_DATA_DIR)

    @classmethod
    def create(
        cls,
        targets: list[str],
        *,
        matrix_path: Path | str | None = None,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ) -> CrosstalkRabiCollection:
        """Create an empty collection for the given targets."""
        return cls(
            matrix=CrosstalkRabiMatrix.create(targets),
            matrix_path=Path(matrix_path) if matrix_path is not None else None,
            data_dir=Path(data_dir),
        )

    @classmethod
    def load(
        cls,
        matrix_path: Path | str,
        *,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ) -> CrosstalkRabiCollection:
        """Load a collection from a saved matrix file."""
        path = Path(matrix_path)
        return cls(
            matrix=CrosstalkRabiMatrix.load(path),
            matrix_path=path,
            data_dir=Path(data_dir),
        )

    def save(self, matrix_path: Path | str | None = None) -> Path:
        """Save the current matrix and return the written path."""
        target_path = Path(matrix_path) if matrix_path is not None else self.matrix_path
        if target_path is None:
            raise ValueError("matrix_path must be provided before saving.")
        self.matrix_path = target_path
        return self.matrix.save(target_path)

    def update(self, summary: CrosstalkRabiPairSummary) -> None:
        """Update the matrix with one pair summary."""
        self.matrix.update(summary)

    def pending_pairs(self) -> list[tuple[str, str]]:
        """Return drive/measure pairs that are still pending."""
        return self.matrix.pending_pairs()

    def status_table(self) -> list[dict[str, str | float | int | None]]:
        """Return the flattened matrix status table."""
        return self.matrix.status_table()

    def plot_ratio_matrix(
        self, title: str = "Crosstalk Rabi ratio matrix"
    ) -> go.Figure:
        """Plot the stored crosstalk ratio matrix."""
        return self.matrix.plot_ratio_matrix(title=title)

    def plot_status_matrix(
        self,
        title: str = "Crosstalk Rabi acquisition status",
    ) -> go.Figure:
        """Plot the stored crosstalk acquisition-status matrix."""
        return self.matrix.plot_status_matrix(title=title)

    def find_result(
        self,
        drive_target: str,
        measure_target: str,
    ) -> ExperimentResult[Any]:
        """Load the latest saved ExperimentResult for one drive/measure pair."""
        data_path = self.data_dir
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_path}")

        print(
            "Crosstalk Rabi matrix status: "
            f"drive_target={drive_target}, measure_target={measure_target}, "
            f"status={self._pair_status_label(drive_target, measure_target)}"
        )

        matched_records: list[tuple[str, float, str, ExperimentResult[Any]]] = []
        for path in sorted(data_path.glob(f"*_{SAVE_FILENAME}_*.json")):
            record = ExperimentRecord.load(path.name, data_dir=str(data_path))
            experiment_result = record.data
            if not isinstance(experiment_result, ExperimentResult):
                continue
            if len(experiment_result.data) != 1:
                continue

            target_data = next(iter(experiment_result.data.values()))
            if (
                getattr(target_data, "drive_target", None) == drive_target
                and getattr(
                    target_data,
                    "measure_target",
                    getattr(target_data, "target", None),
                )
                == measure_target
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

    def plot_rabi(
        self,
        drive_target: str,
        measure_target: str,
        **kwargs,
    ) -> go.Figure | None:
        """Load and plot the latest saved Rabi result for one drive/measure pair."""
        experiment_result = self.find_result(
            drive_target=drive_target,
            measure_target=measure_target,
        )
        target_data = next(iter(experiment_result.data.values()))
        if not hasattr(target_data, "plot"):
            raise TypeError("Loaded crosstalk-Rabi result does not support plotting.")
        return target_data.plot(**kwargs)

    def _pair_status_label(self, drive_target: str, measure_target: str) -> str:
        """Return the current matrix status label for one drive/measure pair."""
        target_index = _build_target_index(self.matrix.targets)
        try:
            row = target_index[_canonical_target(measure_target)]
            column = target_index[_canonical_target(drive_target)]
        except KeyError:
            return "unknown"

        status = int(self.matrix.status_matrix[row, column])
        return STATUS_LABELS.get(status, str(status))
