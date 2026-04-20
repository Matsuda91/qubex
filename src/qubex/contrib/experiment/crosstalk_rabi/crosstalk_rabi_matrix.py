"""Matrix helpers for crosstalk Rabi characterization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .crosstalk_rabi_pair_summary import CrosstalkRabiPairSummary

jsonpickle_numpy.register_handlers()

STATUS_NOT_MEASURED = 0
STATUS_MEASURED = 1
STATUS_FIT_FAILED = 2
STATUS_SKIPPED = 3

STATUS_LABELS = {
    STATUS_NOT_MEASURED: "not_measured",
    STATUS_MEASURED: "measured",
    STATUS_FIT_FAILED: "fit_failed",
    STATUS_SKIPPED: "skipped",
}

STATUS_BY_SUMMARY = {
    "not_measured": STATUS_NOT_MEASURED,
    "measured": STATUS_MEASURED,
    "fit_failed": STATUS_FIT_FAILED,
    "skipped": STATUS_SKIPPED,
}


def _build_target_index(targets: list[str]) -> dict[str, int]:
    return {target: index for index, target in enumerate(targets)}


@dataclass
class CrosstalkRabiMatrix:
    """Store the crosstalk ratio matrix and its acquisition status."""

    targets: list[str]
    r_matrix: NDArray[np.float64]
    status_matrix: NDArray[np.int_]

    @classmethod
    def create(cls, targets: list[str]) -> CrosstalkRabiMatrix:
        """Create an empty matrix initialized with NaNs and not-measured status."""
        size = len(targets)
        return cls(
            targets=list(targets),
            r_matrix=np.full((size, size), np.nan, dtype=np.float64),
            status_matrix=np.full((size, size), STATUS_NOT_MEASURED, dtype=np.int_),
        )

    @classmethod
    def load(cls, path: Path | str) -> CrosstalkRabiMatrix:
        """Load a matrix from a JSON file encoded with jsonpickle."""
        file_path = Path(path)
        with file_path.open("r") as file:
            loaded = jsonpickle.decode(file.read())  # noqa: S301
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded)}")
        return loaded

    def save(self, path: Path | str) -> Path:
        """Save the matrix to a JSON file encoded with jsonpickle."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = jsonpickle.encode(self, unpicklable=True)
        if not isinstance(encoded, str):
            raise TypeError("jsonpickle.encode() must return a string.")
        with file_path.open("w") as file:
            file.write(encoded)
        return file_path

    def update(self, summary: CrosstalkRabiPairSummary) -> None:
        """Update one matrix entry from a pair summary."""
        target_index = _build_target_index(self.targets)
        try:
            row = target_index[summary.measure_target]
            column = target_index[summary.drive_target]
        except KeyError as exc:
            raise ValueError(f"Unknown target in summary: {exc.args[0]}") from exc

        self.r_matrix[row, column] = summary.ratio
        self.status_matrix[row, column] = STATUS_BY_SUMMARY.get(
            summary.status, STATUS_NOT_MEASURED
        )

    def pending_pairs(self) -> list[tuple[str, str]]:
        """Return all drive/measure pairs that have not been measured yet."""
        pending: list[tuple[str, str]] = []
        for row, measure_target in enumerate(self.targets):
            for column, drive_target in enumerate(self.targets):
                if self.status_matrix[row, column] == STATUS_NOT_MEASURED:
                    pending.append((drive_target, measure_target))
        return pending

    def status_table(self) -> list[dict[str, str | float | int | None]]:
        """Return a flat table view for quick notebook inspection."""
        rows: list[dict[str, str | float | int | None]] = []
        for row, measure_target in enumerate(self.targets):
            for column, drive_target in enumerate(self.targets):
                value = self.r_matrix[row, column]
                rows.append(
                    {
                        "drive_target": drive_target,
                        "measure_target": measure_target,
                        "ratio": None if np.isnan(value) else float(value),
                        "status": int(self.status_matrix[row, column]),
                        "status_label": STATUS_LABELS[
                            int(self.status_matrix[row, column])
                        ],
                    }
                )
        return rows

    def plot_ratio_matrix(
        self, title: str = "Crosstalk Rabi ratio matrix"
    ) -> go.Figure:
        """Render the ratio matrix as a heatmap."""
        text = np.where(
            np.isnan(self.r_matrix), "", np.round(self.r_matrix, 4).astype(str)
        )
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=self.r_matrix,
                    x=self.targets,
                    y=self.targets,
                    text=text,
                    texttemplate="%{text}",
                    colorbar_title="r_kj",
                    hovertemplate=(
                        "measure=%{y}<br>drive=%{x}<br>r_kj=%{z:.6f}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title="drive target j",
            yaxis_title="measure target k",
        )
        return fig

    def plot_status_matrix(
        self, title: str = "Crosstalk Rabi acquisition status"
    ) -> go.Figure:
        """Render the acquisition status as a categorical heatmap."""
        label_matrix = np.vectorize(STATUS_LABELS.get)(self.status_matrix)
        colorscale = [
            [0.0, "#d9d9d9"],
            [0.25, "#d9d9d9"],
            [0.25, "#1f77b4"],
            [0.5, "#1f77b4"],
            [0.5, "#d62728"],
            [0.75, "#d62728"],
            [0.75, "#ff7f0e"],
            [1.0, "#ff7f0e"],
        ]
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=self.status_matrix,
                    x=self.targets,
                    y=self.targets,
                    text=label_matrix,
                    texttemplate="%{text}",
                    colorscale=colorscale,
                    zmin=STATUS_NOT_MEASURED,
                    zmax=STATUS_SKIPPED,
                    colorbar=dict(
                        title="status",
                        tickvals=list(STATUS_LABELS),
                        ticktext=[STATUS_LABELS[key] for key in STATUS_LABELS],
                    ),
                    hovertemplate=(
                        "measure=%{y}<br>drive=%{x}<br>status=%{text}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title="drive target j",
            yaxis_title="measure target k",
        )
        return fig
