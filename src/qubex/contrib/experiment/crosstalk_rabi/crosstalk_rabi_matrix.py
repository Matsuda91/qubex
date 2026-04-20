"""Matrix helpers for crosstalk Rabi characterization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
from numpy.typing import NDArray

from .crosstalk_rabi_result import CrosstalkRabiPairSummary
from .crosstalk_rabi_status import (
    STATUS_BY_SUMMARY,
    STATUS_LABELS,
    STATUS_NOT_CROSSTALK_PAIR,
    STATUS_NOT_MEASURED,
    STATUS_SKIPPED,
)

jsonpickle_numpy.register_handlers()


def _canonical_target(target: str) -> str:
    if target.startswith("Q") and target[1:].isdigit():
        return f"Q{int(target[1:])}"
    return target


def _build_target_index(targets: list[str]) -> dict[str, int]:
    target_index: dict[str, int] = {}
    for index, target in enumerate(targets):
        target_index.setdefault(target, index)
        target_index.setdefault(_canonical_target(target), index)
    return target_index


def _matrix_figure_size(target_count: int) -> int:
    return int(np.clip(420 + 12 * max(target_count, 1), 720, 1800))


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
            row = target_index[_canonical_target(summary.measure_target)]
            column = target_index[_canonical_target(summary.drive_target)]
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
                        "ratio": None if np.isnan(value) else float(np.round(value, 2)),
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
        figure_size = _matrix_figure_size(len(self.targets))
        label_matrix = np.vectorize(STATUS_LABELS.get)(self.status_matrix)
        text_size = int(np.clip(np.round(600 / max(len(self.targets), 1)), 7, 12))
        text = np.where(
            np.isnan(self.r_matrix), "", np.round(self.r_matrix, 2).astype(str)
        )
        fig = viz.make_figure()
        fig.add_trace(
            go.Heatmap(
                z=self.r_matrix,
                x=self.targets,
                y=self.targets,
                text=text,
                customdata=label_matrix,
                texttemplate="%{text}",
                textfont={"size": text_size},
                colorscale="Viridis",
                colorbar_title="r_kj",
                zmin=np.min(self.r_matrix[np.isfinite(self.r_matrix)]),
                zmax=np.max(self.r_matrix[np.isfinite(self.r_matrix)]),
                hovertemplate=(
                    "measure=%{y}<br>drive=%{x}<br>r_kj=%{z:.5f}<br>status=%{customdata}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title,
            width=figure_size,
            height=figure_size,
            xaxis_title="drive target j",
            yaxis_title="measure target k",
            xaxis=dict(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)"),
        )
        return fig

    def plot_status_matrix(
        self, title: str = "Crosstalk Rabi acquisition status"
    ) -> go.Figure:
        """Render the acquisition status as a categorical heatmap."""
        figure_size = _matrix_figure_size(len(self.targets))
        label_matrix = np.vectorize(STATUS_LABELS.get)(self.status_matrix)
        colorscale = [
            [0.0, "#ffffff"],
            [0.2, "#ffffff"],
            [0.2, "#b2b2b2"],
            [0.4, "#b2b2b2"],
            [0.4, "#1f77b4"],
            [0.6, "#1f77b4"],
            [0.6, "#d62728"],
            [0.8, "#d62728"],
            [0.8, "#ff7f0e"],
            [1.0, "#ff7f0e"],
        ]
        fig = viz.make_figure()
        fig.add_trace(
            go.Heatmap(
                z=self.status_matrix,
                x=self.targets,
                y=self.targets,
                text=label_matrix,
                texttemplate="%{text}",
                colorscale=colorscale,
                zmin=STATUS_NOT_CROSSTALK_PAIR,
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
        )
        fig.update_layout(
            title=title,
            width=figure_size,
            height=figure_size,
            xaxis_title="drive target j",
            yaxis_title="measure target k",
        )
        return fig
