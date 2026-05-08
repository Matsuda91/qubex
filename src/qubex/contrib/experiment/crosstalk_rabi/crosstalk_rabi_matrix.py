"""Matrix helpers for crosstalk Rabi characterization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import plotly.graph_objects as go
import qxvisualizer as viz
from numpy.typing import NDArray

from .crosstalk_rabi_constants import HIGH_INDEX, LOW_INDEX
from .crosstalk_rabi_result import CrosstalkRabiPairSummary
from .crosstalk_rabi_status import (
    STATUS_BY_SUMMARY,
    STATUS_FIT_FAILED,
    STATUS_LABELS,
    STATUS_MEASURED,
    STATUS_NOT_CROSSTALK_PAIR,
    STATUS_NOT_MEASURED,
    STATUS_SKIPPED,
)

if TYPE_CHECKING:
    from .crosstalk_rabi_record import CrosstalkRabiRecord

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


def _target_frequency_group(target: str) -> Literal["Low", "High"]:
    if not target.startswith("Q"):
        raise ValueError(f"Target must start with 'Q': {target}")
    target_mod = int(target[1:]) % 4
    if target_mod in LOW_INDEX:
        return "Low"
    if target_mod in HIGH_INDEX:
        return "High"
    raise ValueError(f"Unsupported target frequency group: {target}")


def _finite_z_range(values: NDArray[np.float64]) -> tuple[float | None, float | None]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None, None
    return float(np.min(finite_values)), float(np.max(finite_values))


def _title_with_subtitle(title: str, subtitle: str) -> dict[str, object]:
    return {"text": title, "subtitle": {"text": subtitle}}


def _format_group_values(values: dict[str, str]) -> str:
    group_order = ("Low-Low", "Low-High", "High-Low", "High-High")
    return ", ".join(f"{group}: {values[group]}" for group in group_order)


def _load_matrix_cache(path: Path | str) -> CrosstalkRabiMatrix:
    """Load a cached matrix snapshot from disk."""
    return CrosstalkRabiMatrix._load_cache(path)


def _save_matrix_cache(matrix: CrosstalkRabiMatrix, path: Path | str) -> Path:
    """Save a cached matrix snapshot to disk."""
    return matrix._save_cache(path)


def _update_matrix(
    matrix: CrosstalkRabiMatrix,
    summary: CrosstalkRabiPairSummary,
) -> None:
    """Update one matrix entry from a pair summary."""
    matrix._update(summary)


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
    def _load_cache(cls, path: Path | str) -> CrosstalkRabiMatrix:
        """Load a cached matrix snapshot from a JSON file encoded with jsonpickle."""
        file_path = Path(path)
        with file_path.open("r") as file:
            loaded = jsonpickle.decode(file.read())  # noqa: S301
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(loaded)}")
        return loaded

    @classmethod
    def from_records(
        cls,
        targets: list[str],
        records: Iterable[CrosstalkRabiRecord],
    ) -> CrosstalkRabiMatrix:
        """Build a matrix projection from pair-level records."""
        matrix = cls.create(targets)
        for record in records:
            matrix._update(record.pair_summary)
        return matrix

    def _save_cache(self, path: Path | str) -> Path:
        """Save the matrix as a cached JSON snapshot."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = jsonpickle.encode(self, unpicklable=True)
        if not isinstance(encoded, str):
            raise TypeError("jsonpickle.encode() must return a string.")
        with file_path.open("w") as file:
            file.write(encoded)
        return file_path

    def _update(self, summary: CrosstalkRabiPairSummary) -> None:
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

    def _status_summary_subtitle(self) -> str:
        group_counts = {
            "Low-Low": [0, 0],
            "Low-High": [0, 0],
            "High-Low": [0, 0],
            "High-High": [0, 0],
        }
        counted_statuses = (STATUS_MEASURED, STATUS_FIT_FAILED)

        for row, measure_target in enumerate(self.targets):
            measure_group = _target_frequency_group(_canonical_target(measure_target))
            for column, drive_target in enumerate(self.targets):
                status = int(self.status_matrix[row, column])
                if status not in counted_statuses:
                    continue

                drive_group = _target_frequency_group(_canonical_target(drive_target))
                group = f"{measure_group}-{drive_group}"
                group_counts[group][1] += 1
                if status == STATUS_MEASURED:
                    group_counts[group][0] += 1

        summary: dict[str, str] = {}
        for group, (success_count, total_count) in group_counts.items():
            if total_count == 0:
                summary[group] = "N/A (0/0)"
                continue
            rate = success_count / total_count * 100
            summary[group] = f"{rate:.1f}% ({success_count}/{total_count})"
        return "Measured rate measured/(fit_failed+measured): " + _format_group_values(
            summary
        )

    def _ratio_summary_subtitle(self, mode: Literal["ratio", "dB"] = "dB") -> str:
        group_values = {
            "Low-Low": [],
            "Low-High": [],
            "High-Low": [],
            "High-High": [],
        }

        for row, measure_target in enumerate(self.targets):
            measure_group = _target_frequency_group(_canonical_target(measure_target))
            for column, drive_target in enumerate(self.targets):
                ratio = self.r_matrix[row, column]
                if not np.isfinite(ratio):
                    continue
                if mode == "dB":
                    if ratio <= 0:
                        continue
                    ratio = 20 * np.log10(ratio)
                elif mode != "ratio":
                    raise ValueError(f"Unsupported mode: {mode}")

                drive_group = _target_frequency_group(_canonical_target(drive_target))
                group_values[f"{measure_group}-{drive_group}"].append(float(ratio))

        summary: dict[str, str] = {}
        for group, values in group_values.items():
            if not values:
                summary[group] = "N/A"
                continue
            summary[group] = f"{np.mean(values):.5g}"
        unit = "dB" if mode == "dB" else "ratio"
        return f"Mean {unit}: " + _format_group_values(summary)

    def plot_ratio_matrix(
        self,
        title: str = "Crosstalk Rabi ratio matrix",
        mode: Literal["ratio", "dB"] = "dB",
    ) -> go.Figure:
        """Render the ratio matrix as a heatmap."""
        if mode == "ratio":
            z_matrix = self.r_matrix
            colorbar_title = "r_kj"
            hover_value = "r_kj=%{z:.5f}"
        elif mode == "dB":
            z_matrix = np.full_like(self.r_matrix, np.nan, dtype=np.float64)
            np.log10(
                self.r_matrix,
                out=z_matrix,
                where=np.isfinite(self.r_matrix) & (self.r_matrix > 0),
            )
            z_matrix *= 20
            colorbar_title = "r_kj (dB)"
            hover_value = "r_kj(dB)=%{z:.2f}"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        figure_size = _matrix_figure_size(len(self.targets))
        label_matrix = np.vectorize(STATUS_LABELS.get)(self.status_matrix)
        text_size = int(np.clip(np.round(600 / max(len(self.targets), 1)), 7, 12))
        text = np.full(z_matrix.shape, "", dtype=object)
        finite_mask = np.isfinite(z_matrix)
        if mode == "dB":
            text[finite_mask] = z_matrix[finite_mask].astype(int).astype(str)
        else:
            text[finite_mask] = np.round(z_matrix[finite_mask], 2).astype(str)
        zmin, zmax = _finite_z_range(z_matrix)
        fig = viz.make_figure()
        fig.add_trace(
            go.Heatmap(
                z=z_matrix,
                x=self.targets,
                y=self.targets,
                text=text,
                customdata=label_matrix,
                texttemplate="%{text}",
                textfont={"size": text_size},
                colorscale="Cividis",
                colorbar_title=colorbar_title,
                zmin=zmin,
                zmax=zmax,
                hovertemplate=(
                    f"measure=%{{y}}<br>drive=%{{x}}<br>{hover_value}<br>status=%{{customdata}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=_title_with_subtitle(title, self._ratio_summary_subtitle(mode=mode)),
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
            title=_title_with_subtitle(title, self._status_summary_subtitle()),
            width=figure_size,
            height=figure_size,
            xaxis_title="drive target j",
            yaxis_title="measure target k",
        )
        return fig
