"""Unified access to crosstalk-Rabi matrices and saved pair results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from qubex.experiment.models.experiment_record import ExperimentRecord
from qubex.experiment.models.experiment_result import ExperimentResult

from .crosstalk_rabi_constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_IMAGES_DIR,
    SAVE_FILENAME,
)
from .crosstalk_rabi_matrix import (
    CrosstalkRabiMatrix,
    _build_target_index,
    _canonical_target,
    _load_matrix_cache,
    _save_matrix_cache,
    _update_matrix,
)
from .crosstalk_rabi_record import CrosstalkRabiRecord
from .crosstalk_rabi_result import CrosstalkRabiPairSummary
from .crosstalk_rabi_status import (
    STATUS_LABELS,
)


@dataclass
class CrosstalkRabiCollection:
    """Provide one entry point for matrix views and saved pair-level Rabi data."""

    matrix: CrosstalkRabiMatrix
    matrix_path: Path | None = None
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    images_dir: Path = Path(DEFAULT_IMAGES_DIR)
    _records: list[CrosstalkRabiRecord] | None = None

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
            _records=[],
        )

    @classmethod
    def _load_cache(
        cls,
        matrix_path: Path | str,
        *,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ) -> CrosstalkRabiCollection:
        """Load a collection from a cached matrix snapshot."""
        path = Path(matrix_path)
        return cls(
            matrix=_load_matrix_cache(path),
            matrix_path=path,
            data_dir=Path(data_dir),
            _records=None,
        )

    @classmethod
    def load_records(
        cls,
        *,
        chip_id: str,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ) -> CrosstalkRabiCollection:
        """Load a collection by rebuilding the matrix from saved pair records."""
        if "Q64" in chip_id:
            n = 64
        else:
            n = 144

        targets = [f"Q{i:03d}" for i in range(int(n))]
        base_path = Path(data_dir)
        records = CrosstalkRabiRecord.list(data_dir=base_path)
        return cls(
            matrix=CrosstalkRabiMatrix.from_records(targets, records),
            matrix_path=None,
            data_dir=base_path,
            _records=records,
        )

    def _save_cache(self, matrix_path: Path | str | None = None) -> Path:
        """Save the current matrix as a cached snapshot and return the written path."""
        target_path = Path(matrix_path) if matrix_path is not None else self.matrix_path
        if target_path is None:
            raise ValueError("matrix_path must be provided before saving.")
        self.matrix_path = target_path
        return _save_matrix_cache(self.matrix, target_path)

    def _update(self, summary: CrosstalkRabiPairSummary) -> None:
        """Update the matrix with one pair summary."""
        _update_matrix(self.matrix, summary)

    def rebuild_matrix(self, targets: list[str] | None = None) -> CrosstalkRabiMatrix:
        """Rebuild the in-memory matrix from saved pair records."""
        matrix_targets = (
            list(targets) if targets is not None else list(self.matrix.targets)
        )
        self._records = CrosstalkRabiRecord.list(data_dir=self.data_dir)
        self.matrix = CrosstalkRabiMatrix.from_records(
            matrix_targets,
            self._records,
        )
        return self.matrix

    def pending_pairs(self) -> list[tuple[str, str]]:
        """Return drive/measure pairs that are still pending."""
        return self.matrix.pending_pairs()

    def status_table(self) -> list[dict[str, str | float | int | None]]:
        """Return the flattened matrix status table."""
        return self.matrix.status_table()

    def plot_ratio_matrix(
        self,
        title: str = "Crosstalk Rabi ratio matrix",
        mode: Literal["ratio", "dB"] = "dB",
        save_figure: bool = False,
    ) -> None:
        """Plot the stored crosstalk ratio matrix."""
        fig = self.matrix.plot_ratio_matrix(title=title, mode=mode)
        self._apply_record_hover(fig, include_ratio=True, ratio_mode=mode)
        fig.show()
        if save_figure:
            now = datetime.now().strftime("%Y%m%d")
            image_dir = self.images_dir
            image_dir.mkdir(parents=True, exist_ok=True)
            figure_path = image_dir / f"{SAVE_FILENAME}_ratio_matrix_{now}.pdf"
            fig.write_image(figure_path)
            print(f"Saved ratio matrix figure to {figure_path}")

    def plot_status_matrix(
        self,
        title: str = "Crosstalk Rabi acquisition status",
        save_figure: bool = False,
    ) -> None:
        """Plot the stored crosstalk acquisition-status matrix."""
        fig = self.matrix.plot_status_matrix(title=title)
        self._apply_record_hover(fig, include_ratio=False)
        fig.show()
        if save_figure:
            now = datetime.now().strftime("%Y%m%d")
            image_dir = self.images_dir
            image_dir.mkdir(parents=True, exist_ok=True)
            figure_path = image_dir / f"{SAVE_FILENAME}_status_matrix_{now}.pdf"
            fig.write_image(figure_path)
            print(f"Saved status matrix figure to {figure_path}")

    def load_record(
        self,
        *,
        date: str | int,
        id: int,
    ) -> CrosstalkRabiRecord:
        """Load one saved CrosstalkRabiRecord by date and file id."""
        file_name = self._build_saved_file_name(
            date=date,
            id=id,
            record_kind="record",
        )
        return CrosstalkRabiRecord.load(file_name, data_dir=self.data_dir)

    def load_result(
        self,
        *,
        date: str | int,
        id: int,
    ) -> ExperimentResult[Any]:
        """Load one saved crosstalk-Rabi ExperimentResult by date and file id."""
        file_name = self._build_saved_file_name(
            date=date,
            id=id,
            record_kind="result",
        )
        loaded = ExperimentRecord.load(file_name, data_dir=str(self.data_dir)).data
        if not isinstance(loaded, ExperimentResult):
            raise TypeError(f"Expected ExperimentResult, got {type(loaded)}")
        return loaded

    def find_result(
        self,
        drive_target: str,
        measure_target: str,
        *,
        result_kind: Literal["kj", "jj"] = "kj",
    ) -> ExperimentResult[Any]:
        """Load the saved ExperimentResult referenced by the matrix-backed pair record."""
        data_path = self.data_dir
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_path}")

        pair_record = self._find_record(
            drive_target=drive_target,
            measure_target=measure_target,
        )

        print(
            "Crosstalk Rabi matrix status: "
            f"drive_target={drive_target}, measure_target={measure_target}, "
            f"status={self._pair_status_label(drive_target, measure_target)}"
        )

        result_file = self._select_result_file(pair_record, result_kind=result_kind)
        if result_file is None:
            raise FileNotFoundError(
                f"No saved {result_kind} crosstalk-Rabi ExperimentResult is "
                "referenced by the latest pair record for "
                f"drive_target={drive_target}, measure_target={measure_target}."
            )

        saved_result = ExperimentRecord.load(result_file, data_dir=str(data_path)).data
        if not isinstance(saved_result, ExperimentResult):
            raise TypeError(f"Expected ExperimentResult, got {type(saved_result)}")
        return saved_result

    def find_result_fit(
        self,
        drive_target: str,
        measure_target: str,
        *,
        result_kind: Literal["kj", "jj"] = "kj",
        is_damped: bool = True,
    ) -> None:
        """Load and fit the saved ExperimentResult for one drive/measure pair."""
        if result_kind == "jj":
            target = f"{drive_target}-{drive_target}"
        elif result_kind == "kj":
            target = f"{measure_target}-{drive_target}"

        record = self._find_record(
            drive_target=drive_target,
            measure_target=measure_target,
        )

        result = self.find_result(
            drive_target=drive_target,
            measure_target=measure_target,
            result_kind=result_kind,
        )
        fit_result = result.fit(
            is_damped=is_damped,
            plot=False,
        )
        print(fit_result)
        figure: go.Figure = fit_result[target].get_figure()
        title = figure.layout.title.to_plotly_json()
        title["subtitle"] = {
            "text": f"Hardware Amplitude jj: {record.pair_summary.amplitude_hw_for_jj}, kj: {record.pair_summary.amplitude_hw_for_kj}",
        }
        figure.update_layout(title=title)
        figure.show()

    def plot_rabi(
        self,
        drive_target: str,
        measure_target: str,
        *,
        result_kind: Literal["kj", "jj"] = "kj",
        **kwargs,
    ) -> go.Figure | None:
        """Load and plot the latest saved Rabi result for one drive/measure pair."""
        experiment_result = self.find_result(
            drive_target=drive_target,
            measure_target=measure_target,
            result_kind=result_kind,
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

    def _build_saved_file_name(
        self,
        *,
        date: str | int,
        id: int,
        record_kind: Literal["record", "result"],
    ) -> str:
        """Build one saved JSON file name from the common crosstalk-Rabi pattern."""
        if record_kind == "record":
            return f"{date}_{SAVE_FILENAME}Record_{id}.json"
        if record_kind == "result":
            return f"{date}_{SAVE_FILENAME}_{id}.json"
        raise ValueError(f"Unsupported record_kind: {record_kind}")

    def _select_result_file(
        self,
        pair_record: CrosstalkRabiRecord,
        *,
        result_kind: Literal["kj", "jj"],
    ) -> str | None:
        """Return the saved result file referenced by the pair record."""
        if result_kind == "kj":
            return pair_record.kj_result_file
        if result_kind == "jj":
            return pair_record.jj_result_file
        raise ValueError(f"Unsupported result_kind: {result_kind}")

    def _apply_record_hover(
        self,
        fig: go.Figure,
        *,
        include_ratio: bool,
        ratio_mode: Literal["ratio", "dB"] = "ratio",
    ) -> None:
        """Attach latest record metadata to matrix hover when record data is loaded."""
        if self._records is None or not fig.data:
            return

        fig.data[0].customdata = self._record_hover_customdata()
        if include_ratio:
            if ratio_mode == "ratio":
                ratio_hover = "r_kj=%{z:.5f}"
            elif ratio_mode == "dB":
                ratio_hover = "r_kj(dB)=%{z:.2f}"
            else:
                raise ValueError(f"Unsupported ratio_mode: {ratio_mode}")

            fig.data[0].hovertemplate = (
                "measure=%{y}<br>"
                "drive=%{x}<br>"
                f"{ratio_hover}<br>"
                "status=%{customdata[0]}<br>"
                "jj_fit_status=%{customdata[1]}<br>"
                "kj_fit_status=%{customdata[2]}<br>"
                "jj_frequency=%{customdata[3]}<br>"
                "kj_frequency=%{customdata[4]}<br>"
                "record_warning=%{customdata[5]}<br>"
                "jj_result_file=%{customdata[6]}<br>"
                "kj_result_file=%{customdata[7]}<extra></extra>"
            )
            return

        fig.data[0].hovertemplate = (
            "measure=%{y}<br>"
            "drive=%{x}<br>"
            "status=%{customdata[0]}<br>"
            "jj_fit_status=%{customdata[1]}<br>"
            "kj_fit_status=%{customdata[2]}<br>"
            "jj_frequency=%{customdata[3]}<br>"
            "kj_frequency=%{customdata[4]}<br>"
            "record_warning=%{customdata[5]}<br>"
            "jj_result_file=%{customdata[6]}<br>"
            "kj_result_file=%{customdata[7]}<extra></extra>"
        )

    def _record_hover_customdata(self) -> np.ndarray:
        """Build per-cell hover metadata from the latest record for each pair."""
        customdata = np.empty(
            (len(self.matrix.targets), len(self.matrix.targets), 8),
            dtype=object,
        )
        customdata[:] = "-"

        latest_records = self._latest_record_map()
        target_index = _build_target_index(self.matrix.targets)

        for (drive_target, measure_target), record in latest_records.items():
            try:
                row = target_index[_canonical_target(measure_target)]
                column = target_index[_canonical_target(drive_target)]
            except KeyError:
                continue

            summary = record.pair_summary
            customdata[row, column] = [
                summary.status,
                self._stringify_hover_value(summary.jj_fit_status),
                self._stringify_hover_value(summary.kj_fit_status),
                self._stringify_hover_value(summary.jj_frequency),
                self._stringify_hover_value(summary.kj_frequency),
                self._stringify_hover_value(summary.warning),
                self._stringify_hover_value(record.jj_result_file),
                self._stringify_hover_value(record.kj_result_file),
            ]

        return customdata

    def _latest_record_map(self) -> dict[tuple[str, str], CrosstalkRabiRecord]:
        """Return the latest loaded record for each drive/measure pair."""
        latest_records: dict[tuple[str, str], CrosstalkRabiRecord] = {}
        if self._records is None:
            return latest_records

        for record in self._records:
            latest_records[(record.drive_target, record.measure_target)] = record
        return latest_records

    def _stringify_hover_value(self, value: Any) -> str:
        """Convert optional hover metadata into stable human-readable strings."""
        if value is None:
            return "-"
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.6g}"
        if hasattr(value, "name"):
            return str(value.name).lower()
        return str(value)

    def _find_record(
        self,
        drive_target: str,
        measure_target: str,
    ) -> CrosstalkRabiRecord:
        """Return the latest pair record used to build the current matrix state."""
        if self._records is None:
            raise ValueError(
                "find_result() requires a record-backed collection. "
                "Use load_records() or rebuild_matrix() before loading results."
            )

        for record in reversed(self._records):
            if (
                record.drive_target == drive_target
                and record.measure_target == measure_target
            ):
                return record

        raise FileNotFoundError(
            "No crosstalk-Rabi pair record found for "
            f"drive_target={drive_target}, measure_target={measure_target}."
        )
