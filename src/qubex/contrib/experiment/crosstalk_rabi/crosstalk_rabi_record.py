"""Internal pair-level persisted records for crosstalk Rabi measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qubex.experiment.models.experiment_record import ExperimentRecord

from .crosstalk_rabi_constants import (
    DEFAULT_DATA_DIR,
    SAVE_DESCRIPTION_TEMPLATE,
    SAVE_FILENAME,
)
from .crosstalk_rabi_result import CrosstalkRabiPairSummary


@dataclass
class CrosstalkRabiRecord:
    """Persist one drive/measure pair summary with saved result references."""

    drive_target: str
    measure_target: str
    pair_summary: CrosstalkRabiPairSummary
    jj_result_file: str | None = None
    kj_result_file: str | None = None

    def save(
        self,
        data_dir: Path | str | None = None,
    ) -> None:
        """Persist this pair-level record using the experiment record store."""
        record = ExperimentRecord(
            data=self,
            name=f"{SAVE_FILENAME}Record",
            description=SAVE_DESCRIPTION_TEMPLATE(
                self.drive_target,
                self.measure_target,
            ),
        )
        record.save(data_path=str(data_dir) if data_dir is not None else None)

    @classmethod
    def load(
        cls,
        name: str,
        *,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ) -> CrosstalkRabiRecord:
        """Load one pair-level record from the experiment record store."""
        record = ExperimentRecord.load(name, data_dir=str(data_dir))
        data = record.data
        if not isinstance(data, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(data)}")
        return data

    @classmethod
    def list(
        cls,
        *,
        data_dir: Path | str = DEFAULT_DATA_DIR,
    ) -> list[CrosstalkRabiRecord]:
        """Load all saved pair-level records from the experiment record store."""
        base_path = Path(data_dir)
        if not base_path.exists():
            return []

        loaded_records: list[tuple[str, float, str, CrosstalkRabiRecord]] = []
        for path in sorted(base_path.glob(f"*_{SAVE_FILENAME}Record_*.json")):
            loaded = ExperimentRecord.load(path.name, data_dir=str(base_path))
            data = loaded.data
            if isinstance(data, cls):
                loaded_records.append(
                    (loaded.created_at, path.stat().st_mtime, path.name, data)
                )

        loaded_records.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in loaded_records]
