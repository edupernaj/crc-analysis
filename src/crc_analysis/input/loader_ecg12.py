from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import numpy as np

from crc_analysis.input.csv_reader import read_ecg12_data
from crc_analysis.input.filter import filter_ecg12


class DataLoaderEcg12:
    """Index an experiment directory as deterministic sessions and conditions.

    Raw ECG loading and filtering are cached at condition level so a pipeline does not
    repeatedly read and filter the same field recording.
    """

    def __init__(
        self,
        experiment_name: str,
        data_root: str | Path = "data",
    ):
        # Each child directory is treated as one recording session.
        self.name = experiment_name
        self.path = Path(data_root) / experiment_name

        if not self.path.exists():
            raise FileNotFoundError(f"Experiment not found: {self.path}")

        self._sessions = {
            folder.name: Session(folder.name, folder)
            for folder in sorted(self.path.iterdir())
            if folder.is_dir()
        }

    @property
    def sessions(self) -> list[str]:
        """Return the names of all sessions found under the experiment directory."""
        return list(self._sessions.keys())

    def __getitem__(self, session: str | int) -> Session:
        """Fetch a session by name or numeric index."""
        return self._sessions[str(session)]


@dataclass
class Session:
    """Represent one participant/session and its available recording conditions."""

    number: str
    path: Path

    def __post_init__(self):
        # Keep condition discovery sorted so batch processing is reproducible.
        self._conditions = {
            folder.name: Condition(folder.name, folder)
            for folder in sorted(self.path.iterdir())
            if folder.is_dir()
        }

    @property
    def conditions(self) -> list[str]:
        """Return the available conditions within this session."""
        return list(self._conditions.keys())

    def __getitem__(self, condition: str) -> Condition:
        """Fetch one condition by name."""
        return self._conditions[condition]


@dataclass
class Condition:
    """One ECG recording condition inside a session."""

    name: str
    path: Path

    @property
    def ecg_file(self) -> Path:
        """Return the sole ECG text file or fail clearly when selection is ambiguous."""
        files = sorted(self.path.glob("*.txt"))

        if not files:
            raise FileNotFoundError(f"No ECG TXT file found in {self.path}")
        if len(files) > 1:
            raise ValueError(
                f"Multiple ECG TXT files found in {self.path}; "
                "select the intended recording explicitly: "
                f"{[file.name for file in files]}"
            )

        return files[0]

    @property
    def xml_file(self) -> Path | None:
        """Return the first optional XML metadata file, if present."""
        files = list(self.path.glob("*.xml"))
        return files[0] if files else None

    @cached_property
    def _input_data(self) -> dict:
        """Read and validate the raw ECG table once for this condition."""
        return read_ecg12_data(self.ecg_file)

    @cached_property
    def fs(self) -> int:
        """Return the configured original ECG sampling rate in Hz."""
        return self._input_data["fs"]

    @cached_property
    def ecg(self) -> np.ndarray:
        """Return the cached, filtered 12-lead ECG for feature extraction."""
        return filter_ecg12(self._input_data["ecg12"], self.fs)

    @property
    def metadata(self) -> str:
        """Return the ECG file text for legacy metadata inspection."""
        return self.ecg_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
