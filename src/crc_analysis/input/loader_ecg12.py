from dataclasses import dataclass
from pathlib import Path
import numpy as np

from crc_analysis.input.csv_reader import read_ecg12_data
from crc_analysis.input.filter import filter_ecg12


class DataLoaderEcg12:
    """Index the experiment directory as a set of sessions and conditions.

    This class provides a lightweight data registry for the project folder structure.
    Rather than manually navigating the file tree, the analysis code can ask for a
    specific session or condition by name, which keeps the experimental metadata easy to
    inspect and reduces the risk of path mistakes during signal processing.
    """

    def __init__(
        self,
        experiment_name: str,
        data_root: str | Path = "data",
    ):
        # The experiment folder sits at the top of the hierarchy. Each child directory is a
        # distinct recording session, which is useful because different sessions are often
        # collected at different times or under slightly different conditions.
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
    """Represent one experimental session and the conditions recorded within it."""

    number: str
    path: Path

    def __post_init__(self):
        # Each session is a container for multiple measurement conditions, such as baseline,
        # recovery, or a protocol-specific intervention. Grouping them here makes the study
        # structure explicit and keeps the downstream analysis predictable.
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
    """One measurement condition, such as baseline or recovery, inside a session."""

    name: str
    path: Path

    @property
    def ecg_file(self) -> Path:
        """Find the ECG data file for this condition."""
        files = list(self.path.glob("*.txt"))

        if not files:
            raise FileNotFoundError(f"No ECG TXT file found in {self.path}")

        return files[0]

    @property
    def xml_file(self) -> Path | None:
        """Optional metadata file if additional study information is stored alongside the ECG."""
        files = list(self.path.glob("*.xml"))
        return files[0] if files else None

    @property
    def _input_data(self) -> dict:
        """Read the raw ECG table and sampling metadata in a single place."""
        return read_ecg12_data(self.ecg_file)

    @property
    def fs(self) -> int:
        """Original sampling rate of the raw ECG trace, in Hz."""
        return self._input_data["fs"]

    @property
    def ecg(self) -> np.ndarray:
        """Return the cleaned ECG trace ready for feature extraction."""
        return filter_ecg12(self._input_data["ecg12"], self.fs)

    @property
    def metadata(self) -> str:
        """Read any supplementary text metadata written next to the ECG file."""
        return self.ecg_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
