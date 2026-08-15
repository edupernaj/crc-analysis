from dataclasses import dataclass
from pathlib import Path
import numpy as np

from crc_analysis.input.csv_reader import read_ecg12_data
from crc_analysis.input.filter import filter_ecg12


class DataLoaderEcg12:
    """
    Central interface for accessing ECG data from an experiment.

    The loader expects the data to be organised by experiment, session,
    and experimental condition. It discovers the available sessions
    automatically and provides asccess to their conditions and ECG data.

    Example:
        experiment = DataLoaderEcg12("experiment_name")
        baseline = experiment["1"]["Baseline"]
        ecg = baseline.ecg
    """

    def __init__(
        self,
        experiment_name: str,
        data_root: str | Path = "data",
    ):
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
        return list(self._sessions.keys())

    def __getitem__(self, session: str | int) -> Session:
        return self._sessions[str(session)]


@dataclass
class Session:
    """
    Represents one experimental session.

    A session contains one or more experimental conditions, which are
    discovered automatically from the session directory.
    """

    number: str
    path: Path

    def __post_init__(self):
        self._conditions = {
            folder.name: Condition(folder.name, folder)
            for folder in sorted(self.path.iterdir())
            if folder.is_dir()
        }

    @property
    def conditions(self) -> list[str]:
        return list(self._conditions.keys())

    def __getitem__(self, condition: str) -> Condition:
        return self._conditions[condition]


@dataclass
class Condition:
    """
    Represents one experimental condition within a session.

    A condition contains the ECG measurement file and, optionally,
    an XML file containing additional measurement information.
    """

    name: str
    path: Path

    @property
    def ecg_file(self) -> Path:
        """Return the ECG TXT file for this condition."""
        files = list(self.path.glob("*.txt"))

        if not files:
            raise FileNotFoundError(f"No ECG TXT file found in {self.path}")

        return files[0]

    @property
    def xml_file(self) -> Path | None:
        """Return the XML file for this condition."""
        files = list(self.path.glob("*.xml"))
        return files[0] if files else None

    @property
    def _input_data(self) -> dict:
        """
        Load the raw ECG data and associated sampling information.

        The data is read using the central ECG12 reader and cached conceptually
        as the input data for subsequent steps.
        """
        return read_ecg12_data(self.ecg_file)

    @property
    def fs(self) -> int:
        """Return the original ECG sampling frequency in Hz."""
        return self._input_data["fs"]

    @property
    def fs_new(self) -> int:
        """Return the target sampling frequency used for further processing in Hz."""
        return self._input_data["fs_new"]

    @property
    def ecg(self) -> np.ndarray:
        """
        Return the filtered ECG data.

        The raw ECG signal is passed through the ECG12 filtering pipeline
        using the original sampling frequency.
        """
        return filter_ecg12(self._input_data["ecg12"], self.fs)

    @property
    def metadata(self) -> str:
        """Return the contents of the TXT file."""
        return self.ecg_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
