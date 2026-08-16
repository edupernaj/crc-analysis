import pandas as pd
import numpy as np

from pathlib import Path
from scipy import stats


def read_ecg12_data(filepath: Path, fs: int = 250) -> dict:
    """Load a 12-lead ECG CSV and return the data in a simple analysis-friendly form.

    Parameters
    ----------
    filepath
        Path to the CSV file containing the numeric ECG table.
    fs
        Sampling frequency in Hz used to build the time axis.

    Returns
    -------
    dict
        Dictionary with the raw pandas table, sampling rate, resampled rate,
        time vector, and the ECG matrix stored as 12-by-n samples.

    In plain English: this takes a CSV saved from the lab or a data export, turns it
    into a NumPy array, builds a time axis, and checks that the file actually contains
    12 ECG leads. If the layout is odd, we transpose it so the code can work with the
    expected shape.
    """
    fs_new = 4  # We resample the signal to a lower rate for downstream analysis.
    data = pd.read_table(filepath, sep=";")

    # Build a time vector from sample count and the sample rate.
    time = np.arange(0, len(data["V1"])) / fs

    # Keep the raw table plus the numeric ECG matrix for downstream processing.
    ecg12 = data.values
    if ecg12.shape[0] != 12 and ecg12.shape[1] != 12:
        raise ValueError("Input CSV must contain 12 leads of ECG data.")
    if ecg12.shape[1] == 12 and ecg12.shape[0] != 12:
        ecg12 = np.transpose(ecg12)

    return {"data": data, "fs": fs, "fs_new": fs_new, "time": time, "ecg12": ecg12}


def read_vcg_data(filepath: Path, fs_new: int = 4) -> dict:
    """Read a VCG-style CSV and reconstruct a time-aligned RR/EDR signal pair.

    Parameters
    ----------
    filepath
        CSV file containing RR timing and respiratory surrogate values.
    fs_new
        New sampling frequency in Hz used to interpolate the signals.

    Returns
    -------
    dict
        A dictionary with interpolated RR intervals, an EDR surrogate, the new
        sampling rate, and the corresponding time axis.

    This is a convenience loader for datasets where RR intervals and a respiratory
    surrogate are stored in a table. We turn the stored interval values into a time
    axis, interpolate the surrogate onto that time grid, and drop obvious outliers.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = pd.read_csv(filepath)

    # We build a new evenly spaced time vector based on the total RR duration.
    time = np.arange(0, sum(data["RR_ms"].dropna() / 1000), 1 / fs_new)

    # A small outlier cleanup keeps sudden impossible RR values from skewing the signal.
    rr_intervals = data["RR_ms"].fillna(data["RR_ms"].dropna().mean())
    time_rr = np.cumsum(rr_intervals / 1000)  # Each RR interval adds a little time.
    edr = data["Rpeak_ca"]

    # Interpolate the respiratory surrogate onto a fixed sampling grid.
    edr_p = np.interp(time, time_rr, edr)
    rr_intervals_clean = rr_intervals[np.abs(stats.zscore(rr_intervals)) < 3]
    rr_int_p = np.interp(
        time,
        np.linspace(min(time_rr), max(time_rr), len(rr_intervals_clean)),
        rr_intervals_clean,
    )

    return {"rr_intervals": rr_int_p, "edr": edr_p, "fs_new": fs_new, "time": time}
