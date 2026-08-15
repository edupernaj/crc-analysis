from pathlib import Path

from scipy import stats
import numpy as np
import pandas as pd


def read_csv_data_vcg(filepath: str, fs_new: int = 4) -> dict:
    """
    Reads CSV data from csv file and returnes a dictionary containing the rr_intervals,
    respiratory surrogate, sampling frequency, resampled frequency, time array.
    It's specific to data from VCG calculation

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the data.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    data = pd.read_csv(filepath)
    time = np.arange(
        0, sum(data["RR_ms"].dropna() / 1000), 1 / fs_new
    )  # sampled time using new fs
    rr_intervals = data["RR_ms"].fillna(data["RR_ms"].dropna().mean())
    time_rr = np.cumsum(rr_intervals / 1000)  # time of each RR interval
    edr = data["Rpeak_ca"]
    edr_p = np.interp(time, time_rr, edr)
    rr_intervals_clean = rr_intervals[np.abs(stats.zscore(rr_intervals)) < 3]
    rr_int_p = np.interp(
        time,
        np.linspace(min(time_rr), max(time_rr), len(rr_intervals_clean)),
        rr_intervals_clean,
    )

    return {"rr_intervals": rr_int_p, "edr": edr_p, "fs_new": fs_new, "time": time}
