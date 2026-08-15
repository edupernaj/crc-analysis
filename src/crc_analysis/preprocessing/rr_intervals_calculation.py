import numpy as np
from scipy import stats


def calculate_rr_intervals(
    r_peaks: np.ndarray, tEdr: np.ndarray, fs: float = 250.0
) -> np.ndarray:
    """
    Calculate RR intervals from R-peak indices and preprocess the signal. Includes outlier removal and
    resampling

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak indices.
    fs : float
        Sampling frequency in Hz.
    tEdr : np.ndarray
        Time vector for the resampled EDR waveform.

    Returns
    -------
    rr_intervals : np.ndarray
        Array of RR intervals in milliseconds.
    """
    # Need at least two R-peaks to form RR intervals
    r_peaks = np.asarray(r_peaks)
    tEdr = np.asarray(tEdr)
    if r_peaks.size < 2:
        raise ValueError("At least two R-peaks are required to calculate RR intervals.")
    rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to milliseconds
    rr_int_clean = rr_intervals[np.abs(stats.zscore(rr_intervals)) < 3]

    # If cleaning removed all intervals, return NaNs
    if rr_int_clean.size == 0:
        return np.full_like(tEdr, np.nan, dtype=float)

    rr_int_p = np.interp(
        tEdr, np.linspace(min(tEdr), max(tEdr), len(rr_int_clean)), rr_int_clean
    )
    return rr_int_p
