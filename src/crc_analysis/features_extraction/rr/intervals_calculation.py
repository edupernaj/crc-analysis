import numpy as np
from scipy import stats


def calculate_rr_intervals(
    r_peaks: np.ndarray, time_edr: np.ndarray, fs: float = 250.0
) -> np.ndarray:
    """Turn detected R-peaks into beat-to-beat RR intervals in milliseconds.

    This function takes the places where the QRS peaks occur and subtracts one beat from the
    next. The result is the time gap between heartbeats, expressed in milliseconds. We then
    remove a few obvious outliers and interpolate the cleaned sequence back onto the EDR time axis.

    Parameters
    ----------
    r_peaks
        1D array of detected R-peak positions in samples.
    time_edr
        Time vector used to re-sample the RR trace onto the EDR grid.
    fs
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        RR interval trace in milliseconds, interpolated onto the EDR time axis.
    """
    # We need at least two beats before we can compare one heartbeat to the next.
    r_peaks = np.asarray(r_peaks)
    time_edr = np.asarray(time_edr)
    if r_peaks.size < 2:
        raise ValueError("At least two R-peaks are required to calculate RR intervals.")

    # The distance between successive beats is the RR interval in samples; convert it to ms.
    rr_intervals = np.diff(r_peaks) / fs * 1000

    # A small z-score cleanup removes pathological jumps caused by missed or false peaks.
    rr_int_clean = rr_intervals[np.abs(stats.zscore(rr_intervals)) < 3]

    # If all intervals are invalid, return a NaN-filled trace instead of letting the rest of the
    # pipeline fail on an empty signal.
    if rr_int_clean.size == 0:
        return np.full_like(time_edr, np.nan, dtype=float)

    # Create an evenly spaced array that is the same length as time_edr (this is needed because
    # you need signal of the same length for the wavelet analysis).
    rr_int_p = np.interp(
        time_edr,
        np.linspace(min(time_edr), max(time_edr), len(rr_int_clean)),
        rr_int_clean,
    )

    return rr_int_p
