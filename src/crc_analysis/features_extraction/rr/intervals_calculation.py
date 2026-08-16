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


def calculate_rr_intervals2(
    r_peaks: np.ndarray,
    time_edr: np.ndarray,
    fs: float = 250.0,
) -> np.ndarray:
    """Interpolate beat-to-beat RR intervals onto the EDR time grid.

    Each RR interval is associated with the time of the *second* R peak that
    defines that interval. This preserves the real cardiac timing instead of
    assigning the cleaned intervals to an artificial, evenly spaced time axis
    (which is what `calculate_rr_intervals` does instead).

    RR intervals are expressed in milliseconds. Intervals with non-finite or
    implausible values are treated as missing and linearly interpolated using
    their original cardiac timestamps. The output is therefore aligned to
    ``time_edr`` while retaining the original temporal information.

    Parameters
    ----------
    r_peaks
        1-D array containing R-peak sample indices in the original ECG signal.
    time_edr
        Target time grid, in seconds, onto which the RR series is interpolated.
    fs
        Original ECG sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        RR interval series in milliseconds sampled at ``time_edr``.

    Raises
    ------
    ValueError
        If there are fewer than two R peaks, ``fs`` is not positive, or the
        target time vector is invalid.
    """
    r_peaks = np.asarray(r_peaks, dtype=float).ravel()
    time_edr = np.asarray(time_edr, dtype=float).ravel()

    if r_peaks.size < 2:
        raise ValueError("At least two R-peaks are required to calculate RR intervals.")

    if fs <= 0:
        raise ValueError("fs must be positive.")

    if time_edr.size == 0:
        raise ValueError("time_edr must contain at least one time point.")

    if not np.all(np.isfinite(r_peaks)):
        raise ValueError("r_peaks must contain only finite values.")

    if np.any(np.diff(r_peaks) <= 0):
        raise ValueError("r_peaks must be strictly increasing.")

    # ------------------------------------------------------------------
    # Calculate the actual RR intervals.
    #
    # RR_i = R_i - R_(i-1)
    #
    # The timestamp of RR_i is assigned to R_i, i.e. the second beat
    # defining the interval. This preserves the actual cardiac timing.
    # ------------------------------------------------------------------
    rr_times = r_peaks[1:] / fs
    rr_intervals = np.diff(r_peaks) / fs * 1000.0

    # ------------------------------------------------------------------
    # Remove physiologically implausible intervals.
    #
    # These limits are deliberately broad:
    #   250 ms  -> 240 bpm
    #   2500 ms -> 24 bpm
    #
    # They are intended as a safety filter rather than as a clinical
    # definition of a valid RR interval.
    # ------------------------------------------------------------------
    valid = (
        np.isfinite(rr_intervals) & (rr_intervals >= 250.0) & (rr_intervals <= 2500.0)
    )

    if np.count_nonzero(valid) == 0:
        return np.full(time_edr.shape, np.nan, dtype=float)

    valid_times = rr_times[valid]
    valid_rr = rr_intervals[valid]

    # Make sure interpolation receives a strictly increasing x-axis.
    unique_times, unique_idx = np.unique(
        valid_times,
        return_index=True,
    )

    valid_times = unique_times
    valid_rr = valid_rr[unique_idx]

    if valid_times.size == 1:
        # One valid interval cannot define a time-varying signal.
        # Returning a constant is preferable to inventing a temporal trend.
        return np.full(
            time_edr.shape,
            valid_rr[0],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # We interpolate using the REAL RR timestamps.
    #
    # We do NOT do:
    #
    # np.linspace(min(time_edr), max(time_edr), len(rr_clean))
    #
    # because removing one RR interval must not shift all subsequent
    # intervals in time.
    # ------------------------------------------------------------------
    rr_interpolated = np.interp(
        time_edr,
        valid_times,
        valid_rr,
        left=valid_rr[0],
        right=valid_rr[-1],
    )

    return rr_interpolated
