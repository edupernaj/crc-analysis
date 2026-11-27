import numpy as np


def calculate_rr_intervals(
    r_peaks: np.ndarray,
    time_edr: np.ndarray,
    fs: float = 250.0,
) -> np.ndarray:
    """Interpolate beat-to-beat RR intervals onto the EDR time grid.

    Each RR interval is associated with the time of the *second* R peak that
    defines that interval. This preserves real cardiac timing when invalid intervals
    are excluded.

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
    # Remove physiologically implausible intervals using broad safety limits.
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

    # Interpolate using the real RR timestamps so removing one interval does not shift all
    # subsequent intervals in time.
    rr_interpolated = np.interp(
        time_edr,
        valid_times,
        valid_rr,
        left=valid_rr[0],
        right=valid_rr[-1],
    )

    return rr_interpolated
