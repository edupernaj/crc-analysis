import numpy as np
from scipy import stats


def edr_postprocessing(
    edr: np.ndarray,
    time_edr: np.ndarray,
) -> np.ndarray:
    """Remove EDR outliers while preserving the original time axis.

    Outliers are marked as missing and interpolated on the original ``time_edr`` grid;
    no samples are deleted and no replacement time axis is constructed.

    Samples that are clearly abnormal are marked as invalid and are then
    linearly interpolated from their nearest valid neighbours. At the beginning
    or end of the signal, the nearest valid value is carried to the edge.

    Outliers are detected using a robust median/MAD rule. This is less sensitive
    to extreme values than a conventional mean/standard-deviation z-score.

    Parameters
    ----------
    edr
        EDR waveform sampled on ``time_edr``.
    time_edr
        Time vector in seconds corresponding one-to-one with ``edr``.

    Returns
    -------
    np.ndarray
        Cleaned EDR waveform on the exact same time grid as the input.

    Raises
    ------
    ValueError
        If the arrays have different lengths, the time vector is invalid,
        or the time vector is not strictly increasing.
    """
    edr = np.asarray(edr, dtype=float).ravel()
    time_edr = np.asarray(time_edr, dtype=float).ravel()

    if edr.size != time_edr.size:
        raise ValueError("edr and time_edr must have the same length.")
    if edr.size == 0:
        raise ValueError("edr and time_edr must not be empty.")
    if not np.all(np.isfinite(time_edr)):
        raise ValueError("time_edr must contain only finite values.")
    if np.any(np.diff(time_edr) <= 0):
        raise ValueError("time_edr must be strictly increasing.")

    # Identify samples that are already finite.
    finite = np.isfinite(edr)
    if not np.any(finite):
        return np.full_like(edr, np.nan, dtype=float)
    finite_values = edr[finite]

    # Robust centre and scale:
    #   MAD = median absolute deviation.
    #   For normally distributed data:
    #     robust_z ≈ 0.6745 * (x - median) / MAD
    #   A threshold of 3.5 is a conventional robust outlier criterion.
    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    mad_norm_constant = float(stats.norm.ppf(0.75))
    threshold = 3.5
    if mad > np.finfo(float).eps:
        robust_z = mad_norm_constant * (edr - median) / mad
        valid = finite & (np.abs(robust_z) < threshold)
    else:
        # If MAD is zero, the signal is either constant or nearly constant.
        # Fall back to an SD-based rule.
        std = float(np.std(finite_values))
        if std > np.finfo(float).eps:
            valid = finite & (np.abs(edr - median) < 3.0 * std)
        else:
            # Completely constant signal: every finite value is acceptable.
            valid = finite.copy()

    if not np.any(valid):
        return np.full_like(edr, np.nan, dtype=float)

    valid_times = time_edr[valid]
    valid_values = edr[valid]

    if valid_times.size == 1:
        return np.full_like(edr, valid_values[0], dtype=float)

    # Interpolate invalid points using their ORIGINAL timestamps.
    # The time axis itself is never modified.
    cleaned = np.interp(
        time_edr,
        valid_times,
        valid_values,
        left=valid_values[0],
        right=valid_values[-1],
    )

    return cleaned
