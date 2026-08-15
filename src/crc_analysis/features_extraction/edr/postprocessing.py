import numpy as np
from scipy import stats


def edr_postprocessing(edr: np.ndarray, time_edr: np.ndarray) -> np.ndarray:
    """
    Post-processes the EDR signal by removing outliers and resampling.

    Parameters
    ----------
    edr : np.ndarray
        Array of EDR values.
    time_edr : np.ndarray
        Time vector for the resampled EDR waveform.

    Returns
    -------
    edr_clean : np.ndarray
        Processed EDR signal.
    """
    # Remove outliers using z-score method
    edr_clean = edr[np.abs(stats.zscore(edr)) < 3]

    # If cleaning removed all samples, return an array of NaNs matching time_edr
    if edr_clean.size == 0:
        return np.full_like(time_edr, np.nan, dtype=float)

    # Resample the cleaned EDR signal to match the time vector time_edr
    edr_resampled = np.interp(
        time_edr, np.linspace(min(time_edr), max(time_edr), len(edr_clean)), edr_clean
    )

    return edr_resampled
