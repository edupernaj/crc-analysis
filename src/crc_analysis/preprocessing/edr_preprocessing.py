import numpy as np
from scipy import stats


def preprocessing_edr(edr: np.ndarray, tEdr: np.ndarray) -> np.ndarray:
    """
    Preprocess the EDR signal by removing outliers and resampling.

    Parameters
    ----------
    edr : np.ndarray
        Array of EDR values.
    tEdr : np.ndarray
        Time vector for the resampled EDR waveform.

    Returns
    -------
    edr_clean : np.ndarray
        Preprocessed EDR signal.
    """
    # Remove outliers using z-score method
    edr_clean = edr[np.abs(stats.zscore(edr)) < 3]

    # If cleaning removed all samples, return an array of NaNs matching tEdr
    if edr_clean.size == 0:
        return np.full_like(tEdr, np.nan, dtype=float)

    # Resample the cleaned EDR signal to match the time vector tEdr
    edr_resampled = np.interp(
        tEdr, np.linspace(min(tEdr), max(tEdr), len(edr_clean)), edr_clean
    )

    return edr_resampled
