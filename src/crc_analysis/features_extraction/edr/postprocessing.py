import numpy as np
from scipy import stats


def edr_postprocessing(edr: np.ndarray, time_edr: np.ndarray) -> np.ndarray:
    """Clean a respiration proxy by removing bizarre spikes and re-sampling it.

    EDR is a rough estimate of breathing, so a few obvious outliers
    can ruin the analysis. We drop values that are unusually far from the center and then
    redraw the waveform onto the target time grid so later steps see a regular signal.

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
    # Z-score filtering removes values that are more than a few standard deviations away.
    edr_clean = edr[np.abs(stats.zscore(edr)) < 3]

    # If the cleaning step removes everything, return a safe NaN-filled signal instead of
    # crashing downstream code.
    if edr_clean.size == 0:
        return np.full_like(time_edr, np.nan, dtype=float)

    # Interpolate the cleaned signal back onto the desired time axis for uniform sampling.
    edr_resampled = np.interp(
        time_edr, np.linspace(min(time_edr), max(time_edr), len(edr_clean)), edr_clean
    )

    return edr_resampled
