import numpy as np


def calculate_average(
    crc: np.ndarray, wct: np.ndarray, freqs: np.ndarray, resp_freq=0.2
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    """
    Calculate the average CRC value within a frequency band around frequency of maximum CRC component.

    Parameters
    ----------
    crc : np.ndarray
        2D array of CRC values.
    wct : np.ndarray
        2D array of Wavelet Coherence Transform
    freqs : np.ndarray
        Array of frequencies corresponding to the CRC values.
    resp_freq : np.ndarray
        Frequency with the maximum average CRC.

    Returns
    -------
    energy_crc : np.ndarray
        Array of energy CRC values for each frequency.
    avg_crc_hf : float
        Average CRC value within the specified frequency band.
    details : dict
        Dictionary containing additional details about the frequency bands.
    """
    # Ensure crc and freqs are numpy arrays
    crc = np.asarray(crc)
    freqs = np.asarray(freqs)

    # Check if crc is a 2D array and freqs is a 1D array
    if crc.ndim != 2:
        raise ValueError("crc must be a 2D array.")
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1D array.")
    if len(freqs) != crc.shape[0]:
        raise ValueError("Length of freqs must match the number of rows in crc.")

    energy_crc = crc.mean(axis=1)  # Average CRC across time for each frequency
    en_CRC_respband = np.where((freqs >= 0.1) & (freqs <= 0.4), energy_crc, 0)

    # Define frequency bands around the frequency of maximum CRC component
    LF, HF = [
        [0.04, resp_freq - 0.05],
        [resp_freq - 0.05, resp_freq + 0.05],
    ]

    # Identify indices corresponding to the specified frequency band
    freq_indices = np.where((freqs >= HF[0]) & (freqs <= HF[1]))[0]

    details = {
        "resp_freq": resp_freq,
        "HF_band": HF,
        "LF_band": LF,
    }

    # If no frequencies fall within the specified band, return NaN
    if len(freq_indices) == 0:
        return energy_crc, None, None, details

    # Calculate the average CRC value within the specified frequency band
    avg_crc_hf = np.median(crc[freq_indices, :], axis=0)
    avg_wct_hf = np.median(wct[freq_indices, :], axis=0)
    return energy_crc, avg_crc_hf, avg_wct_hf, details
