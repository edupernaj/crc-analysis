import numpy as np


def calculate_average(
    crc: np.ndarray,
    wct: np.ndarray,
    freqs: np.ndarray,
    resp_freq: float,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict,
]:
    """Collapse the time-frequency CRC map into a simpler band-based summary.

    A full CRC map is rich but difficult to read at a glance. This helper reduces the
    problem to a few interpretable summaries: the overall frequency profile and the
    average coupling strength inside a narrow band near the expected respiratory rhythm.
    That makes it easier to inspect whether coupling is concentrated where physiology
    suggests it should be, without having to inspect every time-frequency pixel.

    Parameters
    ----------
    crc
        Two-dimensional CRC map with shape (n_freqs, n_time_points).
    wct
        Wavelet coherence matrix aligned to the same time-frequency grid.
    freqs
        Frequency axis of the wavelet coefficients, one value per row of ``crc``.
    resp_freq
        Reference respiratory frequency around which the high-frequency band is chosen.

    Returns
    -------
    tuple
        A four-element tuple containing the average CRC spectrum, the respiration-band
        average CRC trace, the corresponding coherence trace, and diagnostic metadata.
    """
    # Convert the inputs to NumPy arrays so the indexing and shape checks are predictable.
    crc = np.asarray(crc)
    freqs = np.asarray(freqs)

    # Check if crc is a 2D array and freqs is a 1D array
    if crc.ndim != 2:
        raise ValueError("crc must be a 2D array.")
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1D array.")
    if len(freqs) != crc.shape[0]:
        raise ValueError("Length of freqs must match the number of rows in crc.")

    # The first summary is a frequency-wise average over time. This removes the time axis
    # and highlights which frequencies carry the strongest CRC energy overall.
    energy_crc = crc.mean(axis=1)

    # Calculate the resp_freq (this is not needed now, might be needed if we want to
    # allow `None` as parameter value for resp_freq)
    _ = np.where((freqs >= 0.1) & (freqs <= 0.4), energy_crc, 0)

    # We then focus on a narrow band around the expected respiratory frequency. This keeps
    # the interpretation aligned with the physiology while reducing the sensitivity to
    # unrelated low-frequency drift or isolated noise bursts.
    low_freq, high_freq = [
        [0.04, resp_freq - 0.05],
        [resp_freq - 0.05, resp_freq + 0.05],
    ]

    # Identify indices corresponding to the specified frequency band
    freq_indices = np.where((freqs >= high_freq[0]) & (freqs <= high_freq[1]))[0]

    details = {
        "resp_freq": resp_freq,
        "hf_band": high_freq,
        "lf_band": low_freq,
    }

    # If no frequencies fall within the specified band, return NaN
    if len(freq_indices) == 0:
        return energy_crc, None, None, details

    # A median across the band is more robust than a mean when the data contain local
    # fluctuations or isolated artifacts in a small subset of frequency bins.
    avg_crc_hf = np.median(crc[freq_indices, :], axis=0)
    avg_wct_hf = np.median(wct[freq_indices, :], axis=0)

    return energy_crc, avg_crc_hf, avg_wct_hf, details
