import numpy as np


def calculate_average(
    crc: np.ndarray,
    wct: np.ndarray,
    freqs: np.ndarray,
    resp_freq: float,
    coi: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict,
]:
    """Collapse a CRC time-frequency map into frequency and HF-band summaries.

    The function returns a frequency-wise energy profile and median CRC/WCT traces inside
    a ±0.05-Hz band around the supplied respiratory frequency. When a cone of influence is
    provided, time points are retained only when the complete HF band is outside the
    edge-contaminated region; rejected points are returned as NaN.

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
    coi
        Optional cone-of-influence period in seconds for each time point.

    Returns
    -------
    tuple
        Frequency-wise CRC energy, HF CRC trace, HF WCT trace, and metadata including
        the selected bands and cone-of-influence coverage.
    """
    # Convert inputs before validating dimensions and frequency/time alignment.
    crc = np.asarray(crc, dtype=float)
    wct = np.asarray(wct, dtype=float)
    freqs = np.asarray(freqs, dtype=float).ravel()

    # Require one coherent time-frequency grid for CRC and WCT.
    if crc.ndim != 2:
        raise ValueError("crc must be a 2D array.")
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1D array.")
    if len(freqs) != crc.shape[0]:
        raise ValueError("Length of freqs must match the number of rows in crc.")
    if wct.shape != crc.shape:
        raise ValueError("wct must have the same shape as crc.")
    if not np.isfinite(freqs).all() or np.any(freqs <= 0):
        raise ValueError("freqs must contain finite positive values.")
    if not np.isfinite(resp_freq) or resp_freq <= 0:
        raise ValueError("resp_freq must be a finite positive frequency.")

    if coi is None:
        coi_valid = np.ones(crc.shape[1], dtype=bool)
    else:
        coi = np.asarray(coi, dtype=float).ravel()
        if coi.size != crc.shape[1]:
            raise ValueError("coi must have one value per time point in crc.")
        coi_valid = np.isfinite(coi) & (coi > 0)

    # The frequency profile summarizes finite CRC energy over time.
    valid_counts = np.sum(np.isfinite(crc), axis=1)
    energy_crc = np.divide(
        np.nansum(crc, axis=1),
        valid_counts,
        out=np.full(crc.shape[0], np.nan, dtype=float),
        where=valid_counts > 0,
    )

    # Use a narrow band around the measured or expected respiration frequency.
    low_freq = [0.04, max(0.04, resp_freq - 0.05)]
    high_freq = [max(0.04, resp_freq - 0.05), resp_freq + 0.05]

    # Identify the wavelet rows included in the HF summary.
    freq_indices = np.where((freqs >= high_freq[0]) & (freqs <= high_freq[1]))[0]

    details = {
        "resp_freq": resp_freq,
        "hf_band": high_freq,
        "lf_band": low_freq,
        "n_hf_frequency_bins": int(len(freq_indices)),
    }

    # A coarse wavelet grid may not contain any requested HF rows.
    if len(freq_indices) == 0:
        return energy_crc, None, None, details

    # A median across frequency rows is less sensitive to isolated noisy bins. The cone of
    # influence then removes time points where the complete requested band is contaminated.
    coi_boundary_hz = (
        np.divide(
            1.0,
            np.asarray(coi, dtype=float),
            out=np.full_like(np.asarray(coi, dtype=float), np.inf),
            where=np.isfinite(coi) & (coi > 0),
        )
        if coi is not None
        else np.zeros(crc.shape[1], dtype=float)
    )
    if coi is not None:
        coi_valid &= coi_boundary_hz <= high_freq[0]

    avg_crc_hf = np.nanmedian(crc[freq_indices, :], axis=0)
    avg_wct_hf = np.nanmedian(wct[freq_indices, :], axis=0)
    avg_crc_hf[~coi_valid] = np.nan
    avg_wct_hf[~coi_valid] = np.nan
    details["n_coi_valid"] = int(np.count_nonzero(coi_valid))
    details["coi_valid_fraction"] = float(np.mean(coi_valid))

    return energy_crc, avg_crc_hf, avg_wct_hf, details
