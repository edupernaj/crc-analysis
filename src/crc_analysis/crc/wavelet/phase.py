import numpy as np


def calculate_phase(
    CRC: np.ndarray,
    awct: np.ndarray,
    freqs: np.ndarray,
    hf_band: list[float],
    coi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate phase delay and directionality within a selected HF band.

    The wavelet transform gives a local phase difference between the RR signal and the
    respiratory surrogate at each time and frequency. If that phase delay is consistent,
    it indicates a stable lead/lag relationship rather than mere correlation from noise.
    Phase is averaged circularly over the requested band, while the dominant CRC
    frequency at each time point determines the phase-to-delay conversion. Optional cone
    of influence masking excludes edge-contaminated time points. The directionality index
    is a NaN-aware moving average of the delay sign.

    Parameters
    ----------
    CRC
        Time-frequency CRC map used to identify the dominant coupling frequency.
    awct
        Complex wavelet phase matrix for the same time-frequency grid.
    freqs
        Frequency axis aligned with the rows of ``CRC`` and ``awct``.
    hf_band
        Lower and upper bounds of the respiratory band to analyze.
    coi
        Optional cone-of-influence period in seconds for each time point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time delay estimate `tau` and a smoothed directionality index `DI`.
        - `tau` is a 1D array of CRC time delay between RR-intervals and respiratory signal.
            Positive tau values represent cardiac signal leading respiratory activity
            and negative tau values mean respiratory activity leading cardiac signals.
        - `DI` is a 1D directionality index. Positive values indicate predominantly
          positive delay, negative values negative delay, and values near zero mixed signs.
    """
    CRC = np.asarray(CRC, dtype=float)
    awct = np.asarray(awct, dtype=float)
    freqs = np.asarray(freqs, dtype=float).ravel()
    if CRC.ndim != 2 or awct.shape != CRC.shape:
        raise ValueError("CRC and awct must be matching 2-D arrays.")
    if freqs.size != CRC.shape[0]:
        raise ValueError("freqs must match the frequency dimension of CRC.")
    if len(hf_band) != 2 or not (0 < hf_band[0] < hf_band[1]):
        raise ValueError("hf_band must be an increasing positive frequency interval.")

    frequency_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    if not np.any(frequency_mask):
        return np.full(CRC.shape[1], np.nan), np.full(CRC.shape[1], np.nan)

    phase_band = awct[frequency_mask, :]
    crc_band = CRC[frequency_mask, :]
    band_freqs = freqs[frequency_mask]
    tau_score = np.full(CRC.shape[1], np.nan, dtype=float)

    # Work column by column so isolated bad pixels do not invalidate an entire recording. The
    # selected frequency is indexed within the same HF subset used for phase estimation.
    for time_idx in range(CRC.shape[1]):
        finite_phase = np.isfinite(phase_band[:, time_idx])
        finite_crc = np.isfinite(crc_band[:, time_idx])
        if not finite_phase.any() or not finite_crc.any():
            continue
        phase_values = phase_band[finite_phase, time_idx]
        wave_phase = np.arctan2(
            np.mean(np.sin(phase_values)), np.mean(np.cos(phase_values))
        )
        peak_idx = int(
            np.nanargmax(np.where(finite_crc, crc_band[:, time_idx], np.nan))
        )
        tau_score[time_idx] = wave_phase / (2 * np.pi * band_freqs[peak_idx])

    if coi is not None:
        coi = np.asarray(coi, dtype=float).ravel()
        if coi.size != CRC.shape[1]:
            raise ValueError("coi must have one value per time point.")
        coi_boundary_hz = np.divide(
            1.0,
            coi,
            out=np.full_like(coi, np.inf),
            where=np.isfinite(coi) & (coi > 0),
        )
        tau_score[~(coi_boundary_hz <= hf_band[0])] = np.nan

    # Smooth delay signs while ignoring NaN edge and quality-mask samples.
    W_samp = 15 * 4 + 1
    di_score = _moving_average(np.sign(tau_score), W_samp)

    return tau_score, di_score


def _moving_average(a, n=3):
    """Apply a centered, NaN-aware moving average to a one-dimensional signal."""
    if n < 1:
        raise ValueError("The moving-average window must be positive.")
    if n % 2 == 0:
        n += 1
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return a.copy()
    n = min(n, a.size if a.size % 2 else a.size - 1)
    n = max(n, 1)
    finite = np.isfinite(a)
    values = np.where(finite, a, 0.0)
    numerator = np.convolve(values, np.ones(n), mode="same")
    denominator = np.convolve(finite.astype(float), np.ones(n), mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
