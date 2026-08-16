import numpy as np


def calculate_phase(
    CRC: np.ndarray, awct: np.ndarray, freqs: np.ndarray, HF_band: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the delay and directionality of coupling within a chosen frequency band.

    The wavelet transform gives a local phase difference between the RR signal and the
    respiratory surrogate at each time and frequency. If that phase delay is consistent,
    it indicates a stable lead/lag relationship rather than mere correlation from noise.
    We first average the phase information inside the respiratory band, then convert the
    phase shift into a time delay. The directionality index is then smoothed to reduce
    short-term fluctuations and make the temporal trend easier to interpret.

    Parameters
    ----------
    CRC
        Time-frequency CRC map used to identify the dominant coupling frequency.
    awct
        Complex wavelet phase matrix for the same time-frequency grid.
    freqs
        Frequency axis aligned with the rows of ``CRC`` and ``awct``.
    HF_band
        Lower and upper bounds of the respiratory band to analyze.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time delay estimate `tau` and a smoothed directionality index `DI`.
        - `tau` is a 1D array of CRC time delay between RR-intervals and respiratory signal.
            Positive tau values represent cardiac signal leading respiratory activity
            and negative tau values mean respiratory activity leading cardiac signals.
        - `DI` is a 1D array of the Directionality Index. +1 indicate positive delay,
            -1 indicate negative delay, around 0 means mixed directions or balancing.
    """
    # Restrict the phase analysis to the band we actually care about. This prevents the
    # estimate from being dominated by frequencies that are physiologically irrelevant.
    phase_band = awct[(freqs >= HF_band[0]) & (freqs <= HF_band[1]), :]

    # Average the phase across the band to get one representative phase angle per time point.
    wave_phase_HF = np.atan2(
        np.mean(np.sin(phase_band), axis=0), np.mean(np.cos(phase_band), axis=0)
    )

    # The strongest local coupling frequency can vary slightly over time. We identify the
    # dominant frequency at each time point and translate the phase delay into a time lag.
    peaksid_crchf = np.argmax(
        CRC[(freqs >= HF_band[0]) & (freqs <= HF_band[1]), :], axis=0
    )
    tau_score = wave_phase_HF / (2 * np.pi * freqs[peaksid_crchf])

    # A short moving average stabilizes the sign of the delay and makes the directionality
    # trend less sensitive to single-sample phase jitter.
    W_samp = 15 * 4 + 1
    di_score = _moving_average(np.sign(tau_score), W_samp)

    return tau_score, di_score


def _moving_average(a, n=3):
    """Apply a centered moving average to smooth directional estimates over time."""
    assert n % 2 == 1
    N = len(a)
    cumsum_vec = np.cumsum(np.insert(np.pad(a, (n - 1, n - 1), "constant"), 0, 0))
    d = np.hstack(
        (np.arange(n // 2 + 1, n), np.ones(N - n) * n, np.arange(n, n // 2, -1))
    )
    return (cumsum_vec[n + n // 2 : -n // 2 + 1] - cumsum_vec[n // 2 : -n - n // 2]) / d
