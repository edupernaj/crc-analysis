import numpy as np


def calculate_phase(
    CRC: np.ndarray, awct: np.ndarray, freqs: np.ndarray, HF_band: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the average phase of the Cardiorespiratory Coupling (CRC) in the HF band based on the cross wavelet spectrum phase

    Parameters
    -----------
    awct : np.ndarray
        2D array of cross wavelet spectrum phase values.
    freqs: np.ndarray
        Array of frequencies of the CRC.
    HF_band: list[float]
        Frequency band for high-frequency analysis.

    Returns
    ----------
    tau: np.ndarray
        1D array of CRC time delay between RR-intervals and respiratory signal. Positive tau values
        represent cardiac signal leading respiratory activity and negative tau values mean respiratory activity
        leading cardiac signals.
    DI : np.ndarray
        1D array of the Directionality Index. +1 indicate positive delay, -1 indicate negative delay,
        around 0 means mixed directions or balancing.
    """

    phase_band = awct[(freqs >= HF_band[0]) & (freqs <= HF_band[1]), :]
    wave_phase_HF = np.atan2(
        np.mean(np.sin(phase_band), axis=0), np.mean(np.cos(phase_band), axis=0)
    )
    peaksid_crchf = np.argmax(
        CRC[(freqs >= HF_band[0]) & (freqs <= HF_band[1]), :], axis=0
    )
    tau = wave_phase_HF / (2 * np.pi * freqs[peaksid_crchf])
    W_samp = 15 * 4 + 1
    DI = _moving_average(np.sign(tau), W_samp)

    return tau, DI


def _moving_average(a, n=3):
    assert n % 2 == 1
    N = len(a)
    cumsum_vec = np.cumsum(np.insert(np.pad(a, (n - 1, n - 1), "constant"), 0, 0))
    d = np.hstack(
        (np.arange(n // 2 + 1, n), np.ones(N - n) * n, np.arange(n, n // 2, -1))
    )
    return (cumsum_vec[n + n // 2 : -n // 2 + 1] - cumsum_vec[n // 2 : -n - n // 2]) / d
