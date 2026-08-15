import numpy as np
import pycwt as wavelet


def crc_calculation(
    rr_intervals: np.ndarray, edr: np.ndarray, tEdr: np.ndarray, fs: float = 4
) -> dict:
    """
    Calculate the Cardiorespiratory Coupling (CRC) using wavelet coherence and cross wavelet spectrum.
    CRC = |XWT|^2 * |WCT| according to Li et al. (2023) "A novel dynamic cardiorespiratory coupling
    quantification method reveals the effect of aging on the autonomic nervous system"


    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of RR intervals in milliseconds.
    edr : np.ndarray
        Array of EDR values.
    tEdr : np.ndarray
        Time vector for the resampled EDR waveform.
    fs : float
        Sampling frequency in Hz for the wavelet transform.

    Returns
    -------
    dict
        Dictionary containing the CRC results, including wavelet coherence, cross wavelet spectrum, and other relevant information.
    """
    # Ensure rr_intervals and tEdr are numpy arrays
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    edr = np.asarray(edr, dtype=float)
    tEdr = np.asarray(tEdr, dtype=float)

    # Check if there are enough valid data points for wavelet analysis
    if len(rr_intervals) < 2 or len(tEdr) < 2 or len(edr) < 2:
        raise ValueError("Not enough valid data points for wavelet analysis.")
    # Check if rr_intervals and tEdr have the same length
    if len(rr_intervals) != len(edr):
        raise ValueError("rr_intervals and edr must have the same length.")
    if len(rr_intervals) != len(tEdr):
        raise ValueError("rr_intervals and tEdr must have the same length.")

    # Define wavelet parameters
    mother = wavelet.Morlet(6)
    dt = 1 / fs  # Time step based on the sampling frequency

    # Frequency limits for the wavelet transform
    f_min = 0.02  # Minimum frequency in Hz
    f_max = 0.6  # Maximum frequency in Hz

    dj = 1 / 12  # Scale resolution
    S0 = 1 / (mother.flambda() * f_max)
    J = int(np.ceil(np.log2(f_max / f_min) / dj))
    # Perform wavelet coherence analysis
    wct, awct, coi, freq, sig = wavelet.wct(
        rr_intervals, edr, dt=dt, dj=dj, s0=S0, J=J, wavelet=mother, sig=False
    )

    # Calculate the cross wavelet spectrum between RR intervals and EDR signal.
    # In some real-world RR traces, PyCWT's internal AR(1) significance
    # estimate raises a Warning on the input series even though the underlying
    # wavelet decomposition itself is valid. In that case, compute the XWT
    # directly from the two CWTs and skip the unstable AR(1) branch.
    try:
        xwt, coi_xwt, freq_xwt, sig_xwt = wavelet.xwt(
            rr_intervals, edr, dt=dt, dj=dj, s0=S0, J=J, wavelet=mother
        )
    except Exception:
        w1, _, _, _, _, _ = wavelet.cwt(
            rr_intervals, dt, dj=dj, s0=S0, J=J, wavelet=mother
        )
        w2, _, _, _, _, _ = wavelet.cwt(edr, dt, dj=dj, s0=S0, J=J, wavelet=mother)
        xwt = w1 * np.conjugate(w2)
        sig_xwt = np.ones_like(xwt)

    CRC = np.absolute(xwt) ** 2 * np.absolute(wct)

    return {
        "CRC": CRC,
        "wct": wct,
        "phase": awct,
        "freqs": freq,
        "coi": coi,
        "xwt": xwt,
    }


def crc_average(
    crc: np.ndarray, wct: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, float, dict]:
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

    Returns
    -------
    en_CRC : np.ndarray
        Array of energy CRC values for each frequency.
    avg_crc_HF : float
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

    en_CRC = crc.mean(axis=1)  # Average CRC across time for each frequency
    en_CRC_respband = np.where((freqs >= 0.1) & (freqs <= 0.4), en_CRC, 0)
    resoFreq = 0.2  # Frequency with the maximum average CRC
    LF, HF = [
        [0.04, resoFreq - 0.05],
        [resoFreq - 0.05, resoFreq + 0.05],
    ]  # Define frequency bands around the frequency of maximum CRC component+
    # Check if the length of freqs matches the number of rows in crc

    # Identify indices corresponding to the specified frequency band
    freq_indices = np.where((freqs >= HF[0]) & (freqs <= HF[1]))[0]

    details = {
        "resoFreq": resoFreq,
        "HF_band": HF,
        "LF_band": LF,
    }

    # If no frequencies fall within the specified band, return NaN
    if len(freq_indices) == 0:
        return en_CRC, np.nan, details

    # Calculate the average CRC value within the specified frequency band
    avg_crc_HF = np.median(crc[freq_indices, :], axis=0)
    avg_wct_HF = np.median(wct[freq_indices, :], axis=0)
    return en_CRC, avg_crc_HF, avg_wct_HF, details


def crc_phase(
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

    def moving_average(a, n=3):
        assert n % 2 == 1
        N = len(a)
        cumsum_vec = np.cumsum(np.insert(np.pad(a, (n - 1, n - 1), "constant"), 0, 0))
        d = np.hstack(
            (np.arange(n // 2 + 1, n), np.ones(N - n) * n, np.arange(n, n // 2, -1))
        )
        return (
            cumsum_vec[n + n // 2 : -n // 2 + 1] - cumsum_vec[n // 2 : -n - n // 2]
        ) / d

    phase_band = awct[(freqs >= HF_band[0]) & (freqs <= HF_band[1]), :]
    wave_phase_HF = np.atan2(
        np.mean(np.sin(phase_band), axis=0), np.mean(np.cos(phase_band), axis=0)
    )
    peaksid_crchf = np.argmax(
        CRC[(freqs >= HF_band[0]) & (freqs <= HF_band[1]), :], axis=0
    )
    tau = wave_phase_HF / (2 * np.pi * freqs[peaksid_crchf])
    W_samp = 15 * 4 + 1
    DI = moving_average(np.sign(tau), W_samp)

    return tau, DI
