import numpy as np
import pycwt as wavelet

def rsa_calculation(
        rr_intervals: np.ndarray,
        time: np.ndarray,
        resp_freq: float= 0.2,
        fs: float = 4
) -> dict:
    """
    Calculate the RSA spectral component of the RR-interval signal using continuous wavelet transform
    around a narrow HF band aroun the respiratory frequency

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of RR intervals in miliseconds
    time : np.ndarray
        Time vector for the rr intervals
    resp_freq: float
        Main respiration frequency component from the experiment
    fs: float
        sampling frequency in Hz of the signals

    Returns
    --------
    dict
        Dictionary containing the mean value across time of the RSA contribution
        of the spectral representation using WCT and other relevant information
    """

    # Ensure rr_intervals and tEdr are numpy arrays
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    time = np.asarray(time, dtype=float)

    # Check if there are enough valid data points for wavelet analysis
    if len(rr_intervals) < 2 or len(time) < 2:
        raise ValueError("Not enough valid data points for wavelet analysis.")
    # Check if rr_intervals and time vector have the same length
    if len(rr_intervals) != len(time):
        raise ValueError("rr_intervals and time must have the same length.")

    # Define wavelet parameters
    mother = wavelet.Morlet(6)
    dt = 1 / fs  # Time step based on the sampling frequency

    # Frequency limits for the wavelet transform
    f_min = 0.02  # Minimum frequency in Hz
    f_max = 0.6  # Maximum frequency in Hz
    dj = 1 / 12  # Scale resolution
    s0 = 1 / (mother.flambda() * f_max)
    J = int(np.ceil(np.log2(f_max / f_min) / dj))
    # Compute CWT
    wct, scales, freqs, coi, _, _ = wavelet.cwt(
        signal= rr_intervals-rr_intervals.mean(), dt=dt, dj=dj, s0=s0, J=J, wavelet=mother
    )

    LF, HF = [
        [0.04, resp_freq - 0.05],
        [resp_freq - 0.05, resp_freq + 0.05],
    ]
    freq_indices = np.where((freqs >= HF[0]) & (freqs <= HF[1]))[0]

    details = {
        "resp_freq": resp_freq,
        "HF_band": HF,
        "LF_band": LF,
    }

    avg_wct_HF = wct[freq_indices, :].mean(axis=0)

    return {
        "wct": wct,
        "freqs": freqs,
        "coi": coi,
        "avg_wct_HF": avg_wct_HF,
        "HF_band": HF,
    }