import numpy as np
import pycwt as wavelet


def calculate_crc(
    rr_intervals: np.ndarray, edr: np.ndarray, time_edr: np.ndarray, fs: float
) -> dict:
    """Estimate cardiorespiratory coupling using wavelet coherence and the cross-wavelet spectrum.

    We ask whether the timing of heartbeats and the breathing proxy change in
    the same frequency bands over time. The output is a time-frequency map showing where the two
    signals are moving together, which is the core signal we want to study.

    CRC = |XWT|^2 * |WCT| according to Li et al. (2023) "A novel dynamic cardiorespiratory coupling
    quantification method reveals the effect of aging on the autonomic nervous system"

    Parameters
    ----------
    rr_intervals
        RR interval trace in milliseconds aligned to the EDR time axis.
    edr
        ECG-derived respiration waveform sampled on the same time grid.
    time_edr
        Time vector in seconds corresponding to the EDR and RR series.
    fs
        Sampling frequency of the resampled EDR signal in Hz.

    Returns
    -------
    dict
        Dictionary containing the CRC map, coherence map, phase estimate, frequency axis,
        cone-of-influence boundary, and the cross-wavelet spectrum.
    """
    # Make sure the arrays are plain NumPy arrays so PyCWT sees predictable input.
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    edr = np.asarray(edr, dtype=float)
    time_edr = np.asarray(time_edr, dtype=float)

    # We cannot do wavelet analysis with a tiny or mismatched dataset.
    if len(rr_intervals) < 2 or len(time_edr) < 2 or len(edr) < 2:
        raise ValueError("Not enough valid data points for wavelet analysis.")
    if len(rr_intervals) != len(edr):
        raise ValueError("rr_intervals and edr must have the same length.")
    if len(rr_intervals) != len(time_edr):
        raise ValueError("rr_intervals and time_edr must have the same length.")

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

    # The final CRC value is a strength metric combining shared energy and coherence.
    crc = np.absolute(xwt) ** 2 * np.absolute(wct)

    return {
        "CRC": crc,
        "wct": wct,
        "phase": awct,
        "freqs": freq,
        "coi": coi,
        "xwt": xwt,
    }
