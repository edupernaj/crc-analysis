import numpy as np
import pycwt as wavelet


def calculate_crc(
    rr_intervals: np.ndarray, edr: np.ndarray, time_edr: np.ndarray, fs: float
) -> dict:
    """Estimate time-varying cardiorespiratory coupling with wavelets.

    The function combines cross-wavelet magnitude and wavelet coherence as
    ``CRC = |XWT|² × |WCT|``. Inputs must be finite, aligned, and uniformly sampled;
    these checks prevent malformed field-recording segments from reaching PyCWT. If
    PyCWT's XWT branch fails, a direct CWT product is used and the fallback is recorded.

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
        Dictionary containing CRC, coherence, phase, frequencies, cone of influence,
        XWT, and XWT method/error metadata.
    """
    # Normalize arrays before validating their alignment and sampling assumptions.
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    edr = np.asarray(edr, dtype=float)
    time_edr = np.asarray(time_edr, dtype=float)

    # PyCWT cannot interpret short, misaligned, or missing-value inputs safely.
    if len(rr_intervals) < 2 or len(time_edr) < 2 or len(edr) < 2:
        raise ValueError("Not enough valid data points for wavelet analysis.")
    if len(rr_intervals) != len(edr):
        raise ValueError("rr_intervals and edr must have the same length.")
    if len(rr_intervals) != len(time_edr):
        raise ValueError("rr_intervals and time_edr must have the same length.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if not np.isfinite(rr_intervals).all() or not np.isfinite(edr).all():
        raise ValueError("rr_intervals and edr must contain only finite values.")
    if not np.isfinite(time_edr).all() or np.any(np.diff(time_edr) <= 0):
        raise ValueError("time_edr must be finite and strictly increasing.")
    if not np.allclose(np.diff(time_edr), 1.0 / fs, rtol=1e-3, atol=1e-6):
        raise ValueError("time_edr must be uniformly sampled at the supplied fs.")

    # Fixed Morlet and frequency-grid parameters used by the CRC definition.
    mother = wavelet.Morlet(6)
    dt = 1 / fs  # Time step based on the sampling frequency

    # Frequency limits for the wavelet transform
    f_min = 0.02  # Minimum frequency in Hz
    f_max = 0.6  # Maximum frequency in Hz

    dj = 1 / 12  # Scale resolution
    S0 = 1 / (mother.flambda() * f_max)
    J = int(np.ceil(np.log2(f_max / f_min) / dj))

    # Wavelet coherence supplies the shared time-frequency strength term.
    wct, awct, coi, freq, sig = wavelet.wct(
        rr_intervals, edr, dt=dt, dj=dj, s0=S0, J=J, wavelet=mother, sig=False
    )

    # Calculate the cross-wavelet spectrum between RR intervals and EDR.
    # In some real-world RR traces, PyCWT's internal AR(1) significance
    # estimate raises a Warning on the input series even though the underlying
    # wavelet decomposition itself is valid. In that case, compute the XWT
    # directly from the two CWTs and skip the unstable AR(1) branch.
    xwt_method = "pycwt.xwt"
    xwt_error = None
    try:
        xwt, coi_xwt, freq_xwt, sig_xwt = wavelet.xwt(
            rr_intervals, edr, dt=dt, dj=dj, s0=S0, J=J, wavelet=mother
        )
    except Exception as exc:
        xwt_method = "direct_cwt_fallback"
        xwt_error = f"{type(exc).__name__}: {exc}"
        w1, _, _, _, _, _ = wavelet.cwt(
            rr_intervals, dt, dj=dj, s0=S0, J=J, wavelet=mother
        )
        w2, _, _, _, _, _ = wavelet.cwt(edr, dt, dj=dj, s0=S0, J=J, wavelet=mother)
        xwt = w1 * np.conjugate(w2)

    # Combine cross-wavelet energy and coherence into the CRC strength metric.
    crc = np.absolute(xwt) ** 2 * np.absolute(wct)

    return {
        "CRC": crc,
        "wct": wct,
        "phase": awct,
        "freqs": freq,
        "coi": coi,
        "xwt": xwt,
        "xwt_method": xwt_method,
        "xwt_error": xwt_error,
    }
