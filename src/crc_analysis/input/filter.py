import numpy as np
from scipy import signal


def filter_ecg12(ecg12, fs: int = 250) -> np.ndarray:
    """Apply the fixed preprocessing filter used by the ECG feature extractor.

    The filter removes 50-Hz mains interference when it is below Nyquist, applies a
    37-Hz low-pass filter to reduce high-frequency noise, and applies a 1.2-Hz
    high-pass filter to suppress baseline drift. Zero-phase filtering preserves QRS
    timing. Both ``(n_samples, 12)`` and ``(12, n_samples)`` layouts are accepted and
    returned in their original layout.

    Parameters
    ----------
    ecg12
        Input ECG array with shape (n_samples, 12) or (12, n_samples).
    fs
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Finite filtered ECG with the same shape and orientation as the input.

    Raises
    ------
    ValueError
        If the input is not a finite 12-lead array or is too short for zero-phase filtering.
    """
    ecg12 = np.asarray(ecg12, dtype=float)
    if ecg12.ndim != 2:
        raise ValueError("ecg12 must be a 2-D array.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    # Internally filter one row per lead while preserving the caller's orientation on return.
    if ecg12.shape[0] == 12 and ecg12.shape[1] != 12:
        transposed = False
        filtered = ecg12.copy()
    elif ecg12.shape[1] == 12 and ecg12.shape[0] != 12:
        transposed = True
        filtered = ecg12.T.copy()
    else:
        raise ValueError("ecg12 must have exactly 12 leads.")
    if filtered.shape[1] < 32:
        raise ValueError("ECG recording is too short for zero-phase filtering.")
    if not np.isfinite(filtered).all():
        raise ValueError("ecg12 contains non-finite values.")

    # Remove mains hum only when the requested notch frequency is representable at this fs.
    if 50.0 < fs / 2.0:
        b_notch, a_notch = signal.iirnotch(50, 40, fs)
        filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=-1)

    # Retain QRS morphology while reducing high-frequency noise.
    b_lowpass, a_lowpass = signal.cheby2(4, 8, 37 / (fs / 2), output="ba")
    filtered = signal.filtfilt(b_lowpass, a_lowpass, filtered, axis=-1)

    # Suppress baseline wander that can obscure the heartbeat detector.
    b_highpass, a_highpass = signal.iirdesign(
        1.2 / (fs / 2), 0.5 / (fs / 2), 1, 20, ftype="butter"
    )
    filtered = signal.filtfilt(b_highpass, a_highpass, filtered, axis=-1)

    if not np.isfinite(filtered).all():
        raise ValueError("ECG filtering produced non-finite values.")
    return filtered.T if transposed else filtered
