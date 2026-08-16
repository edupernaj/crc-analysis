import numpy as np
from scipy import signal


def filter_ecg12(ecg12, fs: int = 250) -> np.ndarray:
    """Clean a 12-lead ECG by removing the main interference and keeping the useful signal.

    This takes the raw ECG, removes the 50 Hz power-line hum, smooths
    the signal so the QRS complexes stay visible, and removes slow baseline drift. The
    goal is to make the heartbeat shape easier to detect without damaging the actual ECG.

    Parameters
    ----------
    ecg12
        Input ECG array with shape (n_samples, 12) or (12, n_samples).
    fs
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered 12-lead ECG with mains hum and slow drift suppressed.


    """
    # Remove the local mains hum; in many labs this shows up as a narrow 50 Hz noise.
    b_notch, a_notch = signal.iirnotch(50, 40, fs)
    ecg12 = signal.filtfilt(b_notch, a_notch, ecg12)

    # Gently low-pass the signal so the high-frequency noise is reduced while the ECG
    # morphology remains intact.
    b_lowpass, a_lowpass = signal.cheby2(4, 8, 37 / (fs / 2), output="ba")
    ecg12 = signal.filtfilt(b_lowpass, a_lowpass, ecg12)

    # Remove slow drift such as baseline wandering, because respiration and movement can
    # otherwise dominate the signal and hide the heartbeat peaks.
    b_highpass, a_highpass = signal.iirdesign(
        1.2 / (fs / 2), 0.5 / (fs / 2), 1, 20, ftype="butter"
    )
    ecg12 = signal.filtfilt(b_highpass, a_highpass, ecg12)

    return ecg12
