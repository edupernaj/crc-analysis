import numpy as np
from scipy import signal


def filter_ecg12(ecg12, fs: int = 250) -> np.ndarray:
    """
    Filter 12-lead ECG data.

    Filter the ECG data using a notch filter at 50 Hz, a lowpass
    filter at 120 Hz, and a highpass filter at 0.2 Hz.
    """
    # IIR notch filter at 60hz
    b_notch, a_notch = signal.iirnotch(50, 40, fs)
    ecg12 = signal.filtfilt(b_notch, a_notch, ecg12)

    # Chebyshev 2 lowpass filter at 120hz
    b_lowpass, a_lowpass = signal.cheby2(4, 8, 37 / (fs / 2), output="ba")
    ecg12 = signal.filtfilt(b_lowpass, a_lowpass, ecg12)

    # Butterworth highpass filter at 0.2hz
    b_highpass, a_highpass = signal.iirdesign(
        1.2 / (fs / 2), 0.5 / (fs / 2), 1, 20, ftype="butter"
    )
    ecg12 = signal.filtfilt(b_highpass, a_highpass, ecg12)

    return ecg12
