import numpy as np

from crc_analysis.input.filter import filter_ecg12


def test_filter_ecg12_dummy_signal():
    fs = 250
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Create deterministic 12-lead ECG-like input with a low-frequency drift
    # and a strong 60 Hz interference component.
    base_signal = np.sin(2 * np.pi * 1.0 * t)
    drift = 0.2 * np.linspace(0, 1, t.size)
    noise_60hz = 0.5 * np.cos(2 * np.pi * 60.0 * t)
    ecg12 = np.vstack(
        [base_signal + drift + noise_60hz + 0.1 * lead for lead in range(12)]
    )

    filtered = filter_ecg12(ecg12, fs)

    # Check that the filtered signal has the same shape as the input
    assert filtered.shape == ecg12.shape
    # Filtered signal should contain finite values
    assert np.all(np.isfinite(filtered))
    # Filtered signal should not be identical to the original
    assert not np.allclose(filtered, ecg12)
    # Filtered signal should have less variance than the original
    assert np.std(filtered) < np.std(ecg12)
