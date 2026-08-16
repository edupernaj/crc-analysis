import numpy as np

from crc_analysis.input.filter import filter_ecg12


def test_filter_ecg12_dummy_signal():
    """Verify the ECG filter removes obvious noise while keeping the signal shape intact."""
    fs = 250
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # This signal intentionally contains both slow drift and strong powerline contamination,
    # which is the kind of noise we expect from a real ECG recording.
    base_signal = np.sin(2 * np.pi * 1.0 * t)
    drift = 0.2 * np.linspace(0, 1, t.size)
    noise_60hz = 0.5 * np.cos(2 * np.pi * 60.0 * t)
    ecg12 = np.vstack(
        [base_signal + drift + noise_60hz + 0.1 * lead for lead in range(12)]
    )

    filtered = filter_ecg12(ecg12, fs)

    # The cleaned signal should still have the same number of samples and leads.
    assert filtered.shape == ecg12.shape
    # A real filter should not create NaN or inf values.
    assert np.all(np.isfinite(filtered))
    # The filter should visibly change the signal, not just copy it.
    assert not np.allclose(filtered, ecg12)
    # Noise reduction is the point: the cleaned waveform should be less variable overall.
    assert np.std(filtered) < np.std(ecg12)
