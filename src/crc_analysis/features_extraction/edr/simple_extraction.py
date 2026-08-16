"""Simple ECG-derived respiration from 12-lead ECG amplitude modulation.

The algorithm detects R peaks on one lead, measures beat-to-beat R-wave
amplitude changes across all 12 leads, robustly normalizes and fuses the
lead-wise features, interpolates them to a uniform time axis, band-pass
filters the result in the respiration band, and estimates respiratory rate.
"""

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


def edr_extraction_simple(
    ecg12: np.ndarray,
    fs: float,
    output_fs: float,
    detection_lead: int | None = None,
    resp_band: tuple[float, float] = (0.08, 0.7),
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Estimate ECG-derived respiration from a 12-lead ECG.

    This is the main entry point for the amplitude-modulation EDR method. The idea is to
    locate the R peaks in the ECG, measure how the R-wave amplitude changes from beat to
    beat, and then convert that changing amplitude into a surrogate respiratory signal.
    That signal is then filtered in the respiration band and summarized as an estimated
    breathing rate.

    Parameters
    ----------
    ecg12
        ECG array with shape (n_samples, 12) or (12, n_samples).
    fs
        Sampling frequency in Hz.
    output_fs
        Sampling frequency in Hz for the uniformly resampled EDR waveform.
    detection_lead
        Optional 0-based lead index used for R-peak detection. If None, the lead with the
        clearest QRS activity is chosen automatically.
    resp_band
        Two-element tuple defining the respiration band in Hz.

    Returns
    -------
    edr
        Uniformly sampled ECG-derived respiration waveform.
    time_edr
        Time vector for ``edr`` in seconds.
    resp_rate_bpm
        Estimated respiratory rate in breaths per minute.
    details
        Dictionary containing intermediate results used for inspection, debugging, and tests.
    """

    # Convert the input to a plain float array so each step works with a predictable numeric type.
    ecg = np.asarray(ecg12, dtype=float)
    if ecg.ndim != 2:
        raise ValueError("ecg12 must be a 2-D array.")

    # Some ECG data are stored as (12, n_samples) instead of (n_samples, 12). We normalize that
    # orientation once at the beginning so the rest of the pipeline does not have to think about it.
    if ecg.shape[1] != 12 and ecg.shape[0] == 12:
        ecg = ecg.T
    if ecg.shape[1] != 12:
        raise ValueError("ecg12 must have 12 leads.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if output_fs <= 0:
        raise ValueError("output_fs must be positive.")
    if len(resp_band) != 2 or resp_band[0] <= 0 or resp_band[1] <= resp_band[0]:
        raise ValueError("resp_band must be a two-element increasing tuple.")

    # Choose the lead out of the 12 which has the highest QRS prominence
    if detection_lead is None:
        detection_lead_idx = _choose_detection_lead(ecg, fs)
    else:
        detection_lead_idx = int(np.clip(detection_lead, 0, 11))

    # Use the chosen lead to locate the QRS spikes; this is the beat-timing stage of the pipeline.
    det_lead_signal = _bandpass_for_qrs(ecg[:, detection_lead_idx], fs)
    r_peaks = _detect_r_peaks(det_lead_signal, fs)
    if r_peaks.size < 4:
        raise ValueError(
            "Too few R peaks detected to build a reliable respiration signal."
        )

    # For each detected beat, measure the local R-wave amplitude across all leads. The resulting
    # feature matrix describes how the ECG morphology changes from beat to beat.
    feature_times, amp_by_lead = _calculate_r_amplitude_features(ecg, r_peaks, fs)
    valid_rows = np.all(np.isfinite(amp_by_lead), axis=1)
    feature_times = feature_times[valid_rows]
    amp_by_lead = amp_by_lead[valid_rows]
    if amp_by_lead.shape[0] < 4:
        raise ValueError("Too few valid beats remained after feature extraction.")

    # Standardize the lead-wise amplitudes so one noisy lead does not dominate the final estimate.
    amp_by_lead_norm = _robust_zscore_columns(amp_by_lead)
    combined_feature = np.nanmedian(amp_by_lead_norm, axis=1)

    # Remove any leftover non-finite values and ensure the respiration feature is still usable.
    valid_feature = np.isfinite(combined_feature)
    feature_times = feature_times[valid_feature]
    combined_feature = combined_feature[valid_feature]
    amp_by_lead_norm = amp_by_lead_norm[valid_feature]
    if combined_feature.size < 4:
        raise ValueError("Too few valid feature samples after lead fusion.")

    # Interpolate the beat-level feature onto a regular time grid so the respiratory signal is
    # continuous and ready for filtering.
    time_edr = np.arange(feature_times[0], feature_times[-1] + 1e-12, 1.0 / output_fs)
    interpolator = PchipInterpolator(feature_times, combined_feature, extrapolate=True)
    edr_raw = interpolator(time_edr)
    edr = _bandpass_respiration(edr_raw, output_fs, resp_band)
    edr = edr - np.nanmean(edr)

    # Estimate respiratory rate by identifying the dominant frequency inside the breathing band.
    resp_rate_bpm = _estimate_resp_rate(edr, output_fs, resp_band)

    details = {
        "detection_lead": detection_lead_idx,
        "r_peaks": r_peaks,
        "r_peak_times": r_peaks / fs,
        "feature_times": feature_times,
        "amplitude_by_lead": amp_by_lead,
        "amplitude_by_lead_normalized": amp_by_lead_norm,
        "combined_feature": combined_feature,
        "output_fs": output_fs,
        "resp_band_hz": resp_band,
    }
    return edr, time_edr, resp_rate_bpm, details


def _choose_detection_lead(ecg: np.ndarray, fs: float) -> int:
    """Pick the lead that best highlights the QRS complexes."""
    scores = np.zeros(ecg.shape[1], dtype=float)
    for lead_idx in range(ecg.shape[1]):
        # This preparation isolates the heartbeat frequency band and lets us rank the leads by
        # how clearly the QRS spikes stand out above the baseline noise.
        x = _bandpass_for_qrs(ecg[:, lead_idx], fs)
        scores[lead_idx] = np.percentile(np.abs(x), 95) - np.median(np.abs(x))
    return int(np.argmax(scores))


def _bandpass_for_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Filter the ECG in the band where the QRS complex is most visible."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    sos = butter(2, [5.0, 25.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Find the main R peaks in the filtered ECG signal."""
    # We require enough separation between adjacent beats to avoid counting the same peak twice.
    min_peak_distance = max(int(round(0.50 * fs)), 1)
    prominence = max(0.6 * float(np.nanstd(x)), np.finfo(float).eps)
    r_peaks, _ = find_peaks(x, distance=min_peak_distance, prominence=prominence)

    # The first peak finder can be a bit coarse, so we refine each candidate to the local maximum
    # within a small window around the detected spike.
    search_radius = max(int(round(0.03 * fs)), 1)
    refined = []
    for peak in r_peaks:
        left = max(0, peak - search_radius)
        right = min(x.size, peak + search_radius + 1)
        local_idx = np.argmax(x[left:right])
        refined.append(left + int(local_idx))

    return np.unique(np.asarray(refined, dtype=int))


def _calculate_r_amplitude_features(
    ecg: np.ndarray, r_peaks: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Measure each beat's local R-wave amplitude across all leads."""
    pre = int(round(0.08 * fs))
    post = int(round(0.08 * fs))

    # Ignore beats that are too close to the start or end of the recording to compute a stable local
    # baseline around the R peak.
    valid = (r_peaks > pre) & (r_peaks + post < ecg.shape[0])
    r_peaks = r_peaks[valid]

    amp_by_lead = np.full((r_peaks.size, ecg.shape[1]), np.nan, dtype=float)
    for beat_idx, peak_idx in enumerate(r_peaks):
        left_idx = peak_idx - pre
        right_idx = peak_idx + post
        # The baseline is estimated just before and after the spike, so the amplitude is the
        # excess above that local level rather than the absolute signal value.
        baseline = 0.5 * (ecg[left_idx, :] + ecg[right_idx, :])
        amp_by_lead[beat_idx, :] = ecg[peak_idx, :] - baseline

    feature_times = r_peaks / fs
    return feature_times, amp_by_lead


def _robust_zscore_columns(x: np.ndarray) -> np.ndarray:
    """Normalize each lead to a centered, scale-aware version before combining them."""
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    mad[mad < np.finfo(float).eps] = np.nan
    return (x - med) / mad


def _bandpass_respiration(
    x: np.ndarray, fs: float, band_hz: tuple[float, float]
) -> np.ndarray:
    """Keep only the frequency range where breathing-related variability is expected."""
    sos = butter(2, band_hz, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _estimate_resp_rate(
    edr: np.ndarray, fs: float, band_hz: tuple[float, float]
) -> float:
    """Estimate the dominant breathing frequency by looking at the EDR power spectrum."""
    freqs, pxx = welch(edr, fs=fs)
    valid = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(valid):
        return float("nan")

    band_freqs = freqs[valid]
    band_power = pxx[valid]
    return 60.0 * float(band_freqs[np.argmax(band_power)])
