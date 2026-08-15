"""Simple ECG-derived respiration from 12-lead ECG amplitude modulation.

This module ports the MATLAB amplitude-modulation EDR method into Python.
The algorithm detects R peaks on one lead, measures beat-to-beat R-wave
amplitude changes across all 12 leads, robustly normalizes and fuses the
lead-wise features, interpolates them to a uniform time axis, band-pass
filters the result in the respiration band, and estimates respiratory rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


def ecg_derived_respiration_simple(
    ecg12: np.ndarray,
    fs: float,
    detection_lead: int | None = None,
    output_fs: float = 5.0,
    resp_band: tuple[float, float] = (0.08, 0.7),
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Estimate ECG-derived respiration from a 12-lead ECG.

    Parameters
    ----------
    ecg12
        ECG array with shape (n_samples, 12) or (12, n_samples).
    fs
        Sampling frequency in Hz.
    detection_lead
        Optional 0-based lead index used for R-peak detection. If None, the
        lead with the clearest QRS activity is chosen automatically.
    output_fs
        Sampling frequency in Hz for the uniformly resampled EDR waveform.
    resp_band
        Two-element tuple defining the respiration band in Hz.

    Returns
    -------
    edr
        Uniformly sampled ECG-derived respiration waveform.
    t_edr
        Time vector for ``edr`` in seconds.
    resp_rate_bpm
        Estimated respiratory rate in breaths per minute.
    details
        Dictionary containing intermediate results.
    """

    ecg = np.asarray(ecg12, dtype=float)
    if ecg.ndim != 2:
        raise ValueError("ecg12 must be a 2-D array.")

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

    if detection_lead is None:
        detection_lead_idx = _choose_detection_lead(ecg, fs)
    else:
        detection_lead_idx = int(np.clip(detection_lead, 0, 11))

    det_lead_signal = _bandpass_for_qrs(ecg[:, detection_lead_idx], fs)
    r_peaks = _detect_r_peaks(det_lead_signal, fs)
    if r_peaks.size < 4:
        raise ValueError(
            "Too few R peaks detected to build a reliable respiration signal."
        )

    feature_times, amp_by_lead = _compute_r_amplitude_features(ecg, r_peaks, fs)
    valid_rows = np.all(np.isfinite(amp_by_lead), axis=1)
    feature_times = feature_times[valid_rows]
    amp_by_lead = amp_by_lead[valid_rows]
    if amp_by_lead.shape[0] < 4:
        raise ValueError("Too few valid beats remained after feature extraction.")

    amp_by_lead_norm = _robust_zscore_columns(amp_by_lead)
    combined_feature = np.nanmedian(amp_by_lead_norm, axis=1)

    valid_feature = np.isfinite(combined_feature)
    feature_times = feature_times[valid_feature]
    combined_feature = combined_feature[valid_feature]
    amp_by_lead_norm = amp_by_lead_norm[valid_feature]
    if combined_feature.size < 4:
        raise ValueError("Too few valid feature samples after lead fusion.")

    t_edr = np.arange(feature_times[0], feature_times[-1] + 1e-12, 1.0 / output_fs)
    interpolator = PchipInterpolator(feature_times, combined_feature, extrapolate=True)
    edr_raw = interpolator(t_edr)
    edr = _bandpass_respiration(edr_raw, output_fs, resp_band)
    edr = edr - np.nanmean(edr)

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
    return edr, t_edr, resp_rate_bpm, details


def _choose_detection_lead(ecg: np.ndarray, fs: float) -> int:
    scores = np.zeros(ecg.shape[1], dtype=float)
    for lead_idx in range(ecg.shape[1]):
        x = _bandpass_for_qrs(ecg[:, lead_idx], fs)
        scores[lead_idx] = np.percentile(np.abs(x), 95) - np.median(np.abs(x))
    return int(np.argmax(scores))


def _bandpass_for_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    sos = butter(2, [5.0, 25.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    min_peak_distance = max(int(round(0.50 * fs)), 1)
    prominence = max(0.6 * float(np.nanstd(x)), np.finfo(float).eps)
    r_peaks, _ = find_peaks(x, distance=min_peak_distance, prominence=prominence)

    search_radius = max(int(round(0.03 * fs)), 1)
    refined = []
    for peak in r_peaks:
        left = max(0, peak - search_radius)
        right = min(x.size, peak + search_radius + 1)
        local_idx = np.argmax(x[left:right])
        refined.append(left + int(local_idx))

    return np.unique(np.asarray(refined, dtype=int))


def _compute_r_amplitude_features(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    pre = int(round(0.08 * fs))
    post = int(round(0.08 * fs))

    valid = (r_peaks > pre) & (r_peaks + post < ecg.shape[0])
    r_peaks = r_peaks[valid]

    amp_by_lead = np.full((r_peaks.size, ecg.shape[1]), np.nan, dtype=float)
    for beat_idx, peak_idx in enumerate(r_peaks):
        left_idx = peak_idx - pre
        right_idx = peak_idx + post
        baseline = 0.5 * (ecg[left_idx, :] + ecg[right_idx, :])
        amp_by_lead[beat_idx, :] = ecg[peak_idx, :] - baseline

    feature_times = r_peaks / fs
    return feature_times, amp_by_lead


def _robust_zscore_columns(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    mad[mad < np.finfo(float).eps] = np.nan
    return (x - med) / mad


def _bandpass_respiration(
    x: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
) -> np.ndarray:
    sos = butter(2, band_hz, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _estimate_resp_rate(
    edr: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
) -> float:
    freqs, pxx = welch(edr, fs=fs)
    valid = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(valid):
        return float("nan")

    band_freqs = freqs[valid]
    band_power = pxx[valid]
    return 60.0 * float(band_freqs[np.argmax(band_power)])
