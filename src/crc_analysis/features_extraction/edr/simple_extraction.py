"""ECG-derived respiration from multi-lead R-wave amplitude modulation.

The extractor chooses a QRS-prominent lead, detects positive or negative R waves,
checks broad physiological RR limits, and measures beat-to-beat amplitude changes
across all twelve leads. Robust lead-wise normalization and median fusion produce a
respiratory surrogate, which is interpolated to a uniform grid, filtered, and
summarized by a dominant respiratory frequency. Quality metrics are returned so
field recordings can be audited rather than silently accepted.
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
    min_rr_valid_fraction: float = 0.80,
    min_duration_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Estimate ECG-derived respiration from a 12-lead ECG.

    The method uses QRS amplitude modulation as an ECG-derived respiration surrogate.
    It rejects non-finite or too-short recordings, requires a minimum fraction of
    plausible RR intervals, and reports detection and lead-quality diagnostics.

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
    min_rr_valid_fraction
        Minimum fraction of detected RR intervals between 250 and 2500 ms.
    min_duration_s
        Minimum recording duration accepted by the extractor.

    Returns
    -------
    edr
        Uniformly sampled ECG-derived respiration waveform.
    time_edr
        Time vector for ``edr`` in seconds.
    resp_rate_bpm
        Estimated respiratory rate in breaths per minute.
    details
        Intermediate arrays plus polarity, usable-lead, RR-validity, and heart-rate diagnostics.
    """

    # Normalize to a predictable floating-point representation before validation.
    ecg = np.asarray(ecg12, dtype=float)
    if ecg.ndim != 2:
        raise ValueError("ecg12 must be a 2-D array.")

    # Accept both common storage orientations and use samples-by-leads internally.
    if ecg.shape[1] != 12 and ecg.shape[0] == 12:
        ecg = ecg.T
    if ecg.shape[1] != 12:
        raise ValueError("ecg12 must have 12 leads.")
    if fs <= 50:
        raise ValueError("fs must be greater than 50 Hz for QRS filtering.")
    if output_fs <= 0:
        raise ValueError("output_fs must be positive.")
    if len(resp_band) != 2 or resp_band[0] <= 0 or resp_band[1] <= resp_band[0]:
        raise ValueError("resp_band must be a two-element increasing tuple.")
    if resp_band[1] >= output_fs / 2:
        raise ValueError(
            "resp_band upper bound must be below the output Nyquist frequency."
        )
    if not 0 < min_rr_valid_fraction <= 1:
        raise ValueError("min_rr_valid_fraction must be between 0 and 1.")
    if min_duration_s <= 0:
        raise ValueError("min_duration_s must be positive.")
    if not np.isfinite(ecg).all():
        raise ValueError("ecg12 contains non-finite values.")
    if ecg.shape[0] / fs < min_duration_s:
        raise ValueError(
            f"ECG recording is too short: {ecg.shape[0] / fs:.1f}s "
            f"(minimum {min_duration_s:.1f}s)."
        )

    # Choose the lead with the clearest filtered QRS morphology.
    if detection_lead is None:
        detection_lead_idx = _choose_detection_lead(ecg, fs)
    else:
        detection_lead_idx = int(np.clip(detection_lead, 0, 11))

    # Detect beat timing on the selected lead, including inverted ECG polarity.
    det_lead_signal = _bandpass_for_qrs(ecg[:, detection_lead_idx], fs)
    r_peaks, qrs_polarity = _detect_r_peaks(det_lead_signal, fs)
    if r_peaks.size < 4:
        raise ValueError(
            "Too few R peaks detected to build a reliable respiration signal."
        )

    rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
    valid_rr = (
        np.isfinite(rr_intervals_ms)
        & (rr_intervals_ms >= 250.0)
        & (rr_intervals_ms <= 2500.0)
    )
    rr_valid_fraction = float(np.mean(valid_rr))
    if rr_valid_fraction < min_rr_valid_fraction:
        raise ValueError(
            f"Poor beat-detection quality: only {rr_valid_fraction:.1%} of RR "
            f"intervals are physiologically plausible."
        )

    # Measure local R-wave amplitude modulation across all leads.
    feature_times, amp_by_lead = _calculate_r_amplitude_features(ecg, r_peaks, fs)
    valid_rows = np.all(np.isfinite(amp_by_lead), axis=1)
    feature_times = feature_times[valid_rows]
    amp_by_lead = amp_by_lead[valid_rows]
    if amp_by_lead.shape[0] < 4:
        raise ValueError("Too few valid beats remained after feature extraction.")

    # Robustly normalize leads so one high-amplitude or noisy channel does not dominate fusion.
    amp_by_lead_norm = _robust_zscore_columns(amp_by_lead)
    usable_leads = np.isfinite(amp_by_lead_norm).any(axis=0)
    if int(usable_leads.sum()) < 3:
        raise ValueError("Fewer than three ECG leads have usable amplitude variation.")
    combined_feature = np.nanmedian(amp_by_lead_norm, axis=1)

    # Remove residual invalid fused samples and reject a constant surrogate.
    valid_feature = np.isfinite(combined_feature)
    feature_times = feature_times[valid_feature]
    combined_feature = combined_feature[valid_feature]
    amp_by_lead_norm = amp_by_lead_norm[valid_feature]
    if combined_feature.size < 4:
        raise ValueError("Too few valid feature samples after lead fusion.")

    # Interpolate beat-level modulation onto a regular grid for filtering and wavelet analysis.
    time_edr = np.arange(feature_times[0], feature_times[-1] + 1e-12, 1.0 / output_fs)
    interpolator = PchipInterpolator(feature_times, combined_feature, extrapolate=True)
    edr_raw = interpolator(time_edr)
    edr = _bandpass_respiration(edr_raw, output_fs, resp_band)
    edr = edr - np.nanmean(edr)
    if not np.isfinite(edr).all() or float(np.std(edr)) <= np.finfo(float).eps:
        raise ValueError("EDR extraction produced a non-finite or constant signal.")

    # Estimate rate from the dominant spectral peak inside the configured breathing band.
    resp_rate_bpm = _estimate_resp_rate(edr, output_fs, resp_band)

    details = {
        "detection_lead": detection_lead_idx,
        "qrs_polarity": qrs_polarity,
        "r_peaks": r_peaks,
        "r_peak_times": r_peaks / fs,
        "feature_times": feature_times,
        "amplitude_by_lead": amp_by_lead,
        "amplitude_by_lead_normalized": amp_by_lead_norm,
        "combined_feature": combined_feature,
        "n_usable_leads": int(usable_leads.sum()),
        "rr_valid_fraction": rr_valid_fraction,
        "median_rr_ms": float(np.median(rr_intervals_ms[valid_rr])),
        "median_hr_bpm": float(60000.0 / np.median(rr_intervals_ms[valid_rr])),
        "output_fs": output_fs,
        "resp_band_hz": resp_band,
    }
    return edr, time_edr, resp_rate_bpm, details


def _choose_detection_lead(ecg: np.ndarray, fs: float) -> int:
    """Return the lead with the largest robust filtered-QRS prominence."""
    scores = np.zeros(ecg.shape[1], dtype=float)
    for lead_idx in range(ecg.shape[1]):
        # Rank leads by how clearly heartbeat-band activity stands above its local baseline.
        x = _bandpass_for_qrs(ecg[:, lead_idx], fs)
        scores[lead_idx] = np.percentile(np.abs(x), 95) - np.median(np.abs(x))
    return int(np.argmax(scores))


def _bandpass_for_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    """Band-pass an ECG lead in the QRS-dominant 5--25 Hz range."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    sos = butter(2, [5.0, 25.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _detect_r_peaks(x: np.ndarray, fs: float) -> tuple[np.ndarray, int]:
    """Find the main R peaks, allowing either ECG polarity."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3 or not np.isfinite(x).all():
        raise ValueError("The QRS detection signal must be finite and non-empty.")

    # Field recordings can have either lead polarity. Detect both possibilities and retain the
    # orientation with the clearer, more numerous QRS candidates instead of assuming positive R
    # waves. A 300-ms minimum distance still permits fast sinus rhythm while rejecting double
    # counting within one QRS complex.
    min_peak_distance = max(int(round(0.50 * fs)), 1)
    prominence = max(0.7 * float(np.std(x)), np.finfo(float).eps)
    search_radius = max(int(round(0.03 * fs)), 1)
    candidates: list[tuple[float, np.ndarray, int]] = []

    for polarity in (1, -1):
        oriented = polarity * x
        peaks, properties = find_peaks(
            oriented,
            distance=min_peak_distance,
            prominence=prominence,
        )
        if peaks.size == 0:
            continue

        # Combine prominence and candidate count so an isolated large motion artifact is less
        # likely to win over a consistent sequence of QRS complexes.
        score = float(np.median(properties["prominences"]) * np.log1p(peaks.size))
        refined = []
        for peak in peaks:
            left = max(0, peak - search_radius)
            right = min(x.size, peak + search_radius + 1)
            local_idx = np.argmax(oriented[left:right])
            refined.append(left + int(local_idx))

        candidates.append((score, np.unique(np.asarray(refined, dtype=int)), polarity))

    if not candidates:
        return np.array([], dtype=int), 1

    _, r_peaks, polarity = max(candidates, key=lambda item: item[0])
    return r_peaks, polarity


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
    """Normalize each lead with its median and median absolute deviation."""
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
    """Estimate breaths per minute from the dominant EDR spectral frequency."""
    freqs, pxx = welch(edr, fs=fs)
    valid = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(valid):
        return float("nan")

    band_freqs = freqs[valid]
    band_power = pxx[valid]
    return 60.0 * float(band_freqs[np.argmax(band_power)])
