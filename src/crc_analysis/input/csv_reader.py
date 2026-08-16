import numpy as np
from pathlib import Path
import pandas as pd


# Standard 12-lead ECG exports contain six precordial leads and six limb leads. The project
# previously assumed that the leads were named V1--V12, which rejected the actual export format
# and could also encourage callers to treat arbitrary metadata columns as ECG channels.
ECG_LEADS = tuple(f"V{i}" for i in range(1, 7)) + (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
)


def read_ecg12_data(filepath: Path, fs: int = 250) -> dict:
    """Load and validate a semicolon-delimited standard 12-lead ECG export.

    Parameters
    ----------
    filepath
        Path to the CSV file containing the numeric ECG table.
    fs
        Sampling frequency in Hz used to build the time axis.

    Returns
    -------
    dict
        Dictionary containing the original table (``data``), sampling rate (``fs``),
        sample time vector (``time``), and ECG matrix (``ecg12``) with shape
        ``(12, n_samples)``. Channels are ordered as V1--V6, I, II, III, aVR, aVL,
        and aVF.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If the sampling rate, required lead set, numeric content, or sample count is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    data = pd.read_table(filepath, sep=";")
    missing_leads = [lead for lead in ECG_LEADS if lead not in data.columns]
    if missing_leads:
        raise ValueError(
            f"ECG file is missing required leads: {missing_leads}. "
            f"Available columns: {list(data.columns)}"
        )

    lead_data = data.loc[:, ECG_LEADS].apply(pd.to_numeric, errors="coerce")
    values = lead_data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        invalid_count = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"ECG file contains {invalid_count} non-finite or non-numeric lead values."
        )
    if values.shape[0] < 2:
        raise ValueError("ECG file must contain at least two samples.")

    # The file contains one row per sample; the returned matrix contains one row per lead.
    time = np.arange(values.shape[0], dtype=float) / fs
    ecg12 = values.T

    return {"data": data, "fs": fs, "time": time, "ecg12": ecg12}


def read_vcg_data(filepath: Path, fs_new: int) -> dict:
    """Read an RR/respiratory-surrogate table and reconstruct aligned signals.

    Parameters
    ----------
    filepath
        CSV file containing RR timing and respiratory surrogate values.
    fs_new
        New sampling frequency in Hz used to interpolate the signals.

    Returns
    -------
    dict
        A dictionary with interpolated RR intervals, an EDR surrogate, the new
        sampling rate, and the corresponding time axis.

    RR intervals outside 250--2500 ms are excluded from interpolation. Their median
    value is used only to keep the cumulative clock continuous after a bad row; invalid
    RR measurements themselves are never treated as valid observations.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if fs_new <= 0:
        raise ValueError("fs_new must be positive.")

    data = pd.read_csv(filepath)
    required = {"RR_ms", "Rpeak_ca"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"VCG file is missing required columns: {sorted(missing)}")

    rr_intervals = pd.to_numeric(data["RR_ms"], errors="coerce").to_numpy(float)
    edr = pd.to_numeric(data["Rpeak_ca"], errors="coerce").to_numpy(float)
    valid_rr = (
        np.isfinite(rr_intervals) & (rr_intervals >= 250.0) & (rr_intervals <= 2500.0)
    )
    if valid_rr.sum() < 2:
        raise ValueError(
            "VCG file has fewer than two physiologically valid RR intervals."
        )
    valid_edr = np.isfinite(edr)
    if valid_edr.sum() < 2:
        raise ValueError("VCG file has fewer than two finite respiratory values.")

    # Associate each interval with the timestamp of its ending beat. Invalid intervals are
    # replaced only for the clock construction so one bad row does not collapse all later
    # timestamps to the same time; invalid RR values are still excluded from the interpolation.
    replacement_rr = float(np.median(rr_intervals[valid_rr]))
    rr_for_time = np.where(valid_rr, rr_intervals, replacement_rr)
    time_rr = np.cumsum(rr_for_time / 1000.0)
    duration = float(time_rr[-1])
    time = np.arange(0.0, duration, 1.0 / fs_new)
    if time.size < 2:
        raise ValueError("VCG recording is too short after resampling.")

    # Interpolate the respiratory surrogate onto a fixed sampling grid.
    edr_p = np.interp(
        time,
        time_rr[valid_edr],
        edr[valid_edr],
        left=edr[valid_edr][0],
        right=edr[valid_edr][-1],
    )
    rr_int_p = np.interp(
        time,
        time_rr[valid_rr],
        rr_intervals[valid_rr],
        left=rr_intervals[valid_rr][0],
        right=rr_intervals[valid_rr][-1],
    )

    return {"rr_intervals": rr_int_p, "edr": edr_p, "fs_new": fs_new, "time": time}
