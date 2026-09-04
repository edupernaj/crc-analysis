"""
Generate block-level CRC results for the CPT condition, used for the statistical analysis report.

Experimental design
-------------------
Only the condition

    CPT mit PB ohne Thermometer + Pumpe

is analysed. ``Baseline_`` and ``Recovery`` are ignored.

The selected CPT recording contains nine sequential 60-s segments:

    segment 1 -> warm / non-pain
    segment 2 -> cold / pain
    segment 3 -> warm / non-pain
    segment 4 -> cold / pain
    segment 5 -> warm / non-pain
    segment 6 -> cold / pain
    segment 7 -> warm / non-pain
    segment 8 -> cold / pain
    segment 9 -> excluded

Thus the experiment contains four paired experimental blocks:

    pair 1: segment 1 warm + segment 2 cold
    pair 2: segment 3 warm + segment 4 cold
    pair 3: segment 5 warm + segment 6 cold
    pair 4: segment 7 warm + segment 8 cold

The output contains one row per participant x experimental pair x condition. Five
seconds at both ends of each 60-second segment are excluded to reduce contamination
from condition transitions and wavelet edge effects.

Quality control
--------------
Non-finite CRC/WCT samples, poor beat detection, unusable ECG leads, and incomplete
segments are not silently converted into model observations. Segment-level quality
metrics are written to the output table, while participant/session-level failures and
incomplete segment sets are written to ``output/results/processing_failures.csv``.

Primary outcome
---------------
``crc_log_median`` is the natural-log transformed block-level CRC median.

The raw CRC median is retained as ``crc_median`` for transparency, but the
statistical analysis should use ``crc_log_median``.

The log transformation is appropriate because CRC is an energy-like positive
quantity and the transformation makes multiplicative differences easier to
model and interpret. Because the median is strictly positive, the natural
logarithm is used directly; no offset is added. The report models the four
experimental pairs as a categorical block factor and estimates block-specific
cold-versus-warm contrasts. Each contrast can be exponentiated to obtain the
estimated CRC ratio:

    exp(beta_block)

A value below 1 therefore indicates lower CRC during cold/pain than
warm/non-pain. The generated table contains one observation per participant,
block, and condition, which is the analysis level expected by the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import crc_analysis.crc.wavelet as crc_wavelet
from crc_analysis.input import DataLoaderEcg12
from crc_analysis.pipelines import crc_assessment


DATA_ROOT = Path("data")
EXPERIMENT_NAME = "DRKS00034187"
CPT_CONDITION = "CPT mit PB ohne Thermometer + Pumpe"

SEGMENT_DURATION = 60.0  # nominal experimental segment duration in seconds
TRANSITION_BUFFER_SECONDS = 5.0  # discard each transition-adjacent edge in seconds
MIN_VALID_FRACTION = 0.80  # minimum finite CRC/WCT fraction inside the analysis window
N_EXPERIMENTAL_PAIRS = 4
N_RETAINED_SEGMENTS = N_EXPERIMENTAL_PAIRS * 2
TOTAL_SEGMENTS = 9

FS_NEW = 4  # we resample the signal to a lower rate for downstream analysis.
RESP_FREQ = 0.2  # protocol respiratory-frequency reference in Hz
# Respiration is expected in a physiological band; including frequencies below 0.08 Hz would
# allow slow drift to dominate the EDR rate estimate in noisy field recordings.
RESP_BAND = (0.08, 0.7)

OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER = OUTPUT_FOLDER.joinpath("plots")
PLOTS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = OUTPUT_FOLDER.joinpath("results")
RESULTS_FOLDER.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------
def _safe_nanmedian(values: np.ndarray) -> float:
    """Return the median of finite values or NaN if none exist."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _safe_nanmean(values: np.ndarray) -> float:
    """Return the mean of finite values or NaN if none exist."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _safe_nanstd(values: np.ndarray) -> float:
    """Return the sample SD of finite values or NaN if fewer than two exist."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def _log_crc(value: float) -> float:
    """
    Log-transform a CRC value.

    CRC must be strictly positive for the natural logarithm. We deliberately
    do not add an arbitrary constant to the data because that would change
    the scientific meaning of the outcome.
    """
    if not np.isfinite(value):
        return float("nan")
    if value <= 0:
        raise ValueError(f"CRC median must be > 0 for log transformation; got {value}.")
    return float(np.log(value))


# ---------------------------------------------------------------------------
# Segment summarisation
# ---------------------------------------------------------------------------
def summarize_segment(
    result: dict,
    segment_number: int,
    segment_duration_s: float,
    minimum_valid_fraction: float,
    transition_buffer_s: float = 0.0,
) -> dict | None:
    """
    Summarise one warm/cold segment after transition and quality filtering.

    Segment numbering is one-based. Only segments 1-8 are retained.
    Segment 9 is intentionally excluded.

    Odd segments are warm/non-pain and even segments are cold/pain.
    The returned ``block`` is the paired experimental block number 1-4.

    Parameters
    ----------
    result
        Pipeline result containing the aligned time, CRC, WCT, and feature diagnostics.
    segment_number
        One-based protocol segment number; segment 9 is excluded by design.
    segment_duration_s
        Nominal duration of a protocol segment.
    minimum_valid_fraction
        Minimum finite CRC/WCT proportion retained inside the analysis window.
    transition_buffer_s
        Seconds removed from both ends of the nominal segment.

    Returns
    -------
    dict or None
        Block summary and quality metrics, or ``None`` when the segment fails quality checks.
    """
    if not 1 <= segment_number <= TOTAL_SEGMENTS:
        raise ValueError(f"Invalid segment number: {segment_number}.")

    if segment_number > N_RETAINED_SEGMENTS:
        return None

    if segment_duration_s <= 0:
        raise ValueError("segment_duration_s must be positive.")
    if transition_buffer_s < 0 or 2 * transition_buffer_s >= segment_duration_s:
        raise ValueError(
            "transition_buffer_s must be non-negative and leave data inside the segment."
        )

    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be between 0 and 1.")

    time = np.asarray(result["time_edr"], dtype=float)
    crc = result.get("avg_crc_hf")
    wct = result.get("avg_wct_hf")

    if crc is None:
        return None

    crc = np.asarray(crc, dtype=float)

    if time.shape != crc.shape:
        raise ValueError("time_edr and avg_crc_hf must have the same shape.")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time_edr must be finite and strictly increasing.")

    if time.size == 0:
        return None

    recording_start = float(time[0])
    segment_start = recording_start + (segment_number - 1) * segment_duration_s
    segment_end = segment_start + segment_duration_s

    analysis_start = segment_start + transition_buffer_s
    analysis_end = segment_end - transition_buffer_s
    mask = (time >= analysis_start) & (time < analysis_end)

    if not np.any(mask):
        return None

    crc_segment = crc[mask]
    valid_crc = np.isfinite(crc_segment)
    valid_fraction = float(np.mean(valid_crc))

    if valid_fraction < minimum_valid_fraction:
        return None

    crc_median = _safe_nanmedian(crc_segment)

    if not np.isfinite(crc_median):
        return None

    crc_log_median = _log_crc(crc_median)

    if wct is not None:
        wct = np.asarray(wct, dtype=float)

        if wct.shape != time.shape:
            raise ValueError("time_edr and avg_wct_hf must have the same shape.")

        wct_segment = wct[mask]
    else:
        wct_segment = np.full(np.count_nonzero(mask), np.nan)

    valid_wct = np.isfinite(wct_segment)
    wct_valid_fraction = float(np.mean(valid_wct))
    if wct is not None and wct_valid_fraction < minimum_valid_fraction:
        return None

    pain = int(segment_number % 2 == 0)
    phase = "pain_cold" if pain else "non_pain_warm"

    pair_block = (segment_number + 1) // 2

    return {
        "segment": int(segment_number),
        "block": int(pair_block),
        "pain": pain,
        "phase": phase,
        "block_start_s": segment_start,
        "block_end_s": segment_end,
        "analysis_start_s": analysis_start,
        "analysis_end_s": analysis_end,
        "transition_buffer_s": transition_buffer_s,
        "crc_median": crc_median,
        "crc_log_median": crc_log_median,
        "crc_mean": _safe_nanmean(crc_segment),
        "crc_sd": _safe_nanstd(crc_segment),
        "wct_median": _safe_nanmedian(wct_segment),
        "wct_mean": _safe_nanmean(wct_segment),
        "n_timepoints": int(np.count_nonzero(mask)),
        "n_valid_crc": int(np.count_nonzero(valid_crc)),
        "valid_fraction": valid_fraction,
        "wct_valid_fraction": wct_valid_fraction,
        "rr_valid_fraction": float(
            result.get("feature_details", {}).get("rr_valid_fraction", np.nan)
        ),
        "n_usable_leads": int(
            result.get("feature_details", {}).get("n_usable_leads", 0)
        ),
        "coi_valid_fraction": float(
            result.get("crc_details", {}).get("coi_valid_fraction", np.nan)
        ),
    }


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------
def analyze_session(
    experiment_name: str,
    data_root: Path,
    session_name: str,
    segment_duration_s: float,
    resp_band: tuple[float, float],
    minimum_valid_fraction: float,
    transition_buffer_s: float = 0.0,
) -> list[dict]:
    """
    Run the CRC pipeline for one participant/session.

    Only the CPT condition is analysed. The first eight 60-s segments are
    interpreted as four warm/cold experimental pairs. Segment nine is excluded.

    The CRC pipeline needs a respiratory frequency in order to define the
    narrow high-frequency band. We initially use RESP_FREQ,
    run the complete signal-processing pipeline, and then recompute the
    frequency-band summary using the respiratory frequency estimated by the
    EDR pipeline.

    Parameters
    ----------
    experiment_name
        Name of the experiment directory.
    data_root
        Root directory containing the experiment.
    session_name
        Participant/session identifier.
    segment_duration_s
        Experimental segment duration in seconds.
    resp_band
        Respiration band passed to the EDR extraction.
    minimum_valid_fraction
        Minimum fraction of finite CRC values required per block.
    transition_buffer_s
        Seconds removed from both ends of each segment before summarization.

    Returns
    -------
    list of dict
        One dictionary per valid experimental block.
    """
    experiment = DataLoaderEcg12(
        experiment_name=experiment_name,
        data_root=data_root,
    )

    if CPT_CONDITION not in experiment[session_name].conditions:
        raise ValueError(
            f"Condition '{CPT_CONDITION}' not found for session '{session_name}'."
        )

    # Initial CRC calculation uses the protocol respiratory frequency so the pipeline can
    # estimate EDR. The HF summary is then recomputed using the measured EDR rate.
    result = crc_assessment(
        data_root=data_root,
        experiment_name=experiment_name,
        session_name=session_name,
        condition_name=CPT_CONDITION,
        fs_new=FS_NEW,
        resp_band=resp_band,
        resp_freq=RESP_FREQ,
        do_plot=False,
    )

    # Use the measured respiratory rate for the final HF band summary when available.
    resp_rate_bpm = result.get("resp_rate_bpm", np.nan)

    if np.isfinite(resp_rate_bpm) and resp_rate_bpm > 0:
        measured_resp_freq = float(resp_rate_bpm) / 60.0

        energy_crc, avg_crc_hf, avg_wct_hf, crc_details = crc_wavelet.calculate_average(
            crc=result["crc_results"]["CRC"],
            wct=result["crc_results"]["wct"],
            freqs=result["crc_results"]["freqs"],
            resp_freq=measured_resp_freq,
            coi=result["crc_results"]["coi"],
        )

        result["energy_crc"] = energy_crc
        result["avg_crc_hf"] = avg_crc_hf
        result["avg_wct_hf"] = avg_wct_hf
        result["crc_details"] = crc_details

    # Determine number of experimental segments.
    time = np.asarray(result["time_edr"], dtype=float)

    if time.size == 0:
        return []

    rows: list[dict] = []

    for segment_number in range(1, N_RETAINED_SEGMENTS + 1):
        row = summarize_segment(
            result=result,
            segment_number=segment_number,
            segment_duration_s=segment_duration_s,
            minimum_valid_fraction=minimum_valid_fraction,
            transition_buffer_s=transition_buffer_s,
        )

        if row is None:
            continue

        row.update(
            {
                "participant": str(session_name),
                "session": str(session_name),
                "experiment": experiment_name,
                "condition": CPT_CONDITION,
                "resp_rate_bpm": float(result.get("resp_rate_bpm", np.nan)),
            }
        )

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Complete cohort
# ---------------------------------------------------------------------------
def generate_dataset(
    data_root: Path,
    experiment_name: str,
    output_path: Path,
    segment_duration_s: float,
    resp_band: tuple[float, float],
    minimum_valid_fraction: float,
    transition_buffer_s: float = 0.0,
) -> pd.DataFrame:
    """
    Process all sessions and save the block-level CPT CRC dataset.

    The resulting block-level dataset has at most:

        participants x 4 paired blocks x 2 conditions

    observations.

    Parameters
    ----------
    data_root
        Root data directory.
    experiment_name
        Experiment directory name.
    output_path
        CSV file to create.
    segment_duration_s
        Duration of one experimental segment.
    resp_band
        Respiration frequency band.
    minimum_valid_fraction
    Minimum finite CRC/WCT fraction per segment.
    transition_buffer_s
        Seconds removed from both ends of each segment before summarization.

    Returns
    -------
    pandas.DataFrame
        Block-level analysis dataset with one row per participant, block, and condition.
    """
    experiment = DataLoaderEcg12(
        experiment_name=experiment_name,
        data_root=data_root,
    )

    all_rows: list[dict] = []
    failures: list[dict] = []

    for session_name in experiment.sessions:
        if CPT_CONDITION not in experiment[session_name].conditions:
            print(f"Skipping {session_name}: condition '{CPT_CONDITION}' not found.")
            failures.append(
                {
                    "participant": str(session_name),
                    "condition": CPT_CONDITION,
                    "error_type": "MissingCondition",
                    "error": "CPT condition not found",
                }
            )
            continue

        print(f"Processing participant/session: {session_name}")

        try:
            rows = analyze_session(
                experiment_name=experiment_name,
                data_root=data_root,
                session_name=session_name,
                segment_duration_s=segment_duration_s,
                resp_band=resp_band,
                minimum_valid_fraction=minimum_valid_fraction,
                transition_buffer_s=transition_buffer_s,
            )

            all_rows.extend(rows)
            print(f"  retained {len(rows)} condition segments")
            if len(rows) != N_RETAINED_SEGMENTS:
                retained_segments = {int(row["segment"]) for row in rows}
                missing_segments = sorted(
                    set(range(1, N_RETAINED_SEGMENTS + 1)) - retained_segments
                )
                failures.append(
                    {
                        "participant": str(session_name),
                        "condition": CPT_CONDITION,
                        "error_type": "IncompleteSegments",
                        "error": (
                            f"Retained {len(rows)}/{N_RETAINED_SEGMENTS} segments; "
                            f"missing segments {missing_segments}"
                        ),
                    }
                )

        except Exception as exc:
            # Keep the rest of the cohort running, but record the failure for auditability.
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            failures.append(
                {
                    "participant": str(session_name),
                    "condition": CPT_CONDITION,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(all_rows)

    failure_path = RESULTS_FOLDER / "processing_failures.csv"
    pd.DataFrame(failures).to_csv(failure_path, index=False)
    print(f"Processing failures/skips: {len(failures)}")
    print(f"Failure log: {failure_path}")

    if df.empty:
        raise RuntimeError("No valid CPT results were generated.")

    columns = [
        "participant",
        "session",
        "experiment",
        "condition",
        "block",
        "segment",
        "pain",
        "phase",
        "block_start_s",
        "block_end_s",
        "analysis_start_s",
        "analysis_end_s",
        "transition_buffer_s",
        "crc_median",
        "crc_log_median",
        "crc_mean",
        "crc_sd",
        "wct_median",
        "wct_mean",
        "resp_rate_bpm",
        "n_timepoints",
        "n_valid_crc",
        "valid_fraction",
        "wct_valid_fraction",
        "rr_valid_fraction",
        "n_usable_leads",
        "coi_valid_fraction",
    ]

    df = df[[column for column in columns if column in df.columns]]

    df = df.sort_values(["participant", "block", "pain"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nDataset generated successfully.")
    print(f"Output: {output_path}")
    print(f"Participants: {df['participant'].nunique()}")
    print(f"Observations: {len(df)}")
    print(f"Experimental pairs: {df['block'].nunique()}")
    print(f"Warm/non-pain observations: {(df['pain'] == 0).sum()}")
    print(f"Cold/pain observations: {(df['pain'] == 1).sum()}")
    print("Segment 9: excluded by design.")

    return df


def main() -> None:
    """Generate the complete CPT CRC analysis dataset."""
    output_path = RESULTS_FOLDER / "crc_cpt_blocks.csv"

    generate_dataset(
        data_root=DATA_ROOT,
        experiment_name=EXPERIMENT_NAME,
        output_path=output_path,
        segment_duration_s=SEGMENT_DURATION,
        minimum_valid_fraction=MIN_VALID_FRACTION,
        transition_buffer_s=TRANSITION_BUFFER_SECONDS,
        resp_band=RESP_BAND,
    )


if __name__ == "__main__":
    main()
