from pathlib import Path

import numpy as np
import pandas as pd

from crc_analysis.pipelines import crc_assessment
from crc_analysis import features_extraction
from crc_analysis.input import DataLoaderEcg12
from crc_analysis.input.loader_ecg12 import Condition
import crc_analysis.crc.wavelet as crc_wavelet

DATA_ROOT = Path("data")
EXPERIMENT_NAME = "DRKS00034187"
EXPECTED_CONDITIONS = [
    "Baseline_",
    "CPT mit PB ohne Thermometer + Pumpe",
    "Recovery",
]

FS_NEW = 4  # We resample the signal to a lower rate for downstream analysis.
RESP_FREQ = 0.2  # According to the experiment
RESP_BAND = (0.02, 0.7)

OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER = OUTPUT_FOLDER.joinpath("plots")
PLOTS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = OUTPUT_FOLDER.joinpath("results")
RESULTS_FOLDER.mkdir(exist_ok=True)


def process_condition(
    session_name: str,
    condition_name: str,
) -> pd.DataFrame:
    """
    Run the complete CRC pipeline for one session × condition.

    Returns
    -------
    pandas.DataFrame
        One row per time point containing CRC, tau, DI, etc.
    """

    # Run assessment on (session, condition) data
    print()
    print("-" * 80)
    print(f"Session   : {session_name}")
    print(f"Condition : {condition_name}")
    results = crc_assessment(
        data_root=DATA_ROOT,
        experiment_name=EXPERIMENT_NAME,
        session_name=session_name,
        condition_name=condition_name,
        fs_new=FS_NEW,
        resp_band=RESP_BAND,
        resp_freq=RESP_FREQ,
        do_plot=False,
    )

    # Give feedback to the user
    print(f"EDR samples: {len(results['edr'])}")
    print(f"CRC map shape: {results['crc_results']['CRC'].shape}")
    if results["avg_crc_hf"] is None:
        raise ValueError(
            f"No wavelet frequencies found inside HF band {results['hf_band']}."
        )
    print(f"HF band: {results['crc_details']['hf_band']}")
    print(f"HF CRC shape: {results['avg_crc_hf'].shape}")
    print(f"tau shape: {results['tau_score'].shape}")
    print(f"DI shape: {results['di_score'].shape}")

    # Build output table
    n = len(results["time_edr"])
    if not (
        len(results["avg_crc_hf"]) == n
        and len(results["avg_wct_hf"]) == n
        and len(results["tau_score"]) == n
        and len(results["di_score"]) == n
        and len(results["rr_intervals"]) == n
        and len(results["edr_processed"]) == n
    ):
        raise ValueError(
            "Output arrays do not all have the same length. "
            f"time={n}, CRC={len(results['avg_crc_hf'])}, "
            f"WCT={len(results['avg_wct_hf'])}, tau={len(results['tau_score'])}, "
            f"DI={len(results['di_score'])}, RR={len(results['rr_intervals'])}, "
            f"EDR={len(results['edr_processed'])}"
        )
    df = pd.DataFrame(
        {
            "session_id": str(session_name),
            "condition": condition_name,
            "t_seconds": results["time_edr"],
            "CRC": results["avg_crc_hf"],
            "WCT": results["avg_wct_hf"],
            "tau": results["tau_score"],
            "DI": results["di_score"],
            "RR_ms": results["rr_intervals"],
            "EDR": results["edr_processed"],
            "resp_rate_bpm": results["resp_rate_bpm"],
        }
    )

    return df


def safe_filename(text: str) -> str:
    """Convert a condition/session name into a safe filename."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def main() -> None:
    print()
    print("-" * 80)
    print("CRC RESULT GENERATION")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Output:     {RESULTS_FOLDER.resolve()}")

    # Load experiment
    experiment = DataLoaderEcg12(
        experiment_name=EXPERIMENT_NAME,
        data_root=DATA_ROOT,
    )
    print("Available sessions:")
    for session in experiment.sessions:
        print(f"  {session}")

    # Process every session and every condition
    all_results: list[pd.DataFrame] = []
    for session_name in experiment.sessions:
        session = experiment[session_name]
        print()
        print("#" * 80)
        print(f"# SESSION {session_name}")
        print("#" * 80)
        print("Available conditions:")
        for condition_name in session.conditions:
            print(f"  {condition_name}")
        # Process the conditions that actually exist.
        #
        # We do not hard-fail if one of the expected conditions is absent,
        # because the data loader may contain incomplete sessions.
        for condition_name in EXPECTED_CONDITIONS:
            if condition_name not in session.conditions:
                print()
                print(
                    f"WARNING: Session {session_name} does not contain "
                    f"condition '{condition_name}'. Skipping."
                )
                continue
            try:
                result_df = process_condition(
                    session_name=session_name,
                    condition_name=condition_name,
                )
                # Save individual session × condition result
                filename = (
                    f"session_{safe_filename(session_name)}"
                    f"__{safe_filename(condition_name)}.csv"
                )
                output_path = RESULTS_FOLDER / filename
                result_df.to_csv(output_path, index=False)
                print()
                print(f"Saved: {output_path}")
                print(f"Rows : {len(result_df)}")
                all_results.append(result_df)

            except Exception as exc:
                print()
                print(
                    f"ERROR processing session={session_name}, "
                    f"condition={condition_name}"
                )
                print(f"{type(exc).__name__}: {exc}")

                # Continue with the other recordings rather than losing
                # everything because one recording is problematic.
                continue

    # Combined result
    if not all_results:
        raise RuntimeError(
            "No results were generated. Check the experiment path and data."
        )

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = RESULTS_FOLDER / "all_crc_results.csv"
    combined.to_csv(combined_path, index=False)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Total rows:     {len(combined)}")
    print(f"Sessions:       {combined['session_id'].nunique()}")
    print(f"Conditions:     {combined['condition'].nunique()}")
    print()
    print("Rows by session × condition:")

    counts = (
        combined.groupby(["session_id", "condition"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    print(counts)
    print()
    print(f"Combined result saved to:")
    print(f"  {combined_path.resolve()}")


if __name__ == "__main__":
    main()
