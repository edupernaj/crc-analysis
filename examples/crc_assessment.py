from pathlib import Path
from matplotlib.figure import Figure
from crc_analysis.pipelines import crc_assessment

EXPERIMENT_NAME = "DRKS00034187"
SESSION_NAME = "2"
CONDITION_NAME = "CPT mit PB ohne Thermometer + Pumpe"

FS_NEW = 4  # We resample the signal to a lower rate for downstream analysis.
RESP_FREQ = 0.2  # According to the experiment
RESP_BAND = (0.02, 0.7)

OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER = OUTPUT_FOLDER.joinpath("plots")
PLOTS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = OUTPUT_FOLDER.joinpath("results")
RESULTS_FOLDER.mkdir(exist_ok=True)


def _save_fig(fig: Figure, file_name: str):
    fig.savefig(PLOTS_FOLDER.joinpath(file_name), dpi=300, bbox_inches="tight")


def main() -> None:

    results = crc_assessment(
        data_root = Path("data"),
        experiment_name = EXPERIMENT_NAME,
        session_name = SESSION_NAME,
        condition_name = CONDITION_NAME,
        fs_new = FS_NEW,
        resp_band = RESP_BAND,
        resp_freq = RESP_FREQ,
    )
    print("\nFiltered data shape:", results["filtered_data"].shape)

    print("\nExtracted features (EDR and RR intervals:")
    print(" - EDR shape:", results["edr"].shape)
    print(" - EDR times shape:", results["time_edr"].shape)
    print(" - Respiration rate (BPM):", results["resp_rate_bpm"])
    print(" - RR intervals shape:", results["rr_intervals"].shape)
    print(" - Processed EDR shape:", results["edr_processed"].shape)
    _save_fig(results["fig_rr_edr"], f"RR_EDR_plot_session{SESSION_NAME}.png")

    print("\nCRC metric:")
    print(" - CRC result keys:", results["crc_result"].keys())
    print(" - CRC shape", results["crc_result"]["CRC"].shape)
    print(" - Energy CRC shape:", results["energy_crc"].shape)
    print(" - Average CRC (HF):", results["avg_crc_hf"].shape)
    print(" - Details keys:", results["crc_details"].keys())
    print(" - HF band:", results["crc_details"]["HF_band"])
    print(" - tau shape; ", results["tau_score"].shape)
    print(" - DI shape: ", results["di_score"].shape)
    _save_fig(results["fig_crc"], f"CRC_session{SESSION_NAME}.png")
    _save_fig(results["fig_tau_DI"], f"tau_DI_session{SESSION_NAME}.png")

    print("")

if __name__ == "__main__":
    main()
