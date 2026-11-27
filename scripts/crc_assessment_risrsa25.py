

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from crc_analysis.pipelines_risrsa25 import crc_assessment_risrsa25

EXPERIMENT_NAME = "RIS RSA 25"
SESSION_NAME = "p22"
CONDITION_NAME = "t1"

FS_NEW = 4  # Uniform output rate for EDR, RR, and wavelet inputs.
RESP_FREQ = 0.2  # Protocol respiratory-frequency reference in Hz.
RESP_BAND = (0.05, 0.7)

OUTPUT_FOLDER = Path("output_RISRSA25")
OUTPUT_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER = OUTPUT_FOLDER.joinpath("plots")
PLOTS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = OUTPUT_FOLDER.joinpath("results")
RESULTS_FOLDER.mkdir(exist_ok=True)

def _save_fig(fig: Figure, file_name: str):
    """Save a diagnostic figure and release it from the Matplotlib registry."""
    if fig is None:
        return
    fig.savefig(PLOTS_FOLDER.joinpath(file_name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def main () -> None:
    """
    Process the configured example for RIS RSA 25 data from Max Schneider
    """
    results = crc_assessment_risrsa25(
        data_root=Path("data"),
        experiment_name=EXPERIMENT_NAME,
        session_name=SESSION_NAME,
        condition_name=CONDITION_NAME,
        fs_new=FS_NEW,
        resp_band=RESP_BAND,
        resp_freq=RESP_FREQ,
        do_plot=True,
    )
    print("\nFiltered data shape:", results["filtered_data"].shape)

    print("\nExtracted features (EDR and RR intervals):")
    print(" - EDR shape:", results["edr"].shape)
    print(" - EDR times shape:", results["time_edr"].shape)
    print(" - Respiration rate (BPM):", results["resp_rate_bpm"])
    print(" - RR intervals shape:", results["rr_intervals"].shape)
    print(" - Processed EDR shape:", results["edr_processed"].shape)
    _save_fig(results["fig_rr_edr"], f"RR_EDR_plot_session{SESSION_NAME}.png")

    print("\nCRC metric:")
    print(" - CRC result keys:", results["crc_results"].keys())
    print(" - CRC shape", results["crc_results"]["CRC"].shape)
    print(" - Energy CRC shape:", results["energy_crc"].shape)
    print(" - Average CRC (HF):", results["avg_crc_hf"].shape)
    print(" - Details keys:", results["crc_details"].keys())
    print(" - HF band:", results["crc_details"]["hf_band"])
    print(" - tau shape; ", results["tau_score"].shape)
    print(" - DI shape: ", results["di_score"].shape)
    _save_fig(results["fig_crc"], f"CRC_session{SESSION_NAME}.png")
    _save_fig(results["fig_tau_DI"], f"tau_DI_session{SESSION_NAME}.png")

    print("")


if __name__ == "__main__":
    main()
