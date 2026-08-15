from pathlib import Path
from matplotlib.figure import Figure
from crc_analysis import features_extraction
from crc_analysis.input import DataLoaderEcg12
import crc_analysis.crc.wavelet as crc_wavelet
from crc_analysis.plotting import plot_crc, plot_rr_edr, plot_tau_di


def main() -> None:
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)
    plots_folder = output_folder.joinpath("plots")
    plots_folder.mkdir(exist_ok=True)
    results_folder = output_folder.joinpath("results")
    results_folder.mkdir(exist_ok=True)

    def _save_fig(fig: Figure, file_name: str):
        fig.savefig(plots_folder.joinpath(file_name), dpi=300, bbox_inches="tight")

    # Load ECG 12 data
    experiment = DataLoaderEcg12(experiment_name="DRKS00034187")
    session_name = "2"
    condition_name = "CPT mit PB ohne Thermometer + Pumpe"
    condition_data = experiment[session_name][condition_name]
    filtered_data = condition_data.ecg
    print("Filtered data shape:", filtered_data.shape)

    # Extract features (EDR and RR intervals)
    edr, t_edr, resp_rate_bpm, details = features_extraction.edr_extraction_simple(
        ecg12=filtered_data,
        fs=condition_data.fs,
        output_fs=condition_data.fs_new,
        resp_band=(0.02, 0.7),
    )
    print("EDR shape:", edr.shape)
    print("EDR times shape:", t_edr.shape)
    print("Respiration rate (BPM):", resp_rate_bpm)
    rr_intervals = features_extraction.calculate_rr_intervals(
        r_peaks=details["r_peaks"], time_edr=t_edr, fs=condition_data.fs
    )
    print("RR intervals shape:", rr_intervals.shape)
    edr_processed = features_extraction.edr_postprocessing(edr, t_edr)
    print("Processed EDR shape:", edr_processed.shape)

    # Plot the EDR and RR intervals
    fig_rr_edr = plot_rr_edr.plot_rr_edr(
        rr_int=rr_intervals, time_edr=t_edr, edr=edr_processed
    )
    _save_fig(fig_rr_edr, f"RR_EDR_plot_session{session_name}.png")

    # Calculate CRC metric
    crc_result = crc_wavelet.calculate_crc(
        rr_intervals=rr_intervals,
        edr=edr_processed,
        time_edr=t_edr,
        fs=condition_data.fs_new,
    )
    print("CRC result keys:", crc_result.keys())
    print("CRC shape", crc_result["CRC"].shape)

    energy_crc, avg_crc_hf, avg_wct_hf, details = crc_wavelet.calculate_average(
        crc=crc_result["CRC"], wct=crc_result["wct"], freqs=crc_result["freqs"]
    )
    print("Energy CRC shape:", energy_crc.shape)
    print("Average CRC (HF):", avg_crc_hf.shape)
    print("Details keys:", details.keys())
    print("HF band:", details["HF_band"])

    tau, DI = crc_wavelet.calculate_phase(
        CRC=crc_result["CRC"],
        awct=crc_result["phase"],
        freqs=crc_result["freqs"],
        HF_band=details["HF_band"],
    )
    print("tau shape; ", tau.shape)
    print("DI shape: ", DI.shape)

    # Plot the CRC results
    fig_crc = plot_crc.plot_crc(
        CRC=crc_result["CRC"],
        time_edr=t_edr,
        freqs=crc_result["freqs"],
        coi=crc_result["coi"],
        HF=details["HF_band"],
        energy_crc=energy_crc,
        avg_CRC_HF=avg_crc_hf,
        session_name=session_name,
    )
    _save_fig(fig_crc, f"CRC_session{session_name}.png")

    fig_tau_DI = plot_tau_di.plot_tau_di(
        tau=tau, DI=DI, time=t_edr, session_name=session_name
    )
    _save_fig(fig_tau_DI, f"tau_DI_session{session_name}.png")


if __name__ == "__main__":
    main()
