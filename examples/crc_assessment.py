from pathlib import Path
import matplotlib.pyplot as plt
from crc_analysis import features_extraction
from crc_analysis.input import DataLoaderEcg12
from crc_analysis.wavelet import crc
from crc_analysis.plotting import crc_plot, cardioresp_plot, tau_DI_plot


def main() -> None:
    # Load ECG 12 data
    experiment = DataLoaderEcg12("DRKS00034187")
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
        details["r_peaks"], t_edr, fs=condition_data.fs
    )
    print("RR intervals shape:", rr_intervals.shape)
    edr_processed = features_extraction.edr_postprocessing(edr, t_edr)
    print("Processed EDR shape:", edr_processed.shape)

    fig1 = cardioresp_plot.rr_int_resp_plothelper(rr_intervals, t_edr, edr_processed)
    fig1.show()
    output_folder = Path("thesis_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "RR_EDRplot.pdf"

    fig1.savefig(output_path, dpi=300, bbox_inches="tight")

    crc_result = crc.crc_calculation(
        rr_intervals, edr_processed, t_edr, fs=condition_data.fs_new
    )
    print("CRC result keys:", crc_result.keys())
    print("CRC shape", crc_result["CRC"].shape)
    en_crc, avg_crc_hf, avf_wct_HF, details = crc.crc_average(
        crc_result["CRC"], crc_result["wct"], crc_result["freqs"]
    )
    print("Energy CRC shape:", en_crc.shape)
    print("Average CRC (HF):", avg_crc_hf.shape)
    print("Details keys:", details.keys())
    print("HF band:", details["HF_band"])

    subj = 1  # subject identifier
    fig = crc_plot.CRCplothelper(
        CRC=crc_result["CRC"],
        tEdr=t_edr,
        freqs=crc_result["freqs"],
        coi=crc_result["coi"],
        HF=details["HF_band"],
        en_CRC=en_crc,
        avg_CRC_HF=avg_crc_hf,
        subj=subj,
    )

    tau, DI = crc.crc_phase(
        CRC=crc_result["CRC"],
        awct=crc_result["phase"],
        freqs=crc_result["freqs"],
        HF_band=details["HF_band"],
    )
    print("tau shape; ", tau.shape)
    print("DI shape: ", DI.shape)
    fig2 = tau_DI_plot.tau_DI_plot(tau=tau, DI=DI, time=t_edr, subj=subj)
    fig2.show()
    # Folder and filename
    output_folder = Path("CRC_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"CRC_{subj}.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
