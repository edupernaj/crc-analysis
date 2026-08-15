from pathlib import Path
import matplotlib.pyplot as plt
from crc_analysis.input import csv_reader, preprocessing
from crc_analysis.preprocessing import edr_simple
from crc_analysis.preprocessing import rr_intervals_calculation, edr_preprocessing
from crc_analysis.wavelet import crc
from crc_analysis.plotting import crc_plot, cardioresp_plot, tau_DI_plot


def main() -> None:
    csv_path = ""
    input_data = csv_reader.read_csv_data(str(csv_path))

    filtered_data = preprocessing.preprocess_ecg12(
        input_data["ecg12"], input_data["fs"]
    )

    print("Filtered data shape:", filtered_data.shape)
    edr, t_edr, resp_rate_bpm, details = edr_simple.ecg_derived_respiration_simple(
        ecg12=filtered_data,
        fs=input_data["fs"],
        output_fs=input_data["fs_new"],
        resp_band=(0.02, 0.7),
    )
    print("EDR shape:", edr.shape)
    print("EDR times shape:", t_edr.shape)
    print("Respiration rate (BPM):", resp_rate_bpm)
    rr_intervals = rr_intervals_calculation.calculate_rr_intervals(
        details["r_peaks"], t_edr, fs=input_data["fs"]
    )
    print("RR intervals shape:", rr_intervals.shape)
    edr_processed = edr_preprocessing.preprocessing_edr(edr, t_edr)
    print("Processed EDR shape:", edr_processed.shape)

    fig1 = cardioresp_plot.rr_int_resp_plothelper(rr_intervals, t_edr, edr_processed)
    fig1.show()
    output_folder = Path("thesis_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "RR_EDRplot.pdf"

    fig1.savefig(output_path, dpi=300, bbox_inches="tight")

    crc_result = crc.crc_calculation(
        rr_intervals, edr_processed, t_edr, fs=input_data["fs_new"]
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
