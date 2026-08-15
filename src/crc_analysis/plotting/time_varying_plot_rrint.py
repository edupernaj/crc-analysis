from crc_analysis.input import csv_reader, preprocessing
from crc_analysis.preprocessing import edr_simple
from crc_analysis.preprocessing import rr_intervals_calculation, edr_preprocessing
from crc_analysis.plotting import cwt_plot


def main() -> None:
    csv_path = "/Users/joseeduardopereznajar/Documents/Study/BME HAW/thesis/MATLAB/Kubios-pain-nopain/Measurement/2/CPT mit PB ohne Thermometer + Pumpe/CPT mit PB ohne Thermometer + Pumpe2.txt"
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
    fig = cwt_plot.plot_cwt(signal=rr_intervals, time=t_edr, fs=input_data["fs_new"])
    fig.plot()


if __name__ == "__main__":
    main()
