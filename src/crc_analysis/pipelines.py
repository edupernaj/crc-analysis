from typing import Tuple

from pathlib import Path
from crc_analysis import features_extraction
from crc_analysis.input import DataLoaderEcg12
import crc_analysis.crc.wavelet as crc_wavelet
from crc_analysis.plotting import plot_crc, plot_rr_edr, plot_tau_di

def crc_assessment(
        data_root : Path,
        experiment_name : str,
        session_name : str,
        condition_name : str,
        fs_new : float,
        resp_band : Tuple[float, float],
        resp_freq : float,
):
    # Load ECG 12 data
    experiment = DataLoaderEcg12(experiment_name=experiment_name, data_root=data_root)
    condition_data = experiment[session_name][condition_name]
    filtered_data = condition_data.ecg
    # Extract features (EDR and RR intervals)
    edr, time_edr, resp_rate_bpm, feature_details = features_extraction.edr_extraction_simple(
        ecg12=filtered_data,
        fs=condition_data.fs,
        output_fs=fs_new,
        resp_band=resp_band,
    )
    rr_intervals = features_extraction.calculate_rr_intervals(
        r_peaks=feature_details["r_peaks"], time_edr=time_edr, fs=condition_data.fs
    )
    edr_processed = features_extraction.edr_postprocessing(edr, time_edr)
    # Plot the EDR and RR intervals
    fig_rr_edr = plot_rr_edr.plot_rr_edr(
        rr_int=rr_intervals, time_edr=time_edr, edr=edr_processed
    )
    # Calculate CRC metric
    crc_result = crc_wavelet.calculate_crc(
        rr_intervals=rr_intervals,
        edr=edr_processed,
        time_edr=time_edr,
        fs=fs_new,
    )
    energy_crc, avg_crc_hf, avg_wct_hf, crc_details = crc_wavelet.calculate_average(
        crc=crc_result["CRC"],
        wct=crc_result["wct"],
        freqs=crc_result["freqs"],
        resp_freq=resp_freq,
    )
    tau_score, di_score = crc_wavelet.calculate_phase(
        CRC=crc_result["CRC"],
        awct=crc_result["phase"],
        freqs=crc_result["freqs"],
        HF_band=crc_details["HF_band"],
    )
    # Plot the CRC results
    fig_crc = plot_crc.plot_crc(
        CRC=crc_result["CRC"],
        time_edr=time_edr,
        freqs=crc_result["freqs"],
        coi=crc_result["coi"],
        HF=crc_details["HF_band"],
        energy_crc=energy_crc,
        avg_CRC_HF=avg_crc_hf,
        session_name=session_name,
    )
    fig_tau_DI = plot_tau_di.plot_tau_di(
        tau=tau_score, DI=di_score, time=time_edr, session_name=session_name
    )
    # Return results
    return{
        "filtered_data": filtered_data,
        "edr" : edr,
        "time_edr" : time_edr,
        "resp_rate_bpm" : resp_rate_bpm,
        "feature_details" : feature_details,
        "rr_intervals" : rr_intervals,
        "edr_processed" : edr_processed,
        "fig_rr_edr" : fig_rr_edr,
        "crc_result" : crc_result,
        "energy_crc" : energy_crc,
        "avg_crc_hf" : avg_crc_hf,
        "avg_wct_hf" : avg_wct_hf,
        "crc_details" : crc_details,
        "tau_score" : tau_score,
        "di_score" : di_score,
        "fig_crc": fig_crc,
        "fig_tau_DI": fig_tau_DI,
    }
