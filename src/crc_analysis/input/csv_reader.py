import pandas as pd
import numpy as np


def read_csv_data(filepath: str, fs: int = 250) -> dict:
    """
    Reads CSV data from a file and returns a dictionary containing the data,
    sampling frequency, resampled frequency, time array, and ECG12 array.
    """
    fs_new = 4  # resampled frequency
    data = pd.read_table(filepath, sep=";")

    time = np.arange(0, len(data["V1"])) / fs
    ecg12 = data.values
    if ecg12.shape[0] != 12 and ecg12.shape[1] != 12:
        raise ValueError("Input CSV must contain 12 leads of ECG data.")
    if ecg12.shape[1] == 12 and ecg12.shape[0] != 12:
        ecg12 = np.transpose(ecg12)

    return {"data": data, "fs": fs, "fs_new": fs_new, "time": time, "ecg12": ecg12}
