from pathlib import Path

import numpy as np
import pandas as pd

from crc_analysis.input import csv_reader


def test_read_ecg12_data_demo():
    """Check that the standard ECG CSV reader loads the file in the expected array layout."""
    test_fs = 150
    csv_path = (
        Path(__file__).resolve().parents[1] / "test_data" / "demo_data_ecg12_2.csv"
    )

    # This is the normal input flow: read the CSV, then check the time axis and 12-lead matrix.
    result = csv_reader.read_ecg12_data(csv_path, fs=test_fs)

    assert isinstance(result, dict)
    assert result["fs"] == test_fs

    data = result["data"]
    time = result["time"]
    ecg12 = result["ecg12"]

    assert data.shape[1] == 12
    assert data.shape[0] == time.shape[0]
    assert ecg12.shape == (12, data.shape[0])
    assert np.allclose(ecg12[0], data["V1"].to_numpy())
    assert np.allclose(time, np.arange(data.shape[0]) / test_fs)


def test_read_ecg12_data_missing_file():
    """Make sure a missing file fails in a clear and explicit way."""
    missing_path = Path("missing_ecg12.csv")
    try:
        csv_reader.read_ecg12_data(missing_path)
    except FileNotFoundError as error:
        assert str(missing_path) in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing CSV file")
