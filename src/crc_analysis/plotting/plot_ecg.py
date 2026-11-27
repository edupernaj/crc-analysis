import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_ecg_rpeaks(ecg12, detection_lead, r_peaks, fs) -> Figure:

    plt.rcParams.update(
        {
            # Figure defaults.
            "figure.figsize": (12, 4),
            "figure.dpi": 200,
            "figure.facecolor": "white",
            # Font sizes.
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 14,
            # Axes appearance.
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.2",
            "axes.facecolor": "white",
            # Grid appearance.
            "axes.grid": True,
            "grid.linestyle": (0, (1, 5)),
            "grid.color": "0.65",
            "grid.linewidth": 0.6,
            # Tick direction and sizes.
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            # Line defaults.
            "lines.linewidth": 0.7,
            "lines.color": "0.2",
            # Figure-saving defaults.
            "savefig.bbox": "tight",
        }
    )
    time = np.arange(0, len(ecg12[1, :])) / fs
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.plot(time, ecg12[detection_lead, :], label="Detection lead ECG")
    ax.plot(time[r_peaks], ecg12[detection_lead, r_peaks], "or")

    return fig
