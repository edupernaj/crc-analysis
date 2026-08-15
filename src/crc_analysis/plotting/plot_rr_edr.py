import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_rr_edr(
    rr_int: np.ndarray, time_edr: np.ndarray, edr: np.ndarray
) -> Figure:
    """
    Plot RR intervals and ECG derived respiration.

    Parameters
    ----------
    rr_int : np.ndarray
        Array of RR intervals in milliseconds.
    time_edr : np.ndarray
        Time vector for the resampled EDR waveform.
    edr : np.ndarray
        Array of EDR values.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the RR intervals and EDR plots.
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.figsize": (12, 4),
            "figure.dpi": 200,
            "figure.facecolor": "white",
            # Fonts
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # Axes
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.2",
            "axes.facecolor": "white",
            # Grid
            "axes.grid": True,
            "grid.linestyle": (0, (1, 5)),
            "grid.color": "0.65",
            "grid.linewidth": 0.6,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            # Lines
            "lines.linewidth": 0.7,
            "lines.color": "0.2",
            # Save
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), constrained_layout=True)
    axes[0].plot(time_edr, rr_int, label="RR intervals [mS]", color="black")
    axes[0].set_title("RR intervals [ms]")
    axes[0].set_ylabel("RR intervals [ms]")
    axes[0].set_xlabel("Time [s]")
    axes[0].legend

    axes[1].plot(time_edr, edr, label="ECG derived respiration", color="black")
    axes[1].set_title("ECG derived respiration")
    axes[1].set_ylabel("ECG derived respiration")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend
    return fig
