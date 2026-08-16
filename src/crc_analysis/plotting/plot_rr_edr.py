import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_rr_edr(rr_int: np.ndarray, time_edr: np.ndarray, edr: np.ndarray) -> Figure:
    """Plot the timestamp-aligned RR interval and EDR traces.

    Parameters
    ----------
    rr_int : np.ndarray
        Array of RR intervals in milliseconds.
    time_edr : np.ndarray
        Uniform time vector shared by the RR and EDR traces.
    edr : np.ndarray
        Array of EDR values.

    Returns
    -------
    Figure
        Matplotlib figure containing the aligned RR and EDR panels.
    """
    plt.rcParams.update(
        {
            # Figure and typography settings for diagnostic output.
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
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), constrained_layout=True)

    # Top panel: beat-to-beat interval in milliseconds.
    axes[0].plot(time_edr, rr_int, label="RR intervals [ms]", color="black")
    axes[0].set_title("RR intervals [ms]")
    axes[0].set_ylabel("RR intervals [ms]")
    axes[0].set_xlabel("Time [s]")
    axes[0].legend()

    # Bottom panel: ECG-derived respiratory surrogate.
    axes[1].plot(time_edr, edr, label="ECG derived respiration", color="black")
    axes[1].set_title("ECG derived respiration")
    axes[1].set_ylabel("ECG derived respiration")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    return fig
