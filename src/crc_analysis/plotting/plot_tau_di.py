import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_tau_di(
    tau: np.ndarray, DI: np.ndarray, time: np.ndarray, session_name: str
) -> Figure:
    """Plot wavelet phase delay and its smoothed directionality index.

    Parameters
    ----------
    tau
        Time delay of CRC in the selected HF band.
    DI
        NaN-aware smoothed sign of the phase delay.
    time
        Time vector shared by both traces.
    session_name
        Session identifier used in the title.
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

    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()

    # Tau is the delay estimate; DI summarizes its direction over time.
    ax.plot(time, tau, "-b", label="tau")
    ax.plot(time, DI, "*r", label="DI")
    ax.set_xlabel("Time [s]")
    ax.set_title(f"tau and DI - Session {session_name}")
    ax.legend()

    return fig
