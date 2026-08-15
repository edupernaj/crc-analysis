import matplotlib.pyplot as plt
import numpy as np


def tau_DI_plot(
    tau: np.ndarray, DI: np.ndarray, time: np.ndarray, subj: float
) -> plt.figure:
    """
    Create a figure plot for tau and DI over time.

    Parameters
    ----------
    tau: np.ndarray
        time delay of CRC in HF band
    DI: np.ndarray
        Directionality index of time delay
    time: np.ndarray
        time vector
    subj: float
        subject number
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
            "figure.titlesize": 14,
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

    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.plot(time, tau, "-b", label="tau")
    ax.plot(time, DI, "*r", label="DI")
    ax.set_xlabel("Time [s]")
    ax.set_title("tau and DI")

    return fig
