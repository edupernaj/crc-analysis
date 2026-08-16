import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_crc(
    CRC, time_edr, freqs, coi, HF, energy_crc, avg_CRC_HF, session_name
) -> Figure:
    """
    Create a comprehensive plot for Cardiorespiratory Coupling (CRC) analysis.

    Parameters
    ----------
    CRC : np.ndarray
        2D array of CRC values.
    time_edr : np.ndarray
        Time vector corresponding to the CRC values.
    freqs : np.ndarray
        Array of frequencies corresponding to the CRC values.
    coi : np.ndarray
        Cone of influence values for the wavelet transform.
    HF : tuple
        Frequency band for high-frequency analysis (e.g., (0.15, 0.4)).
    energy_crc : np.ndarray
        Array of energy CRC values for each frequency.
    avg_CRC_HF : np.ndarray
        Average CRC values within the specified high-frequency band.
    session_name : str
        Session identifier for the plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object containing the CRC analysis plots.

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
    mapPos = np.array([0.08, 0.35, 0.7, 0.6])
    yPos = np.array([0.81, 0.35, 0.14, 0.6])
    xPos = np.array([0.145, 0.08, 0.635, 0.2])
    times_change = np.arange(60, 481, 60)

    figprops = dict(figsize=(15, 10), dpi=72)
    fig = plt.figure(**figprops)
    levels = np.linspace(CRC.min(), CRC.max(), 100)

    # The main panel shows where the coupling is strong in frequency and time.
    ax = plt.axes(mapPos)
    pcm = ax.contourf(time_edr, freqs, CRC, levels=levels, cmap="viridis")
    fig.colorbar(
        pcm,
        ax=ax,
        label="CRC",
        location="left",
        fraction=0.03,
        pad=0.065,
        panchor=False,
    )

    # The cone of influence marks the time-frequency region where edge effects are large.
    coi_freq = 1 / coi
    ax.plot(time_edr, coi_freq, "w--", linewidth=2)

    ax.axhline(HF[0], color="white", linestyle="--", label="Narrow HF band")
    ax.axhline(HF[1], color="white", linestyle="--", label="Narrow HF band")
    ax.annotate(
        "Narrow HF band",
        xy=(time_edr[-1] - 60, HF[1] - 0.015),
        xytext=(time_edr[-1] - 60, HF[1] - 0.015),
        color="white",
    )

    # Masking the unreliable edge region keeps the reader from over-interpreting artifacts.
    ax.fill_between(time_edr, 0, coi_freq, color="white", alpha=0.3, hatch="x")

    for i in times_change:
        ax.axvline(i, color="white", linestyle="--")
        if (i / 60) % 2 == 1:
            ax.annotate(
                "no pain", xy=(i - 8, freqs.max() - 0.055), rotation=90, color="white"
            )
        else:
            ax.annotate(
                "pain", xy=(i - 8, freqs.max() - 0.03), rotation=90, color="white"
            )
    ax.set(
        title=f"Cardiorespiratory coupling through wavelet analysis - Session {session_name}",
        ylabel="Frequency [Hz]",
        ylim=(min(freqs), max(freqs)),
        xlim=(min(time_edr), max(time_edr)),
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

    # This secondary panel reduces the full map to a single frequency profile for quick reading.
    bx = plt.axes(yPos, sharey=ax)
    bx.plot(energy_crc, freqs, label="Energy of CRC")
    bx.axhline(HF[0], color="black", linestyle="--", label="Narrow HF band")
    bx.axhline(HF[1], color="black", linestyle="--")
    bx.annotate(
        "Narrow HF band",
        xy=(energy_crc.max() - 25, HF[1] - 0.015),
        xytext=(energy_crc.max() - 25, HF[1] - 0.015),
        color="black",
    )
    bx.set(title="Average CRC across frequencies", xlabel="Average CRC")

    # The last panel shows whether coupling in the respiratory band changes over the recording.
    cx = plt.axes(xPos, sharex=ax)
    cx.plot(time_edr, avg_CRC_HF, label="Average CRC in narrow HF band", color="red")
    for i in times_change:
        cx.axvline(i, color="black", linestyle="--")
        if (i / 60) % 2 == 1:
            cx.annotate(
                "no pain",
                xy=(i - 8, avg_CRC_HF.max() * 0.81),
                rotation=90,
                color="black",
            )
        else:
            cx.annotate(
                "pain", xy=(i - 8, avg_CRC_HF.max() * 0.9), rotation=90, color="black"
            )
    cx.set(
        title="Average CRC in narrow HF band", xlabel="Time [s]", ylabel="Average CRC"
    )
    return fig
