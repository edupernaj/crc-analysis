import matplotlib.pyplot as plt
import numpy as np


def CRCplothelper(CRC, tEdr, freqs, coi, HF, en_CRC, avg_CRC_HF, subj) -> plt.Figure:
    """
    Create a comprehensive plot for Cardiorespiratory Coupling (CRC) analysis.

    Parameters
    ----------
    CRC : np.ndarray
        2D array of CRC values.
    tEdr : np.ndarray
        Time vector corresponding to the CRC values.
    freqs : np.ndarray
        Array of frequencies corresponding to the CRC values.
    coi : np.ndarray
        Cone of influence values for the wavelet transform.
    HF : tuple
        Frequency band for high-frequency analysis (e.g., (0.15, 0.4)).
    en_CRC : np.ndarray
        Array of energy CRC values for each frequency.
    avg_CRC_HF : np.ndarray
        Average CRC values within the specified high-frequency band.
    subj : str
        Subject identifier for the plot title.

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
    yPos = np.array([0.81, 0.35, 0.14, 0.6])  # narrower right plot
    xPos = np.array([0.145, 0.08, 0.635, 0.2])
    times_change = np.arange(60, 481, 60)

    figprops = dict(figsize=(15, 10), dpi=72)
    fig = plt.figure(**figprops)
    levels = np.linspace(CRC.min(), CRC.max(), 100)
    # first subplot colormap of CRC
    ax = plt.axes(mapPos)
    pcm = ax.contourf(tEdr, freqs, CRC, levels=levels, cmap="viridis")
    fig.colorbar(
        pcm,
        ax=ax,
        label="CRC",
        location="left",
        fraction=0.03,
        pad=0.065,
        panchor=False,
    )

    # Cone of influence converted from period to frequency
    coi_freq = 1 / coi
    ax.plot(tEdr, coi_freq, "w--", linewidth=2)

    # plot narrow HF band
    ax.axhline(HF[0], color="white", linestyle="--", label="Narrow HF band")
    ax.axhline(HF[1], color="white", linestyle="--", label="Narrow HF band")
    ax.annotate(
        "Narrow HF band",
        xy=(tEdr[-1] - 60, HF[1] - 0.015),
        xytext=(tEdr[-1] - 60, HF[1] - 0.015),
        color="white",
    )

    # Shade outside COI
    ax.fill_between(tEdr, 0, coi_freq, color="white", alpha=0.3, hatch="x")

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

    # final touches to first subplot
    ax.set(
        title=f"Cardiorespiratory coupling through wavelet analysis - Subject {subj}",
        ylabel="Frequency [Hz]",
        ylim=(min(freqs), max(freqs)),
        xlim=(min(tEdr), max(tEdr)),
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

    # second subplot of energy of CRC across time
    bx = plt.axes(yPos, sharey=ax)
    bx.plot(en_CRC, freqs, label="Energy of CRC")
    bx.axhline(HF[0], color="black", linestyle="--", label="Narrow HF band")
    bx.axhline(HF[1], color="black", linestyle="--")
    bx.annotate(
        "Narrow HF band",
        xy=(en_CRC.max() - 25, HF[1] - 0.015),
        xytext=(en_CRC.max() - 25, HF[1] - 0.015),
        color="black",
    )
    bx.set(title="Average CRC across frequencies", xlabel="Average CRC")

    # third subplot of average CRC in narrow HF band
    cx = plt.axes(xPos, sharex=ax)
    cx.plot(tEdr, avg_CRC_HF, label="Average CRC in narrow HF band", color="red")
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
