from pathlib import Path
import matplotlib.pyplot as plt


def plot_ecgleads(ecg12, time) -> None:
    """
    Plots 5 seconds snippets of timeseries of leads I,II,III of 12 lead ECG

    Parameters:
    - ecg12: 2-D array-like, 12-lead ecg siganl
    - time: 1-D array-like, time vector for the signal
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

    # plot ECG leads I,II,III
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(
        3, 1, left=0.09, right=0.94, bottom=0.09, top=0.93, hspace=0.4
    )
    ax_I = fig.add_subplot(gs[0, 0])
    ax_II = fig.add_subplot(gs[1, 0])
    ax_III = fig.add_subplot(gs[2, 0])

    ax_I.plot(time, ecg12[7, :], color="black", linewidth=0.5)
    ax_I.set_title("Lead I")
    ax_I.set_ylabel("Amplitude")
    ax_I.set_yticklabels([])
    ax_I.set_xticklabels([])

    ax_II.plot(time, ecg12[8, :], color="black", linewidth=0.5)
    ax_II.set_title("Lead II")
    ax_II.set_ylabel("Amplitude")
    ax_II.set_yticklabels([])
    ax_II.set_xticklabels([])

    ax_III.plot(time, ecg12[9, :], color="black", linewidth=0.5)
    ax_III.set_title("Lead III")
    ax_III.set_xlabel("Time (s)")
    ax_III.set_ylabel("Amplitude")
    ax_III.set_yticklabels([])

    output_folder = Path("thesis_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "ecg.pdf"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
