from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet


def plot_cwt(signal: np.ndarray, time: np.ndarray, fs: float) -> plt.figure:
    """
    Plots the Continuous Wavelet Transform (CWT) of a given signal.

    Parameters:
    - signal: 1D array-like, the input signal to analyze.
    - time: 1D array-like, the time vector for the signal.
    - fs: float, the sampling frequency of the signal.
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
    # ensure signal, time and fs are valid
    signal = np.asarray(signal)
    time = np.asarray(time)
    if len(signal) != len(time):
        raise ValueError("Signal and time vector must have the same length.")
    fs = float(fs)

    # WAVELET ANALYSIS
    mother = wavelet.Morlet(6)
    dt = 1 / fs  # Time step based on the sampling frequency

    # Frequency limits
    fmin = 0.02
    fmax = 0.6

    # Wavelet coherence
    dj = 1 / 12  # 12 frequency/scale steps per octave

    s0 = 1 / (mother.flambda() * fmax)
    J = int(np.ceil(np.log2(fmax / fmin) / dj))

    # Compute CWT
    wct, scales, freqs, coi, _, _ = wavelet.cwt(
        signal=signal - signal.mean(), dt=dt, dj=dj, s0=s0, J=J, wavelet=mother
    )

    # Plotting
    fig = plt.figure(figsize=(15, 10), dpi=72)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(6, 0.15),
        height_ratios=(2, 6),
        left=0.09,
        right=0.94,
        bottom=0.09,
        top=0.93,
        wspace=0.04,
        hspace=0.18,
    )
    ax_wave = fig.add_subplot(gs[0, 0])  # left: ECG waveform (vertical)
    ax_cwt = fig.add_subplot(gs[1, 0], sharex=ax_wave)  # center: CWT heatmap
    cax = fig.add_subplot(gs[1, 1])  # right: colorbar

    ax_wave.plot(time, np.real(signal), color="black", lw=0.5)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")

    pcm = ax_cwt.pcolormesh(
        time, freqs, np.abs(wct) ** 2, cmap="viridis", shading="auto"
    )
    ax_cwt.set_title("Continuous Wavelet Transform (CWT)")
    ax_cwt.set_xlabel("Time (s)")
    ax_cwt.set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label("|W(f,t)|")
    # Cone of influence converted from period to frequency
    coi_freq = 1 / coi
    ax_cwt.plot(time, coi_freq, "w--", linewidth=2)

    # Shade outside COI
    ax_cwt.fill_between(time, 0, coi_freq, color="white", alpha=0.3, hatch="x")
    ax_cwt.set_ylim(freqs[-1], freqs[0])

    # plot narrow HF band
    ax_cwt.axhline(0.15, color="white", linestyle="--", label="Narrow HF band")
    ax_cwt.axhline(0.25, color="white", linestyle="--", label="Narrow HF band")
    ax_cwt.annotate(
        "Narrow HF band",
        xy=(time[-1] - 60, 0.25 - 0.015),
        xytext=(time[-1] - 60, 0.25 - 0.015),
        color="white",
    )

    fig.suptitle("ECG Time-Frequency Representation (CWT)")
    labels = ["(a)", "(b)"]

    for i, a in enumerate([ax_wave, ax_cwt]):
        a.text(-0.08, -0.12, labels[i], transform=a.transAxes, fontsize=14, va="top")
    # Folder and filename
    output_folder = Path("thesis_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "cwt_rrinterval.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
