from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_time_frequency_representation(
    signal: np.ndarray, time: np.ndarray, fs: float
) -> None:
    """
    Plots the time and frequency representation of a given signal using FFT.

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

    # Compute the FFT
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    freqs_shifted = np.abs(np.fft.fftshift(freqs))
    fft_magnitude = np.abs(np.fft.fftshift(fft_signal))
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    FMAX = 60  # Hz -- ECG spectral content of interest is mostly under ~40 Hz
    # Time domain plot
    ax1.plot(
        time, signal, color="0.2", linewidth=0.7
    )  # Plot a portion of the signal for clarity
    ax1.set_title("Time Domain Representation")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_yticklabels([])

    # Frequency domain plot (magnitude spectrum)
    # Use np.abs() to get the magnitude of the complex numbers
    ax2.plot(freqs_shifted, fft_magnitude, color="0.2", linewidth=0.7)
    ax2.set_title("Frequency Domain Representation")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, FMAX)  # Limit x-axis to 0-60 Hz for better visibility
    ax2.set_yticklabels([])

    labels = ["(a)", "(b)"]

    # Label the subplots with (a) and (b)
    for i, a in enumerate([ax1, ax2]):
        a.text(-0.08, -0.12, labels[i], transform=a.transAxes, fontsize=14, va="top")
    # Folder and filename
    output_folder = Path("thesis_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "fig_time_freqECG.pdf"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
