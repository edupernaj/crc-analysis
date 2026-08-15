from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def plot_time_frequency_representation(
    signal: np.ndarray, time: np.ndarray, fs: float
) -> None:
    """
    Plots the time and frequency representation of a given signal using STFT.

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

    print(
        f"Signal length: {len(signal)}, Time vector length: {len(time)}, Sampling frequency: {fs} Hz"
    )
    # ============================================================
    # 2. STFT -> one frequency-spectrum "row" per time sample
    #    (noverlap = nperseg-1 gives a step of 1 sample -> Time-res = 1)
    # ============================================================

    _, N = len(signal) / fs, len(signal)
    nperseg = 512
    noverlap = nperseg - 8

    f, tt, Zxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    S = np.abs(Zxx)
    S = S / S.max()  # normalize 0-1
    print(
        f"STFT shape: {S.shape}, time vector length: {len(tt)}, time vector min-max values: {tt[0]} to {tt[-1]}, frequency vector length: {len(f)}"
    )

    # ============================================================
    # 3. Whole-signal Fourier transform, for the bottom panel
    # ============================================================
    freqs_full = np.fft.fftfreq(N, d=1 / fs)
    fft_mag = np.abs(np.fft.fft(signal))
    fft_mag = fft_mag / fft_mag.max()
    print(f"FFT shape: {fft_mag.shape}, frequency vector length: {len(freqs_full)}")
    # ============================================================
    # 4. Figure layout
    # ============================================================
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(1, 6, 0.15),
        height_ratios=(6, 1.6),
        left=0.09,
        right=0.94,
        bottom=0.09,
        top=0.93,
        wspace=0.04,
        hspace=0.18,
    )

    ax_wave = fig.add_subplot(gs[0, 0])  # left: ECG waveform (vertical)
    ax_spec = fig.add_subplot(gs[0, 1], sharey=None)  # center: spectrogram heatmap
    ax_fft = fig.add_subplot(gs[1, 1], sharex=ax_spec)  # bottom: FFT magnitude
    cax = fig.add_subplot(gs[0, 2])  # right: colorbar
    FMAX = 60  # Hz -- ECG spectral content of interest is mostly under ~40 Hz

    # --- center: STFT heatmap ---
    pcm = ax_spec.pcolormesh(f, tt, S.T, shading="gouraud", cmap="grey")
    ax_spec.set_xlim(0, FMAX)
    # ax_spec.set_ylim(time[0], time[-1])
    plt.setp(ax_spec.get_yticklabels(), visible=False)
    ax_spec.tick_params(axis="y", length=0)

    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label("STFT Magnitude (normalized)")

    # --- left: raw ECG signal, plotted vertically (time on shared y-axis) ---
    ax_wave.plot(signal, time, color="black", lw=0.5)
    ax_wave.set_ylim(time[0], time[-1])
    ax_wave.set_ylabel("Time (s)")
    ax_wave.set_xticks([])
    ax_wave.invert_xaxis()
    ax_wave.spines[["top", "right", "bottom"]].set_visible(False)

    # --- bottom: whole-signal FFT magnitude, sharing the frequency axis ---
    ax_fft.plot(freqs_full, fft_mag, color="black", lw=0.7)
    ax_fft.set_xlim(0, FMAX)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("|X(f)|")

    fig.suptitle("ECG Time-Frequency Representation (STFT)", y=0.985)
    # Folder and filename
    output_folder = Path("thesis_plots")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "fig_tfECG.pdf"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
