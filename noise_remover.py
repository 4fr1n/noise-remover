#!/usr/bin/env python3
"""
Audio Noise Remover
-------------------
Takes a WAV file as input, denoises it using spectral subtraction +
Wiener filtering, and outputs a clean WAV with frequency analysis plots.

HOW TO USE:
  The input WAV should have a few seconds of silence at the start —
  this is used as the noise profile. Then the rest of the audio is
  cleaned against that profile.

Usage:
    python noise_remover.py input.wav [options]

Options:
    --output FILE        Output WAV path (default: clean_output.wav)
    --noise-duration S   Seconds of silence at start for noise profile (default: 2.0)
    --sensitivity N      Noise subtraction strength, 0.5–3.0 (default: 1.5)
    --method M           spectral | wiener | both (default: both)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════════════════
#  DSP core
# ══════════════════════════════════════════════════════════════════════════════

def stft(signal: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    frames = []
    for start in range(0, len(signal) - n_fft + 1, hop):
        frames.append(np.fft.rfft(signal[start:start + n_fft] * win))
    return np.array(frames).T          # (freq_bins, time_frames)


def istft(spec: np.ndarray, hop: int, win: np.ndarray) -> np.ndarray:
    n_fft   = (spec.shape[0] - 1) * 2
    length  = (spec.shape[1] - 1) * hop + n_fft
    output  = np.zeros(length)
    win_sum = np.zeros(length)
    for i in range(spec.shape[1]):
        s = i * hop
        output [s:s + n_fft] += np.fft.irfft(spec[:, i])[:n_fft] * win
        win_sum[s:s + n_fft] += win ** 2
    return output / np.maximum(win_sum, 1e-8)


def estimate_noise_power(signal, sr, noise_duration, n_fft, hop, win):
    n = min(int(noise_duration * sr), len(signal))
    spec = stft(signal[:n], n_fft, hop, win)
    return np.mean(np.abs(spec) ** 2, axis=1, keepdims=True)   # (freq_bins, 1)


def spectral_subtraction(spec, noise_power, sensitivity, beta=0.02):
    mag   = np.abs(spec)
    phase = np.angle(spec)
    power = mag ** 2
    clean = np.maximum(power - sensitivity * noise_power, beta * power)
    return np.sqrt(clean) * np.exp(1j * phase)


def wiener_filter(spec, noise_power, sensitivity):
    power        = np.abs(spec) ** 2
    signal_power = np.maximum(power - sensitivity * noise_power, 0)
    snr          = signal_power / (noise_power + 1e-10)
    gain         = snr / (snr + 1.0)
    return gain * spec


def denoise_signal(signal: np.ndarray, sr: int,
                   noise_duration: float, sensitivity: float,
                   method: str) -> np.ndarray:
    n_fft = 2048
    hop   = n_fft // 4
    win   = np.hanning(n_fft)

    noise_power = estimate_noise_power(signal, sr, noise_duration, n_fft, hop, win)

    freq_bins = np.fft.rfftfreq(n_fft, 1 / sr)
    top_idx   = np.argsort(noise_power[:, 0])[::-1][:5]
    top_freqs = ", ".join(f"{freq_bins[i]:.0f} Hz" for i in top_idx)
    print(f"  Noisiest frequency bins: {top_freqs}")

    spec = stft(signal, n_fft, hop, win)

    if method == "spectral":
        clean_spec = spectral_subtraction(spec, noise_power, sensitivity)
    elif method == "wiener":
        clean_spec = wiener_filter(spec, noise_power, sensitivity)
    else:
        clean_spec = spectral_subtraction(spec, noise_power, sensitivity)
        clean_spec = wiener_filter(clean_spec, noise_power, sensitivity * 0.5)

    clean = istft(clean_spec, hop, win)[:len(signal)]

    rms_orig  = np.sqrt(np.mean(signal ** 2))
    rms_clean = np.sqrt(np.mean(clean ** 2))
    if rms_clean > 0:
        clean *= rms_orig / rms_clean

    return clean


# ══════════════════════════════════════════════════════════════════════════════
#  Frequency spectrum plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectra(original: np.ndarray, cleaned: np.ndarray,
                 sr: int, noise_duration: float) -> None:
    """
    Four-panel figure:
      Top-left:  Full frequency spectrum comparison (original vs clean)
      Top-right: Zoom into low frequencies 0–2 kHz
      Bottom-left:  Original spectrogram
      Bottom-right: Cleaned spectrogram
    """
    n_fft = 4096
    win   = np.hanning(n_fft)
    hop   = n_fft // 4

    def db_spectrum(sig):
        # Average magnitude spectrum in dB
        frames = []
        for s in range(0, len(sig) - n_fft + 1, hop):
            frames.append(np.abs(np.fft.rfft(sig[s:s + n_fft] * win)))
        mag = np.mean(np.array(frames), axis=0) if frames else np.zeros(n_fft // 2 + 1)
        return 20 * np.log10(mag + 1e-10)

    def db_spec_2d(sig):
        s = stft(sig, n_fft, hop, win)
        return 20 * np.log10(np.abs(s) + 1e-10)

    freqs    = np.fft.rfftfreq(n_fft, 1 / sr)
    orig_db  = db_spectrum(original)
    clean_db = db_spectrum(cleaned)

    orig_2d  = db_spec_2d(original)
    clean_2d = db_spec_2d(cleaned)

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor="#0f1117")
    fig.suptitle("Mic Noise Remover — Frequency Analysis",
                 fontsize=15, color="white", y=0.98)

    gs   = gridspec.GridSpec(2, 2, figure=fig,
                              hspace=0.38, wspace=0.30,
                              left=0.07, right=0.97,
                              top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    ORIG_COLOR  = "#4fc3f7"   # light blue
    CLEAN_COLOR = "#a5d6a7"   # light green
    BG          = "#1a1d27"
    GRID        = "#2e3250"

    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors="#aab0c8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    def label_ax(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, color="#aab0c8", fontsize=8)
        ax.set_ylabel(ylabel, color="#aab0c8", fontsize=8)
        ax.set_title(title,  color="white",    fontsize=10, pad=6)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # Panel 1 — full spectrum ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(freqs / 1000, orig_db,  color=ORIG_COLOR,  lw=0.9,
            label="Original", alpha=0.9)
    ax.plot(freqs / 1000, clean_db, color=CLEAN_COLOR, lw=0.9,
            label="Cleaned",  alpha=0.9)
    ax.axvspan(0, noise_duration / (len(original) / sr) * freqs[-1] / 1000,
               alpha=0.0)   # (just a placeholder; noise region not frequency-based)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID,
              labelcolor="white", loc="upper right")
    label_ax(ax, "Frequency (kHz)", "Magnitude (dB)",
             "Full Frequency Spectrum")
    ax.set_xlim(0, freqs[-1] / 1000)

    # Panel 2 — low-freq zoom 0–2 kHz ──────────────────────────────────────
    ax = axes[1]
    mask = freqs <= 2000
    ax.plot(freqs[mask], orig_db[mask],  color=ORIG_COLOR,  lw=1.0,
            label="Original", alpha=0.9)
    ax.plot(freqs[mask], clean_db[mask], color=CLEAN_COLOR, lw=1.0,
            label="Cleaned",  alpha=0.9)
    ax.fill_between(freqs[mask], orig_db[mask], clean_db[mask],
                    where=(orig_db[mask] > clean_db[mask]),
                    color="#ef5350", alpha=0.25, label="Noise removed")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID,
              labelcolor="white", loc="upper right")
    label_ax(ax, "Frequency (Hz)", "Magnitude (dB)",
             "Low-Frequency Zoom  (0 – 2 kHz)")
    ax.set_xlim(0, 2000)

    # Panel 3 — original spectrogram ───────────────────────────────────────
    ax   = axes[2]
    dur  = len(original) / sr
    vmin, vmax = -80, 0
    im3  = ax.imshow(orig_2d, origin="lower", aspect="auto",
                     cmap="magma", vmin=vmin, vmax=vmax,
                     extent=[0, dur, freqs[0] / 1000, freqs[-1] / 1000])
    ax.axvline(x=noise_duration, color="#ff8a65", lw=1.2, ls="--",
               label=f"Noise sample end ({noise_duration}s)")
    ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor="white")
    plt.colorbar(im3, ax=ax, label="dB").ax.yaxis.label.set_color("#aab0c8")
    label_ax(ax, "Time (s)", "Frequency (kHz)", "Original Spectrogram")

    # Panel 4 — cleaned spectrogram ────────────────────────────────────────
    ax  = axes[3]
    im4 = ax.imshow(clean_2d, origin="lower", aspect="auto",
                    cmap="magma", vmin=vmin, vmax=vmax,
                    extent=[0, dur, freqs[0] / 1000, freqs[-1] / 1000])
    plt.colorbar(im4, ax=ax, label="dB").ax.yaxis.label.set_color("#aab0c8")
    label_ax(ax, "Time (s)", "Frequency (kHz)", "Cleaned Spectrogram")

    plot_path = "spectrum_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"  Spectrum plot saved → {plot_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Denoise a WAV file using spectral subtraction + Wiener filter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument("--output", default="clean_output.wav",
                        help="Output WAV file (default: clean_output.wav)")
    parser.add_argument("--noise-duration", type=float, default=2.0,
                        metavar="SECS",
                        help="Seconds of silence at start for noise profile (default: 2.0)")
    parser.add_argument("--sensitivity", type=float, default=1.5,
                        help="Noise subtraction strength 0.5–3.0 (default: 1.5)")
    parser.add_argument("--method", choices=["spectral", "wiener", "both"],
                        default="both",
                        help="Denoising algorithm (default: both)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    # ── 1. Load ────────────────────────────────────────────────────────────
    print("═" * 55)
    print("  Audio Noise Remover")
    print("═" * 55)
    print(f"  Input          : {args.input}")
    print(f"  Noise profile  : first {args.noise_duration}s")
    print(f"  Method         : {args.method}")
    print(f"  Sensitivity    : {args.sensitivity}")
    print(f"  Output         : {args.output}")

    audio, sr = sf.read(args.input, always_2d=False)

    # Convert stereo to mono by averaging channels
    if audio.ndim == 2:
        print(f"\n  Stereo file detected — converting to mono")
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)
    duration = len(audio) / sr
    print(f"\n  Loaded {duration:.1f}s of audio ({len(audio):,} samples @ {sr} Hz)")

    if duration < args.noise_duration + 0.5:
        print(f"\nWarning: file is very short. Need at least "
              f"{args.noise_duration + 0.5:.1f}s. Consider a shorter --noise-duration.")

    # ── 2. Denoise ─────────────────────────────────────────────────────────
    print("\n[Denoising]")
    cleaned = denoise_signal(
        audio,
        sr=sr,
        noise_duration=args.noise_duration,
        sensitivity=args.sensitivity,
        method=args.method,
    )

    sf.write(args.output, cleaned, sr)
    print(f"  Clean audio saved  → {args.output}")

    # ── 3. Stats ───────────────────────────────────────────────────────────
    # Use the middle 1s of the noise window to avoid edge effects from
    # RMS normalisation and STFT boundary artefacts
    mid      = int(args.noise_duration * sr / 2)
    half_sec = int(0.5 * sr)
    noise_raw   = audio  [max(0, mid - half_sec) : mid + half_sec]
    noise_clean = cleaned[max(0, mid - half_sec) : mid + half_sec]

    rms_raw   = 20 * np.log10(np.sqrt(np.mean(noise_raw   ** 2)) + 1e-10)
    rms_clean = 20 * np.log10(np.sqrt(np.mean(noise_clean ** 2)) + 1e-10)

    # Signal preservation — compare RMS of the speech/signal portion
    sig_raw   = audio  [int(args.noise_duration * sr):]
    sig_clean = cleaned[int(args.noise_duration * sr):]
    rms_sig_raw   = 20 * np.log10(np.sqrt(np.mean(sig_raw   ** 2)) + 1e-10)
    rms_sig_clean = 20 * np.log10(np.sqrt(np.mean(sig_clean ** 2)) + 1e-10)

    print(f"\n  Noise floor (raw)     : {rms_raw:.1f} dB")
    print(f"  Noise floor (clean)   : {rms_clean:.1f} dB")
    print(f"  Noise floor reduction : {rms_raw - rms_clean:.1f} dB")
    print(f"\n  Signal RMS (raw)      : {rms_sig_raw:.1f} dB")
    print(f"  Signal RMS (clean)    : {rms_sig_clean:.1f} dB")
    print(f"  Signal preserved      : {abs(rms_sig_raw - rms_sig_clean):.1f} dB difference")

    # ── 4. Plot ────────────────────────────────────────────────────────────
    print("\n[Generating frequency spectrum plots …]")
    plot_spectra(audio, cleaned, sr, args.noise_duration)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
