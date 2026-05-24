from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

def to_float32_audio(x: np.ndarray) -> np.ndarray:
    """Convert PCM audio to float32 in [-1, 1]."""
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float32)
    return x


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Normalize audio to avoid clipping."""
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return x
    return x / peak * 0.98


def design_bandpass_remez(fs: int,
                          f1: float = 300.0,
                          f2: float = 3000.0,
                          transition: float = 150.0,
                          numtaps: int = 401) -> np.ndarray:
    """
    Parks–McClellan band-pass FIR.
    Passband: [f1, f2]
    Stopbands: [0, f1-transition] and [f2+transition, fs/2]
    """
    nyq = fs / 2.0
    a = max(0.0, f1 - transition)
    b = min(nyq, f2 + transition)
    if a >= f1 or f2 >= b:
        raise ValueError("Invalid band edges for bandpass design.")
    bands = [0.0, a, f1, f2, b, nyq]
    desired = [0, 1, 0]
    weight = [10, 1, 10]
    taps = signal.remez(numtaps, bands, desired, weight=weight, fs=fs)
    return taps.astype(np.float64)


def design_hilbert_remez(fs: int,
                         passband_low: float = 300.0,
                         passband_high: float = 3000.0,
                         transition: float = 150.0,
                         numtaps: int = 401) -> np.ndarray:
    """
    Parks–McClellan Hilbert transformer FIR.
    It approximates the Hilbert transform on the speech band.
    """
    nyq = fs / 2.0
    lo = max(1.0, passband_low - transition)
    hi = min(nyq - 1.0, passband_high + transition)
    if lo >= passband_low or passband_high >= hi:
        raise ValueError("Invalid Hilbert passband edges.")

    # For type='hilbert', desired amplitude is 1 in the approximation band.
    taps = signal.remez(numtaps, [lo, hi], [1], type='hilbert', fs=fs)
    return taps.astype(np.float64)


def causal_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Causal FIR filtering with linear-phase delay compensation.
    The filter itself is causal; we align the output by removing
    the group delay and padding zeros at the end.
    """
    y = signal.lfilter(h, [1.0], x)
    delay = (len(h) - 1) // 2
    if delay > 0:
        y = np.concatenate([y[delay:], np.zeros(delay, dtype=y.dtype)])
    return y


def invert_spectrum(x: np.ndarray, fs: int, fa: float,
                    bp_taps: np.ndarray, h_taps: np.ndarray) -> np.ndarray:
    """
    Frequency inversion around fa:
        y[n] = x[n] * cos(2π fa n/fs) + H{x}[n] * sin(2π fa n/fs)
    """
    x_bp = causal_fir(x, bp_taps)
    x_h = causal_fir(x_bp, h_taps)

    n = np.arange(len(x_bp), dtype=np.float64)
    c = np.cos(2.0 * np.pi * fa * n / fs)
    s = np.sin(2.0 * np.pi * fa * n / fs)

    y = x_bp * c + x_h * s
    y = causal_fir(y, bp_taps)
    return y


def spectrum_db(x: np.ndarray, fs: int, nfft: int = 8192):
    X = np.fft.rfft(x, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, mag


# =========================
# Main pipeline
# =========================

def main():
    in_path = Path("data files/test.wav")
    if not in_path.exists():
        raise FileNotFoundError("Не найден test.wav в текущей папке.")

    fs, x = wavfile.read(in_path)
    x = to_float32_audio(x)

    # mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    nyq = fs / 2.0

    # Для частотной инверсии нужна несущая в диапазоне:
    # 3000 < fa < nyq - 300
    fa_min = 3000.0 + 1.0
    fa_max = nyq - 300.0 - 1.0

    if fa_max <= fa_min:
        raise ValueError(
            f"Частота дискретизации {fs} Гц слишком мала для схемы кодирования речи 300–3000 Гц."
        )

    # Берём середину допустимого интервала
    fa = 0.5 * (fa_min + fa_max)

    # FIR lengths: odd for linear-phase Hilbert transformer
    numtaps_bp = 401
    numtaps_h = 401

    bp_taps = design_bandpass_remez(fs, 300.0, 3000.0, transition=150.0, numtaps=numtaps_bp)
    h_taps = design_hilbert_remez(fs, 300.0, 3000.0, transition=150.0, numtaps=numtaps_h)

    # Encode / decode
    encoded = invert_spectrum(x, fs, fa, bp_taps, h_taps)
    decoded = invert_spectrum(encoded, fs, fa, bp_taps, h_taps)

    # Final speech band cleanup after decode
    decoded = causal_fir(decoded, bp_taps)

    # Normalize and save
    encoded_out = normalize_audio(encoded)
    decoded_out = normalize_audio(decoded)

    wavfile.write("encoded.wav", fs, np.int16(encoded_out * 32767))
    wavfile.write("decoded.wav", fs, np.int16(decoded_out * 32767))

    # Quantitative error
    min_len = min(len(x), len(decoded_out))
    x0 = x[:min_len]
    d0 = decoded_out[:min_len]
    mse = np.mean((x0 - d0) ** 2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(x0, d0)[0, 1]
    print(f"fs = {fs} Hz")
    print(f"carrier fa = {fa:.2f} Hz")
    print(f"RMSE = {rmse:.6e}")
    print(f"Pearson corr = {corr:.6f}")

    # Plots: waveform
    t = np.arange(min_len) / fs
    show_n = min(min_len, fs // 2)  # first 0.5 sec
    plt.figure(figsize=(12, 4))
    plt.plot(t[:show_n], x0[:show_n], label="original")
    plt.plot(t[:show_n], d0[:show_n], label="decoded", alpha=0.8)
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.title("Original vs decoded waveform")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plots: spectra
    f1, m1 = spectrum_db(x0, fs)
    f2, m2 = spectrum_db(encoded_out[:min_len], fs)
    f3, m3 = spectrum_db(d0, fs)

    plt.figure(figsize=(12, 4))
    plt.plot(f1, m1, label="original")
    plt.plot(f2, m2, label="encoded", alpha=0.8)
    plt.plot(f3, m3, label="decoded", alpha=0.8)
    plt.xlim(0, min(20000, fs / 2))
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Magnitude, dB")
    plt.title("Spectra")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
