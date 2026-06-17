from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


# ------------------------------------------------------------
# Пути
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data files" / "test.wav"
ENCODED_PATH = BASE_DIR / "encoded.wav"
DECODED_PATH = BASE_DIR / "decoded.wav"


# ------------------------------------------------------------
# Функции
# ------------------------------------------------------------
def pcm_to_float(x: np.ndarray) -> np.ndarray:
    """PCM -> float32."""
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        return x.astype(np.float32) / max(abs(info.min), info.max)
    return x.astype(np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    """Нормировка для сохранения без клиппинга."""
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return x.astype(np.float32)
    return (0.98 * x / peak).astype(np.float32)


def choose_carrier(fs: int) -> float:
    """Выбор несущей fa."""
    nyq = fs / 2.0
    fa = 3500.0
    if 3000.0 < fa < nyq - 300.0:
        return fa

    fa_min = 3001.0
    fa_max = nyq - 301.0
    if fa_max <= fa_min:
        raise ValueError(f"fs={fs} слишком мала для полосы 300–3000 Гц.")

    return 0.5 * (fa_min + fa_max)


def design_bandpass(fs: int, numtaps: int = 401) -> np.ndarray:
    """Полосовой КИХ-фильтр 300–3000 Гц."""
    f1, f2, tr = 300.0, 3000.0, 5.0
    nyq = fs / 2.0
    a = max(0.0, f1 - tr)
    b = min(nyq, f2 + tr)

    bands = [0.0, a, f1, f2, b, nyq]
    desired = [0, 1, 0]
    weight = [10, 1, 10]
    return signal.remez(numtaps, bands, desired, weight=weight, fs=fs).astype(np.float64)


def design_hilbert(fs: int, numtaps: int = 201, guard_hz: float = 200.0) -> np.ndarray:
    """Широкополосный преобразователь Гильберта."""
    nyq = fs / 2.0
    lo = guard_hz
    hi = nyq - guard_hz
    if hi <= lo:
        raise ValueError("Слишком малая частота дискретизации для Hilbert-фильтра.")

    return signal.remez(
        numtaps,
        [lo, hi],
        [1],
        type="hilbert",
        fs=fs,
        maxiter=100,
        grid_density=32,
    ).astype(np.float64)


def fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Каузальная FIR-фильтрация."""
    return signal.lfilter(h, [1.0], x)


def left_shift_zeros(x: np.ndarray, n: int) -> np.ndarray:
    """Сдвиг влево с добивкой нулями справа."""
    if n <= 0:
        return x
    if n >= len(x):
        return np.zeros_like(x)
    return np.hstack([x[n:], np.zeros(n, dtype=x.dtype)])


def total_delay(bp_taps: np.ndarray, h_taps: np.ndarray) -> int:
    """Суммарная задержка тракта."""
    d_bp = (len(bp_taps) - 1) // 2
    d_h = (len(h_taps) - 1) // 2
    return 2 * d_bp + d_h


def shift_spectrum(x: np.ndarray, fs: int, fa: float, bp_taps: np.ndarray, h_taps: np.ndarray) -> np.ndarray:
    """
    Частотная инверсия:
        y[n] = x[n] cos(2πfa n/fs) + H{x}[n] sin(2πfa n/fs)
    """
    x_bp = fir(x, bp_taps)
    x_h = fir(x_bp, h_taps)

    n = np.arange(len(x), dtype=np.float64)
    c = np.cos(2.0 * np.pi * fa * n / fs)
    s = np.sin(2.0 * np.pi * fa * n / fs)

    y = x_bp * c + x_h * s
    y = fir(y, bp_taps)

    return left_shift_zeros(y, total_delay(bp_taps, h_taps)).astype(np.float32)


def spectrum_db(x: np.ndarray, fs: int, nfft: int = 8192):
    X = np.fft.rfft(x, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, mag


def freq_response(h: np.ndarray, fs: int, nfft: int = 8192):
    w, H = signal.freqz(h, worN=nfft, fs=fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))
    return w, mag, phase


# ------------------------------------------------------------
# Основной код
# ------------------------------------------------------------
fs, x = wavfile.read(INPUT_PATH)
x = pcm_to_float(x)

if x.ndim == 2:
        x = x.mean(axis=1)

fa = choose_carrier(fs)

bp_taps = design_bandpass(fs, numtaps=401)
h_taps = design_hilbert(fs, numtaps=201, guard_hz=200.0)

encoded = shift_spectrum(x, fs, fa, bp_taps, h_taps)
decoded = shift_spectrum(encoded, fs, fa, bp_taps, h_taps)

# Сохранение
wavfile.write(ENCODED_PATH, fs, normalize(encoded))
wavfile.write(DECODED_PATH, fs, normalize(decoded))

# Оценка
n = min(len(x), len(decoded))
x0 = x[:n]
d0 = decoded[:n]

mse = np.mean((x0 - d0) ** 2)
rmse = np.sqrt(mse)
corr = np.corrcoef(x0, d0)[0, 1]

print(f"fs = {fs} Hz")
print(f"fa = {fa:.2f} Hz")
print(f"MSE  = {mse:.6e}")
print(f"RMSE = {rmse:.6e}")
print(f"Corr = {corr:.6f}")

# --------------------------------------------------------
# Графики
# --------------------------------------------------------
fb, mb, pb = freq_response(bp_taps, fs)
fh, mh, ph = freq_response(h_taps, fs)

plt.figure(figsize=(18, 8))

plt.subplot(2, 2, 1)
plt.plot(fb, mb)
plt.title("АЧХ полосового фильтра")
plt.xlabel("Hz")
plt.ylabel("dB")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(fh, mh)
plt.title("АЧХ Hilbert-фильтра")
plt.xlabel("Hz")
plt.ylabel("dB")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(fb, pb)
plt.title("ФЧХ полосового фильтра")
plt.xlabel("Hz")
plt.ylabel("rad")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(fh, ph)
plt.title("ФЧХ Hilbert-фильтра")
plt.xlabel("Hz")
plt.ylabel("rad")
plt.grid(True)

plt.tight_layout()
plt.show()

t = np.arange(n) / fs
show_n = min(n, fs // 2)

plt.figure(figsize=(18, 6))
plt.plot(t[:show_n], x0[:show_n], label="original")
plt.plot(t[:show_n], encoded[:show_n], label="encoded", alpha=0.8)
plt.plot(t[:show_n], d0[:show_n], label="decoded", alpha=0.8)
plt.title("Сигналы во времени")
plt.xlabel("s")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

f1, m1 = spectrum_db(x0, fs)
f2, m2 = spectrum_db(encoded[:n], fs)
f3, m3 = spectrum_db(d0, fs)

plt.figure(figsize=(18, 6))
plt.plot(f1, m1, label="original")
plt.plot(f2, m2, label="encoded", alpha=0.8)
plt.plot(f3, m3, label="decoded", alpha=0.8)
plt.xlim(0, fs / 2)
plt.title("Амплитудные спектры")
plt.xlabel("Hz")
plt.ylabel("dB")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
