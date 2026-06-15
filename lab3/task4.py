from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


# ============================================================
# Настройки
# ============================================================

# Если скрипт запускается из PyCharm, __file__ обычно доступен.
# Если нет (например, в интерактивной среде), берём текущую папку.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
INPUT_PATH = BASE_DIR / "data files" / "test.wav"
OUTPUT_DIR = BASE_DIR / "data files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODED_PATH = OUTPUT_DIR / "encoded.wav"
DECODED_PATH = OUTPUT_DIR / "decoded.wav"


# ============================================================
# Вспомогательные функции
# ============================================================

def to_float32_audio(x: np.ndarray) -> np.ndarray:
    """
    Переводит PCM-аудио в float32 в диапазоне [-1, 1].
    """
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float32)
    return x


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """
    Нормирует сигнал по максимуму, чтобы избежать клиппинга при сохранении.
    """
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return x.astype(np.float32)
    return (x / peak * 0.98).astype(np.float32)


def choose_carrier(fs: int) -> float:
    """
    Выбирает несущую частоту fa.

    Для речи 300–3000 Гц нужно:
        3000 < fa < fs/2 - 300

    Для fs = 11025 Гц корректно подходит fa = 3500 Гц.
    """
    nyq = fs / 2.0
    preferred = 3500.0

    if 3000.0 < preferred < nyq - 300.0:
        return preferred

    fa_min = 3000.0 + 1.0
    fa_max = nyq - 300.0 - 1.0
    if fa_max <= fa_min:
        raise ValueError(
            f"Частота дискретизации {fs} Гц слишком мала для кодирования речи 300–3000 Гц."
        )

    return 0.5 * (fa_min + fa_max)


def design_bandpass_remez(
    fs: int,
    f1: float = 300.0,
    f2: float = 3000.0,
    transition: float = 150.0,
    numtaps: int = 401,
) -> np.ndarray:
    """
    Полосовой КИХ-фильтр по Parks–McClellan.

    Полоса пропускания: [f1, f2]
    Полосы подавления: [0, f1-transition] и [f2+transition, fs/2]
    """
    nyq = fs / 2.0
    a = max(0.0, f1 - transition)
    b = min(nyq, f2 + transition)

    if a >= f1 or f2 >= b:
        raise ValueError("Некорректные границы полосы для bandpass-фильтра.")

    bands = [0.0, a, f1, f2, b, nyq]
    desired = [0, 1, 0]
    weight = [10, 1, 10]

    taps = signal.remez(numtaps, bands, desired, weight=weight, fs=fs)
    return taps.astype(np.float64)


def design_hilbert_remez(
    fs: int,
    numtaps: int = 201,
    guard_hz: float = 200.0,
) -> np.ndarray:
    """
    Широкополосный КИХ-преобразователь Гильберта по Parks–McClellan.

    Полоса аппроксимации: [guard_hz, fs/2 - guard_hz]
    """
    nyq = fs / 2.0
    lo = guard_hz
    hi = nyq - guard_hz

    if hi <= lo:
        raise ValueError("Слишком маленькая частота дискретизации для Hilbert-фильтра.")

    taps = signal.remez(
        numtaps,
        [lo, hi],
        [1],
        type="hilbert",
        fs=fs,
        maxiter=100,
        grid_density=32,
    )
    return taps.astype(np.float64)


def fir_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Каузальная FIR-фильтрация без компенсации задержки.
    """
    return signal.lfilter(h, [1.0], x)


def shift_left_with_zeros(x: np.ndarray, delay: int) -> np.ndarray:
    """
    Сдвиг сигнала влево на delay отсчётов с заполнением хвоста нулями.
    """
    if delay <= 0:
        return x
    if delay >= len(x):
        return np.zeros_like(x)
    return np.concatenate([x[delay:], np.zeros(delay, dtype=x.dtype)])


def total_delay(bp_taps: np.ndarray, h_taps: np.ndarray) -> int:
    """
    Суммарная групповая задержка тракта:
        bandpass -> Hilbert -> modulation -> bandpass
    """
    d_bp = (len(bp_taps) - 1) // 2
    d_h = (len(h_taps) - 1) // 2
    return 2 * d_bp + d_h


def invert_spectrum(
    x: np.ndarray,
    fs: int,
    fa: float,
    bp_taps: np.ndarray,
    h_taps: np.ndarray,
) -> np.ndarray:
    """
    Частотная инверсия вокруг fa.

    Формула:
        y[n] = x[n] * cos(2π fa n/fs) + H{x}[n] * sin(2π fa n/fs)

    Сначала:
        1) полосовая фильтрация речи,
        2) получение квадратурной компоненты через Hilbert,
        3) перенос спектра,
        4) финальная полосовая очистка.

    Затем компенсируется суммарная групповая задержка.
    """
    x_bp = fir_filter(x, bp_taps)
    x_h = fir_filter(x_bp, h_taps)

    n = np.arange(len(x), dtype=np.float64)
    c = np.cos(2.0 * np.pi * fa * n / fs)
    s = np.sin(2.0 * np.pi * fa * n / fs)

    # Если понадобится, здесь можно проверить альтернативный знак:
    # y = x_bp * c - x_h * s
    y = x_bp * c + x_h * s

    y = fir_filter(y, bp_taps)

    delay = total_delay(bp_taps, h_taps)
    y = shift_left_with_zeros(y, delay)

    return y.astype(np.float32)


def spectrum_db(x: np.ndarray, fs: int, nfft: int = 8192):
    """
    Амплитудный спектр в дБ.
    """
    X = np.fft.rfft(x, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, mag


def freq_response(h: np.ndarray, fs: int, nfft: int = 8192):
    """
    Частотная характеристика FIR-фильтра.
    """
    w, H = signal.freqz(h, worN=nfft, fs=fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.unwrap(np.angle(H))
    return w, mag, phase


# ============================================================
# Основной алгоритм
# ============================================================

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл: {INPUT_PATH}")

    fs, x = wavfile.read(INPUT_PATH)
    x = to_float32_audio(x)

    # Если файл стерео — переводим в моно
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Выбор несущей
    fa = choose_carrier(fs)

    # Параметры фильтров
    numtaps_bp = 401
    numtaps_h = 201

    bp_taps = design_bandpass_remez(
        fs=fs,
        f1=300.0,
        f2=3000.0,
        transition=150.0,
        numtaps=numtaps_bp,
    )
    h_taps = design_hilbert_remez(
        fs=fs,
        numtaps=numtaps_h,
        guard_hz=200.0,
    )

    # Кодирование и декодирование
    encoded = invert_spectrum(x, fs, fa, bp_taps, h_taps)
    decoded = invert_spectrum(encoded, fs, fa, bp_taps, h_taps)

    # Нормализация для сохранения
    encoded_out = normalize_audio(encoded)
    decoded_out = normalize_audio(decoded)

    # Сохранение WAV
    wavfile.write(ENCODED_PATH, fs, encoded_out.astype(np.float32))
    wavfile.write(DECODED_PATH, fs, decoded_out.astype(np.float32))

    # Оценка качества восстановления
    min_len = min(len(x), len(decoded_out))
    x0 = x[:min_len]
    d0 = decoded_out[:min_len]

    mse = np.mean((x0 - d0) ** 2)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(x0, d0)[0, 1]

    print(f"fs = {fs} Hz")
    print(f"carrier fa = {fa:.2f} Hz")
    print(f"bandpass taps = {len(bp_taps)}")
    print(f"hilbert taps = {len(h_taps)}")
    print(f"encoded max |x| = {np.max(np.abs(encoded)):.6f}")
    print(f"decoded max |x| = {np.max(np.abs(decoded)):.6f}")
    print(f"RMSE = {rmse:.6e}")
    print(f"Pearson corr = {corr:.6f}")
    print(f"Saved: {ENCODED_PATH}")
    print(f"Saved: {DECODED_PATH}")

    # ========================================================
    # Графики
    # ========================================================

    # 1) АЧХ полосового фильтра и Hilbert-фильтра
    fb, mb, pb = freq_response(bp_taps, fs)
    fh, mh, ph = freq_response(h_taps, fs)

    plt.figure(figsize=(13, 8))

    plt.subplot(2, 1, 1)
    plt.plot(fb, mb)
    plt.title("Частотная характеристика полосового КИХ-фильтра")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Magnitude, dB")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(fh, mh)
    plt.title("Частотная характеристика преобразователя Гильберта")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Magnitude, dB")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 2) Фазовая характеристика фильтров
    plt.figure(figsize=(13, 8))

    plt.subplot(2, 1, 1)
    plt.plot(fb, pb)
    plt.title("Фазовая характеристика полосового КИХ-фильтра")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Phase, rad")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(fh, ph)
    plt.title("Фазовая характеристика преобразователя Гильберта")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Phase, rad")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 3) Сравнение сигналов во времени
    t = np.arange(min_len) / fs
    show_n = min(min_len, fs // 2)  # первые 0.5 секунды

    plt.figure(figsize=(13, 7))
    plt.plot(t[:show_n], x0[:show_n], label="original")
    plt.plot(t[:show_n], encoded_out[:show_n], label="encoded", alpha=0.8)
    plt.plot(t[:show_n], d0[:show_n], label="decoded", alpha=0.8)
    plt.title("Сигналы во временной области")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) Сравнение спектров
    f1, m1 = spectrum_db(x0, fs)
    f2, m2 = spectrum_db(encoded_out[:min_len], fs)
    f3, m3 = spectrum_db(d0, fs)

    plt.figure(figsize=(13, 7))
    plt.plot(f1, m1, label="original")
    plt.plot(f2, m2, label="encoded", alpha=0.8)
    plt.plot(f3, m3, label="decoded", alpha=0.8)
    plt.xlim(0, min(20000, fs / 2))
    plt.title("Амплитудные спектры")
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Magnitude, dB")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()