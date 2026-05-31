from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

# =========================
# Helpers
# =========================


def to_float32_audio(x: np.ndarray) -> np.ndarray:
    """Конвертирует PCM-аудио в формат float32 в диапазоне [-1, 1].

    Args:
        x (np.ndarray): Входной аудиосигнал.

    Returns:
        np.ndarray: Нормализованный аудиосигнал в формате float32.
    """
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float32) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float32)
    return x


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Нормализует аудиосигнал для предотвращения клиппинга.

    Args:
        x (np.ndarray): Входной аудиосигнал.

    Returns:
        np.ndarray: Нормализованный аудиосигнал с максимальной амплитудой 0.98.
    """
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        return x
    return x / peak * 0.98


def design_bandpass_remez(
    fs: int,
    f1: float = 300.0,
    f2: float = 3000.0,
    transition: float = 150.0,
    numtaps: int = 401,
) -> np.ndarray:
    """Проектирует полосовой КИХ-фильтр методом Паркса-Макклеллана.

    Полоса пропускания: [f1, f2].
    Полосы подавления: [0, f1 - transition] и [f2 + transition, fs/2].

    Args:
        fs (int): Частота дискретизации в Гц.
        f1 (float, optional): Нижняя граница полосы пропускания в Гц. По умолчанию 300.0.
        f2 (float, optional): Верхняя граница полосы пропускания в Гц. По умолчанию 3000.0.
        transition (float, optional): Ширина переходной полосы в Гц. По умолчанию 150.0.
        numtaps (int, optional): Порядок фильтра (число коэффициентов). По умолчанию 401.

    Returns:
        np.ndarray: Коэффициенты спроектированного КИХ-фильтра.

    Raises:
        ValueError: Если заданы некорректные границы полос.
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


def design_hilbert_remez(
    fs: int, numtaps: int = 201, guard_hz: float = 200.0
) -> np.ndarray:
    """Проектирует широкополосный КИХ-преобразователь Гильберта методом Паркса-Макклеллана.

    Полоса аппроксимации: [guard_hz, fs/2 - guard_hz].

    Args:
        fs (int): Частота дискретизации в Гц.
        numtaps (int, optional): Порядок фильтра. По умолчанию 201.
        guard_hz (float, optional): Защитный интервал по краям диапазона (в Гц). По умолчанию 200.0.

    Returns:
        np.ndarray: Коэффициенты преобразователя Гильберта.

    Raises:
        ValueError: Если частота дискретизации слишком мала для создания фильтра.
    """
    nyq = fs / 2.0
    lo = guard_hz
    hi = nyq - guard_hz

    if hi <= lo:
        raise ValueError("Слишком маленькая частота дискретизации для Hilbert-фильтра.")

    taps = signal.remez(
        numtaps, [lo, hi], [1], type="hilbert", fs=fs, maxiter=100, grid_density=32
    )
    return taps.astype(np.float64)


def causal_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Каузальная КИХ-фильтрация с компенсацией задержки линейной фазы.

    Фильтр является каузальным, но выходной сигнал сдвигается для удаления
    групповой задержки, а в конец добавляются нули.

    Args:
        x (np.ndarray): Входной сигнал.
        h (np.ndarray): Коэффициенты каузального КИХ-фильтра.

    Returns:
        np.ndarray: Отфильтрованный сигнал с компенсированной задержкой.
    """
    y = signal.lfilter(h, [1.0], x)
    delay = (len(h) - 1) // 2
    if delay > 0:
        y = np.concatenate([y[delay:], np.zeros(delay, dtype=y.dtype)])
    return y


def invert_spectrum(
    x: np.ndarray, fs: int, fa: float, bp_taps: np.ndarray, h_taps: np.ndarray
) -> np.ndarray:
    """Выполняет инверсию спектра вокруг несущей частоты fa.

    Уравнение преобразования:
    y[n] = x[n] * cos(2π fa n/fs) + H{x}[n] * sin(2π fa n/fs)

    Args:
        x (np.ndarray): Входной аудиосигнал.
        fs (int): Частота дискретизации в Гц.
        fa (float): Частота сдвига (несущая) в Гц.
        bp_taps (np.ndarray): Коэффициенты полосового фильтра.
        h_taps (np.ndarray): Коэффициенты преобразователя Гильберта.

    Returns:
        np.ndarray: Сигнал с инвертированным спектром.
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
    """Вычисляет амплитудный спектр сигнала в децибелах (дБ).

    Args:
        x (np.ndarray): Входной сигнал.
        fs (int): Частота дискретизации в Гц.
        nfft (int, optional): Размер БПФ (FFT). По умолчанию 8192.

    Returns:
        tuple[np.ndarray, np.ndarray]: Кортеж, содержащий массив частот и массив амплитуд в дБ.
    """
    X = np.fft.rfft(x, n=nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, mag


# =========================
# Main pipeline
# =========================


def main():
    """Главная функция для выполнения полного цикла кодирования и декодирования.

    Считывает исходный аудиофайл, проектирует фильтры, кодирует сигнал путем
    частотной инверсии, затем декодирует обратно. Сохраняет результаты и строит графики для сравнения.

    Raises:
        FileNotFoundError: Если не найден входной файл test.wav.
        ValueError: Если частота дискретизации слишком мала.
    """
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
    numtaps_h = 201

    bp_taps = design_bandpass_remez(fs, 300.0, 3000.0, transition=150.0, numtaps=401)
    h_taps = design_hilbert_remez(fs, numtaps=numtaps_h, guard_hz=200.0)

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
