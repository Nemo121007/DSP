import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def load_ecg(path: str):
    """Загружает данные ЭКГ из текстового файла.

    Ожидается, что файл содержит минимум два столбца: время и значение сигнала.
    """
    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Файл должен содержать минимум два столбца: время и сигнал.")

    t = data[:, 0].astype(float)
    x = data[:, 1].astype(float)

    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    return t, x, fs


def estimate_narrow_interference_freq(
    x: np.ndarray,
    fs: float,
    fmin: float = 15.0,
    fmax: float | None = None
):
    """Оценивает частоту узкополосной помехи по максимуму спектра."""
    x = x - np.mean(x)

    if fmax is None:
        fmax = min(120.0, fs / 2 - 1.0)

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amp = np.abs(np.fft.rfft(x))

    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[mask]
    amp_sel = amp[mask]

    if len(amp_sel) == 0:
        raise RuntimeError("В заданном диапазоне частот нет данных для поиска помехи.")

    k = np.argmax(amp_sel)
    return float(freqs_sel[k])


def design_fir_notch(fs: float, f0: float, half_width_hz: float = 0.5, numtaps: int = 1001):
    """Проектирует линейный КИХ band-stop фильтр вокруг частоты f0.

    Args:
        fs: Частота дискретизации.
        f0: Частота помехи.
        half_width_hz: Полуширина режекторной полосы.
        numtaps: Число коэффициентов FIR. Должно быть нечётным для симметричного типа I.
    """
    if numtaps % 2 == 0:
        numtaps += 1

    f1 = max(0.0, f0 - half_width_hz)
    f2 = min(fs / 2 - 1e-6, f0 + half_width_hz)

    if f2 <= f1:
        raise ValueError("Некорректная режекторная полоса.")

    h = signal.firwin(
        numtaps=numtaps,
        cutoff=[f1, f2],
        fs=fs,
        pass_zero="bandstop",
        window="hamming"
    )
    return h, (f1, f2)


def remove_interference_fir(
    x: np.ndarray,
    fs: float,
    f0: float,
    half_width_hz: float = 0.5,
    numtaps: int = 1001
):
    """Удаляет узкополосную помеху линейным КИХ-фильтром."""
    x = x - np.mean(x)

    h, (f1, f2) = design_fir_notch(
        fs=fs,
        f0=f0,
        half_width_hz=half_width_hz,
        numtaps=numtaps
    )

    # Zero-phase filtering: форма ЭКГ не сдвигается по времени.
    y = signal.filtfilt(h, [1.0], x)

    return y, h, (f1, f2)


def spectrum_db(x: np.ndarray, fs: float):
    """Амплитудный спектр в дБ."""
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.rfftfreq(n, d=1 / fs)
    X = np.abs(np.fft.rfft(x)) / n
    Xdb = 20 * np.log10(X + 1e-12)
    return f, Xdb


def plot_filter_response(h: np.ndarray, fs: float):
    """АЧХ FIR-фильтра."""
    w, H = signal.freqz(h, worN=4096, fs=fs)

    plt.figure(figsize=(14, 4))
    plt.plot(w, 20 * np.log10(np.abs(H) + 1e-12))
    plt.title("АЧХ линейного КИХ band-stop фильтра")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда, дБ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all(t, x_noisy, x_clean, fs, seconds_to_show=10.0):
    """Сравнение сигнала до/после фильтрации и их спектров."""
    n_show = min(len(x_noisy), int(seconds_to_show * fs))
    tt = t[:n_show]

    f1, s1 = spectrum_db(x_noisy, fs)
    f2, s2 = spectrum_db(x_clean, fs)

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))

    ax[0, 0].plot(tt, x_noisy[:n_show])
    ax[0, 0].set_title("Зашумленная ЭКГ")
    ax[0, 0].set_xlabel("Время, с")
    ax[0, 0].set_ylabel("Амплитуда")
    ax[0, 0].grid(True)

    ax[0, 1].plot(tt, x_clean[:n_show])
    ax[0, 1].set_title("Очищенная ЭКГ")
    ax[0, 1].set_xlabel("Время, с")
    ax[0, 1].set_ylabel("Амплитуда")
    ax[0, 1].grid(True)

    ax[1, 0].plot(f1, s1)
    ax[1, 0].set_title("Спектр зашумленной ЭКГ (дБ)")
    ax[1, 0].set_xlabel("Частота, Гц")
    ax[1, 0].set_ylabel("Уровень, дБ")
    ax[1, 0].set_xlim(0, min(120, fs / 2))
    ax[1, 0].grid(True)

    ax[1, 1].plot(f2, s2)
    ax[1, 1].set_title("Спектр очищенной ЭКГ (дБ)")
    ax[1, 1].set_xlabel("Частота, Гц")
    ax[1, 1].set_ylabel("Уровень, дБ")
    ax[1, 1].set_xlim(0, min(120, fs / 2))
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


path = "data files/ecg.dat"

t, x, fs = load_ecg(path)
x = x - np.mean(x)

f0 = estimate_narrow_interference_freq(x, fs)
print(f"Оцененная частота помехи: {f0:.2f} Гц")

x_clean, h, (f1, f2) = remove_interference_fir(
    x,
    fs,
    f0=f0,
    half_width_hz=1.5,
    numtaps=1001
)

print(f"КИХ-режекция: [{f1:.2f}, {f2:.2f}] Гц")
print(f"Длина FIR: {len(h)}")

plot_filter_response(h, fs)
plot_all(t, x, x_clean, fs, seconds_to_show=10.0)
