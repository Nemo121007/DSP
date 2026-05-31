import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def load_ecg(path: str):
    """Загружает данные ЭКГ из текстового файла.

    Ожидается, что файл содержит минимум два столбца: время и значение сигнала.

    Args:
        path (str): Путь к файлу с данными.

    Returns:
        tuple: Кортеж, содержащий:
            - np.ndarray: Массив значений времени (t).
            - np.ndarray: Массив значений ЭКГ сигнала (x).
            - float: Оцененную частоту дискретизации (fs).

    Raises:
        ValueError: Если в файле менее двух столбцов данных.
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
    x: np.ndarray, fs: float, fmin: float = 15.0, fmax: float | None = None
):
    """Оценивает частоту узкополосной помехи в сигнале.

    Ищет максимум амплитудного спектра в заданном диапазоне частот.

    Args:
        x (np.ndarray): Входной сигнал.
        fs (float): Частота дискретизации.
        fmin (float, optional): Минимальная частота поиска в Гц. По умолчанию 15.0.
        fmax (float | None, optional): Максимальная частота поиска в Гц. Если None,
            ограничивается частотой Найквиста или 120 Гц.

    Returns:
        float: Оцененная частота помехи в Гц.

    Raises:
        RuntimeError: Если в заданном диапазоне нет данных.
    """
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


def remove_interference(
    x: np.ndarray, fs: float, f0: float, half_width_hz: float = 1.0
):
    """Удаляет узкополосную помеху из сигнала простым занулением спектральной полосы.

    Args:
        x (np.ndarray): Входной зашумленный сигнал.
        fs (float): Частота дискретизации.
        f0 (float): Частота удаляемой помехи в Гц.
        half_width_hz (float, optional): Полуширина вырезаемой полосы в Гц. По умолчанию 1.0.

    Returns:
        np.ndarray: Очищенный сигнал во временной области.
    """
    x = x - np.mean(x)
    n = len(x)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    df = fs / n
    half_width_bins = max(1, int(round(half_width_hz / df)))

    k0 = int(np.argmin(np.abs(freqs - f0)))
    k_left = max(1, k0 - half_width_bins)
    k_right = min(len(X) - 1, k0 + half_width_bins)

    # Вырезаем проблемный участок
    X[k_left : k_right + 1] = 0

    y = np.fft.irfft(X, n=n)
    return y


def spectrum_db(x: np.ndarray, fs: float):
    """Вычисляет амплитудный спектр сигнала в децибелах (дБ).

    Args:
        x (np.ndarray): Входной сигнал.
        fs (float): Частота дискретизации.

    Returns:
        tuple: Кортеж, содержащий:
            - np.ndarray: Массив частот (Гц).
            - np.ndarray: Значения амплитудного спектра в дБ.
    """
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.rfftfreq(n, d=1 / fs)
    X = np.abs(np.fft.rfft(x)) / n
    Xdb = 20 * np.log10(X + 1e-12)
    return f, Xdb


def plot_all(t, x_noisy, x_clean, fs, seconds_to_show=10.0):
    """Строит графики зашумленного и очищенного сигналов, а также их спектров.

    Args:
        t (np.ndarray): Массив значений времени.
        x_noisy (np.ndarray): Зашумленный сигнал.
        x_clean (np.ndarray): Очищенный сигнал.
        fs (float): Частота дискретизации.
        seconds_to_show (float, optional): Длительность отображаемого участка
            во временной области (в секундах). По умолчанию 10.0.
    """
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


# ----------------- основной запуск -----------------
path = "data/ecg.dat"

t, x, fs = load_ecg(path)
x = x - np.mean(x)

f0 = estimate_narrow_interference_freq(x, fs)
print(f"Оцененная частота помехи: {f0:.2f} Гц")

x_clean = remove_interference(
    x,
    fs,
    f0=f0,
    half_width_hz=0.5,
)

plot_all(t, x, x_clean, fs, seconds_to_show=10.0)
