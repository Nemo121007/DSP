import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def load_ecg(path: str, fs: float | None = None):
    data = np.loadtxt(path)

    # Один столбец: только сигнал
    if data.ndim == 1:
        x = data.astype(float)
        if fs is None:
            raise ValueError("Для одноколоночного файла нужно явно задать fs.")
        t = np.arange(len(x)) / fs
        return t, x, fs

    # Два и более столбца
    if data.shape[1] >= 2:
        c0 = data[:, 0].astype(float)
        c1 = data[:, 1].astype(float)

        # Если первый столбец монотонно растет, считаем его временем
        if np.all(np.diff(c0) > 0):
            t = c0
            if fs is None:
                dt = np.median(np.diff(t))
                fs = 1.0 / dt
            return t, c1, fs

        # Иначе считаем, что первый столбец — сам сигнал
        x = c0
        if fs is None:
            raise ValueError("Файл не содержит временную ось. Нужно задать fs.")
        t = np.arange(len(x)) / fs
        return t, x, fs

    raise ValueError("Не удалось интерпретировать формат файла.")


def estimate_narrow_interference_freq(x: np.ndarray, fs: float,
                                      fmin: float = 15.0,
                                      fmax: float | None = None):
    """
    Ищет узкий спектральный пик, который похож на помеху.
    Используется Welch + вычитание сглаженного фона.
    """
    x = x - np.mean(x)
    if fmax is None:
        fmax = min(120.0, fs / 2 - 1.0)

    nperseg = min(len(x), 8192)
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")

    mask = (f >= fmin) & (f <= fmax)
    f = f[mask]
    pxx = pxx[mask]

    logp = np.log10(pxx + 1e-18)

    # Сглаженный "фон" спектра
    k = 31 if len(logp) >= 31 else max(3, (len(logp) // 2) * 2 + 1)
    baseline = signal.medfilt(logp, kernel_size=k)

    # Остаток — кандидаты на узкие линии
    sharp = logp - baseline
    peaks, props = signal.find_peaks(sharp, prominence=np.percentile(sharp, 90) * 0.15)

    if len(peaks) == 0:
        return float(f[np.argmax(pxx)])

    best = peaks[np.argmax(props["prominences"])]
    return float(f[best])


def notch_harmonics(x: np.ndarray, fs: float, f0: float,
                    q: float = 50.0, max_harmonics: int = 5):
    """
    Последовательно подавляет f0, 2f0, 3f0, ...
    Q больше -> уже полоса подавления.
    """
    y = x.astype(float).copy()
    for k in range(1, max_harmonics + 1):
        fk = f0 * k
        if fk >= fs / 2 - 1:
            break
        b, a = signal.iirnotch(f0=fk, Q=q, fs=fs)
        y = signal.filtfilt(b, a, y)
    return y


def spectrum_db(x: np.ndarray, fs: float):
    """
    Односторонний спектр в дБ, чтобы тонкая линия была видна лучше.
    """
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.rfftfreq(n, d=1 / fs)
    X = np.abs(np.fft.rfft(x)) / n
    Xdb = 20 * np.log10(X + 1e-12)
    return f, Xdb


def plot_all(t, x_noisy, x_clean, fs, seconds_to_show=10.0):
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

# Если в файле только один столбец, задайте частоту дискретизации вручную.
# Подберите под ваш файл, если знаете точное значение.
fs_manual = 500.0

t, x, fs = load_ecg(path, fs=fs_manual)
x = x - np.mean(x)

# Автоматически находим узкую помеху
f0 = estimate_narrow_interference_freq(x, fs)
print(f"Оцененная частота помехи: {f0:.2f} Гц")

# Фильтрация
x_clean = notch_harmonics(x, fs, f0=f0, q=60.0, max_harmonics=5)

# Сравнение
plot_all(t, x, x_clean, fs, seconds_to_show=10.0)