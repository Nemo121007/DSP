import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# =========================
# Загрузка ЭКГ из txt
# =========================
path = "data files/ecg.dat"
data = np.loadtxt(path)

if data.ndim != 2 or data.shape[1] < 2:
    raise ValueError("Файл должен содержать минимум два столбца: время и сигнал.")

t = data[:, 0].astype(float)
x = data[:, 1].astype(float)

dt = np.median(np.diff(t))
fs = 1.0 / dt

x = x - np.mean(x)

print(f"fs = {fs:.2f} Hz, duration = {len(x) / fs:.2f} s")

# =========================
# Короткий фрагмент сигнала
# =========================
dur = 10.0
n_show = min(len(x), int(dur * fs))
tt = t[:n_show]

plt.figure(figsize=(14, 4))
plt.plot(tt, x[:n_show])
plt.title("ЭКГ: первые 10 с")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Спектр до фильтрации
# =========================
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), d=1 / fs)
amp = np.abs(X)

plt.figure(figsize=(14, 4))
plt.plot(freqs, 20 * np.log10(amp + 1e-12))
plt.title("Спектр ЭКГ до фильтрации")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда, дБ")
plt.grid(True)
plt.xlim(0, min(120, fs / 2))
plt.tight_layout()
plt.show()

# =========================
# Поиск узкополосной помехи
# =========================
fmin = 15.0
fmax = min(120.0, fs / 2 - 1.0)

mask = (freqs >= fmin) & (freqs <= fmax)
freqs_sel = freqs[mask]
amp_sel = amp[mask]

if len(amp_sel) == 0:
    raise RuntimeError("В заданном диапазоне частот нет данных для поиска помехи.")

peaks, props = signal.find_peaks(
    amp_sel,
    prominence=np.max(amp_sel) * 0.05
)

if len(peaks) == 0:
    raise RuntimeError("Не найден выраженный узкополосный пик. Попробуйте другой критерий поиска.")

best_peak = peaks[np.argmax(amp_sel[peaks])]
f0 = float(freqs_sel[best_peak])

print(f"Обнаруженная частота помехи: {f0:.2f} Hz")

# =========================
# Проектирование FIR band-stop фильтра
# =========================
bw_hz = 1.75
f1 = max(0.0, f0 - bw_hz)
f2 = min(fs / 2 - 1.0, f0 + bw_hz)

if f2 <= f1:
    raise ValueError("Некорректная режекторная полоса.")

numtaps = 1001
if numtaps % 2 == 0:
    numtaps += 1

h = signal.firwin(
    numtaps=numtaps,
    cutoff=[f1, f2],
    fs=fs,
    pass_zero="bandstop",
    window="hamming"
)

sym_err = np.max(np.abs(h - h[::-1]))
print(f"КИХ-режекция: [{f1:.2f}, {f2:.2f}] Hz")
print(f"Длина FIR: {len(h)}")
print(f"Макс. нарушение симметрии ИХ: {sym_err:.3e}")

# =========================
# Фильтрация
# =========================
# causal filtering
y = signal.lfilter(h, [1.0], x)

# Компенсация групповой задержки линейно-фазового FIR
delay = (numtaps - 1) // 2
y = np.roll(y, -delay)
y[-delay:] = 0.0


# =========================
# Визуализация:
# =========================
def spectrum_db(sig: np.ndarray, fs: float):
    sig = sig - np.mean(sig)
    n = len(sig)
    f = np.fft.rfftfreq(n, d=1 / fs)
    X = np.abs(np.fft.rfft(sig)) / n
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


plot_all(t, x, y, fs, seconds_to_show=10.0)

# =========================
# АЧХ фильтра
# =========================
w, H = signal.freqz(h, worN=4096, fs=fs)

plt.figure(figsize=(14, 4))
plt.plot(w, 20 * np.log10(np.abs(H) + 1e-12))
plt.title("АЧХ линейного КИХ band-stop фильтра")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда, дБ")
plt.grid(True)
plt.tight_layout()
plt.show()
