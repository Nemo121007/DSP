import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

# =========================
# Параметры
# =========================
cut_half_width_hz = 0.5  # половина ширины вырезаемой полосы, Гц
search_from_hz = 1000.0  # с какой частоты искать помеху
prominence_ratio = 0.05  # чувствительность поиска пика

# =========================
# Загрузка и приведение к float
# =========================
fs, x = wavfile.read("data/tune.wav")

if x.dtype == np.int16:
    x = x.astype(np.float64) / 32768.0
elif x.dtype == np.int32:
    x = x.astype(np.float64) / 2147483648.0
elif x.dtype == np.uint8:
    x = (x.astype(np.float64) - 128) / 128.0
else:
    x = x.astype(np.float64)

if x.ndim == 2:
    x = x.mean(axis=1)

print(f"fs = {fs} Hz, duration = {len(x)/fs:.2f} s")

# =========================
# Короткий фрагмент waveform
# =========================
dur = 0.1
n_show = int(dur * fs)
t = np.arange(n_show) / fs

plt.figure(figsize=(14, 4))
plt.plot(t, x[:n_show])
plt.title("Сигнал: первые 0.1 с")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Спектрограмма
# =========================
plt.figure(figsize=(14, 5))
plt.specgram(x, NFFT=2048, Fs=fs, noverlap=1536)
plt.title("Спектрограмма")
plt.xlabel("Время, с")
plt.ylabel("Частота, Гц")
plt.colorbar(label="дБ")
plt.tight_layout()
plt.show()

# =========================
# Поиск узкополосной помехи по FFT
# =========================
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), d=1 / fs)
amp = np.abs(X)

mask = freqs > search_from_hz
freqs_hf = freqs[mask]
amp_hf = amp[mask]

peaks, props = signal.find_peaks(amp_hf, prominence=np.max(amp_hf) * prominence_ratio)
if len(peaks) == 0:
    raise RuntimeError("Не найден выраженный узкополосный пик.")

best_peak = peaks[np.argmax(amp_hf[peaks])]
f0 = freqs_hf[best_peak]

print(f"Обнаружена аномальная частота: {f0:.2f} Hz")

# =========================
# Простое вырезание полосы вокруг f0
# =========================
f_left = f0 - cut_half_width_hz
f_right = f0 + cut_half_width_hz

k_left = np.searchsorted(freqs, f_left, side="left")
k_right = np.searchsorted(freqs, f_right, side="right") - 1

k_left = max(1, k_left)
k_right = min(len(X) - 2, k_right)

print(f"Вырезаем полосу: [{freqs[k_left]:.2f}, {freqs[k_right]:.2f}] Hz")

X[k_left : k_right + 1] = 0.0

# Обратное преобразование Фурье
y = np.fft.irfft(X, n=len(x))

# =========================
# Сохранение результата
# =========================
max_abs = np.max(np.abs(y))
if max_abs > 0:
    y = 0.98 * y / max_abs

y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
wavfile.write("tune_filtered.wav", fs, y_int16)

print("Сохранено: tune_filtered.wav")


# =========================
# Визуализация спектра до/после
# =========================
def plot_spectrum(sig, fs, title):
    X = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
    amp = np.abs(X)

    plt.figure(figsize=(14, 4))
    plt.plot(freqs, 20 * np.log10(amp + 1e-12))
    plt.title(title)
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда, дБ")
    plt.grid(True)
    plt.xlim(0, fs / 2)
    plt.tight_layout()
    plt.show()


plot_spectrum(x, fs, "Спектр до фильтрации")
plot_spectrum(y, fs, "Спектр после фильтрации")
