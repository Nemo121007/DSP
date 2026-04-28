import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import iirnotch, filtfilt

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

# Короткий фрагмент waveform
dur = 0.1  # секунд
n_show = int(dur * fs)
t = np.arange(n_show) / fs

plt.figure(figsize=(14, 4))
plt.plot(t, x[:n_show])
plt.title("Waveform: first 0.1 s")
plt.xlabel("Time, s")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Welch PSD
f, pxx = signal.welch(x, fs=fs, window='hann', nperseg=8192, noverlap=4096)
plt.figure(figsize=(14, 4))
plt.plot(f, 10 * np.log10(pxx + 1e-18))
plt.title("Welch PSD")
plt.xlabel("Frequency, Hz")
plt.ylabel("Power, dB")
plt.grid(True)
plt.xlim(0, fs / 2)
plt.tight_layout()
plt.show()

# Спектрограмма
plt.figure(figsize=(14, 5))
plt.specgram(x, NFFT=2048, Fs=fs, noverlap=1536)
plt.title("Spectrogram")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()

fs, x = wavfile.read("data/tune.wav")

# Приведение к float
if x.dtype == np.int16:
    x = x.astype(np.float64) / 32768.0
elif x.dtype == np.int32:
    x = x.astype(np.float64) / 2147483648.0
elif x.dtype == np.uint8:
    x = (x.astype(np.float64) - 128) / 128.0
else:
    x = x.astype(np.float64)

# Если стерео — переводим в моно
if x.ndim == 2:
    x = x.mean(axis=1)

print(f"fs = {fs} Hz, duration = {len(x)/fs:.2f} s")

# =========================
# Поиск пика помехи по Welch PSD
# =========================
f, pxx = signal.welch(x, fs=fs, window="hann", nperseg=8192, noverlap=4096)
pxx_db = 10 * np.log10(pxx + 1e-18)

# Ищем самый сильный пик выше 1 кГц, чтобы не ловить музыкальный фундамент
mask = f > 1000
f2 = f[mask]
p2 = pxx_db[mask]

peaks, props = signal.find_peaks(p2, prominence=10)
if len(peaks) == 0:
    raise RuntimeError("Не найден выраженный узкополосный пик.")

# Самый мощный пик
best_peak = peaks[np.argmax(p2[peaks])]
f0 = f2[best_peak]

print(f"Detected interference frequency: {f0:.2f} Hz")

# =========================
# Notch-фильтр
# =========================
def apply_notch(sig, fs, f0, q=60.0):
    w0 = f0 / (fs / 2)
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig)

# Узкий фильтр вокруг 15 кГц
y = apply_notch(x, fs, f0, q=80.0)

# При необходимости можно подавить ещё и соседние частоты,
# если линия немного "расплывается":
# y = apply_notch(y, fs, f0 - 20, q=80.0)
# y = apply_notch(y, fs, f0 + 20, q=80.0)

# Нормировка
max_abs = np.max(np.abs(y))
if max_abs > 0:
    y = 0.98 * y / max_abs

# =========================
# Сохранение результата
# =========================
y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
wavfile.write("tune_filtered.wav", fs, y_int16)

print("Saved: tune_filtered.wav")

# =========================
# Визуализация до/после
# =========================
def plot_psd(sig, fs, title):
    f, pxx = signal.welch(sig, fs=fs, window="hann", nperseg=8192, noverlap=4096)
    plt.figure(figsize=(14, 4))
    plt.plot(f, 10 * np.log10(pxx + 1e-18))
    plt.title(title)
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Power, dB")
    plt.grid(True)
    plt.xlim(0, fs / 2)
    plt.tight_layout()
    plt.show()

plot_psd(x, fs, "PSD before filtering")
plot_psd(y, fs, "PSD after filtering")