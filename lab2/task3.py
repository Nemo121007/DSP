import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

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
plt.title("Waveform: first 0.1 s")
plt.xlabel("Time, s")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Спектрограмма
# =========================
plt.figure(figsize=(14, 5))
plt.specgram(x, NFFT=2048, Fs=fs, noverlap=1536)
plt.title("Spectrogram")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()

# =========================
# Поиск узкополосной помехи по обычному FFT
# =========================
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), d=1 / fs)
amp = np.abs(X)

# Ищем самый сильный пик выше 1 кГц
mask = freqs > 1000
freqs_hf = freqs[mask]
amp_hf = amp[mask]

peaks, props = signal.find_peaks(amp_hf, prominence=np.max(amp_hf) * 0.05)
if len(peaks) == 0:
    raise RuntimeError("Не найден выраженный узкополосный пик.")

best_peak = peaks[np.argmax(amp_hf[peaks])]
f0 = freqs_hf[best_peak]

print(f"Detected interference frequency: {f0:.2f} Hz")

# =========================
# Вырезка и линейная аппроксимация в спектре
# =========================
k0 = np.argmin(np.abs(freqs - f0))

# Полуширина вырезаемого участка
half_width_bins = 100

k_left = max(1, k0 - half_width_bins)
k_right = min(len(X) - 2, k0 + half_width_bins)

left_val = X[k_left - 1]
right_val = X[k_right + 1]

# Линейная интерполяция комплексного спектра
for k in range(k_left, k_right + 1):
    alpha = (k - k_left + 1) / (k_right - k_left + 2)
    X[k] = (1 - alpha) * left_val + alpha * right_val

# Обратное преобразование Фурье
y = np.fft.irfft(X, n=len(x))

# =========================
# Сохранение результата
# =========================
y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
wavfile.write("tune_filtered.wav", fs, y_int16)

print("Saved: tune_filtered.wav")


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
    plt.xlabel("Frequency, Hz")
    plt.ylabel("Magnitude, dB")
    plt.grid(True)
    plt.xlim(0, fs / 2)
    plt.tight_layout()
    plt.show()


plot_spectrum(x, fs, "Spectrum before filtering")
plot_spectrum(y, fs, "Spectrum after filtering")