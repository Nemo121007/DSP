import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# =========================
# Загрузка и приведение к float
# =========================
fs, x = wavfile.read("data files/tune.wav")

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
# Поиск узкополосной помехи
# =========================
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), d=1 / fs)
amp = np.abs(X)

# Ищем выраженный пик выше 1 кГц,
# чтобы не путать с полезной музыкальной структурой в НЧ-области.
mask = freqs > 1000
freqs_hf = freqs[mask]
amp_hf = amp[mask]

peaks, props = signal.find_peaks(
    amp_hf,
    prominence=np.max(amp_hf) * 0.05
)

if len(peaks) == 0:
    raise RuntimeError("Не найден выраженный узкополосный пик. Попробуйте другой критерий поиска.")

best_peak = peaks[np.argmax(amp_hf[peaks])]
f0 = freqs_hf[best_peak]

print(f"Обнаруженная аномальная частота: {f0:.2f} Hz")

# =========================
# Проектирование линейного КИХ band-stop фильтра
# =========================
# Ширина режекторной зоны вокруг f0.
# Для узкополосной тональной помехи обычно достаточно узкой полосы.
# При необходимости увеличьте bw_hz.
bw_hz = 100.0

f1 = max(0.0, f0 - bw_hz)
f2 = min(fs / 2 - 1.0, f0 + bw_hz)

# Длина FIR: чем уже режекция, тем длиннее фильтр.
# Должна быть нечётной для симметричной линейной фазы (тип I).
numtaps = 1001

# Линейно-фазовый КИХ band-stop.
# firwin с bandstop строит симметричный FIR.
h = signal.firwin(
    numtaps=numtaps,
    cutoff=[f1, f2],
    fs=fs,
    pass_zero="bandstop",
    window="hamming"
)

# Проверка симметрии
sym_err = np.max(np.abs(h - h[::-1]))
print(f"Макс. нарушение симметрии ИХ: {sym_err:.3e}")

# =========================
# Фильтрация
# =========================
# causal filtering
y = signal.lfilter(h, [1.0], x)

# Групповая задержка линейно-фазового FIR:
delay = (numtaps - 1) // 2

# Компенсация задержки
y = np.roll(y, -delay)
y[-delay:] = 0.0

# =========================
# Сохранение результата
# =========================
y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
wavfile.write("tune_filtered_fir.wav", fs, y_int16)

print("Сохранено: tune_filtered_fir.wav")


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
plot_spectrum(y, fs, "Спектр после FIR-фильтрации")

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
