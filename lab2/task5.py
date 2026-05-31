import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def inverse_quarter_permutation_spectrum(x: np.ndarray) -> np.ndarray:
    """Восстанавливает сигнал после перестановки спектральных четвертей.

    Для восстановления делает обратную перестановку:
    [C B D A] -> [A B C D] = [q4, q2, q1, q3]

    Args:
        x (np.ndarray): Искаженный сигнал во временной области.

    Returns:
        np.ndarray: Восстановленный сигнал во временной области.

    Raises:
        ValueError: Если длина сигнала не кратна 4.
    """
    n = len(x)

    # FFT исходного искажённого сигнала
    X = np.fft.fft(x)

    # Разбиваем спектр на 4 части.
    # Предполагается, что длина файла кратна 4.
    n4 = n // 4
    if n % 4 != 0:
        raise ValueError(
            f"Длина сигнала {n} не кратна 4. "
            f"Для точной поквартальной перестановки нужна длина, кратная 4."
        )

    q1 = X[:n4]
    q2 = X[n4 : 2 * n4]
    q3 = X[2 * n4 : 3 * n4]
    q4 = X[3 * n4 :]

    # Обратная перестановка к [C B D A]
    X_restored = np.concatenate([q4, q2, q1, q3])

    # Возврат во временную область
    x_restored = np.fft.ifft(X_restored).real

    return x_restored


def save_wav_like_original(path_in: str, path_out: str):
    """Считывает WAV файл, восстанавливает его спектр и сохраняет в новый файл.

    Поддерживает обработку как mono, так и stereo файлов. Сохраняет
    исходный тип данных аудио.

    Args:
        path_in (str): Путь к исходному (искаженному) WAV файлу.
        path_out (str): Путь для сохранения восстановленного WAV файла.

    Returns:
        tuple: Кортеж, содержащий:
            - int: Частоту дискретизации.
            - np.ndarray: Исходные данные.
            - np.ndarray: Восстановленные данные.
    """
    fs, data = wavfile.read(path_in)

    # Обрабатываем и mono, и stereo
    if data.ndim == 1:
        x = data.astype(np.float64)
        xr = inverse_quarter_permutation_spectrum(x)

        # Возвращаем в исходный целочисленный формат
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            xr = np.clip(np.rint(xr), info.min, info.max).astype(data.dtype)
        else:
            xr = xr.astype(data.dtype)

    else:
        channels = []
        for ch in range(data.shape[1]):
            x = data[:, ch].astype(np.float64)
            xr_ch = inverse_quarter_permutation_spectrum(x)
            channels.append(xr_ch)

        xr = np.stack(channels, axis=1)

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            xr = np.clip(np.rint(xr), info.min, info.max).astype(data.dtype)
        else:
            xr = xr.astype(data.dtype)

    wavfile.write(path_out, fs, xr)
    return fs, data, xr


def plot_spectra(original_data, restored_data, fs):
    """Визуализирует амплитудный спектр до и после восстановления."""
    if original_data.ndim > 1:
        # Для стерео берем только первый канал для визуализации
        orig_channel = original_data[:, 0]
        rest_channel = restored_data[:, 0]
    else:
        orig_channel = original_data
        rest_channel = restored_data

    N = len(orig_channel)

    # Вычисляем спектры
    X_orig = np.fft.fft(orig_channel)
    X_rest = np.fft.fft(rest_channel)

    # Вычисляем частоты и делаем сдвиг для отображения [-fs/2, fs/2]
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    mag_orig = np.abs(np.fft.fftshift(X_orig))
    mag_rest = np.abs(np.fft.fftshift(X_rest))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, mag_orig, color="red")
    plt.title("Спектр амплитуд ДО восстановления")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs, mag_rest, color="green")
    plt.title("Спектр амплитуд ПОСЛЕ восстановления")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---- запуск ----
input_file = "data/test5.wav"
output_file = "test5_restored.wav"

fs, original_data, restored_data = save_wav_like_original(input_file, output_file)

print(f"Готово. Восстановленный файл сохранён как: {output_file}")
print(f"Частота дискретизации: {fs} Гц")
print(f"Длина сигнала: {len(original_data)} отсчётов")

# Визуализация результатов
plot_spectra(original_data, restored_data, fs)
