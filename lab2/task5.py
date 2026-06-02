import numpy as np
from scipy.io import wavfile


def inverse_quarter_permutation_spectrum(x: np.ndarray) -> np.ndarray:
    """
    Восстанавливает сигнал после перестановки спектральных четвертей:
    [A B C D] -> [C B D A]
    """
    x = np.asarray(x, dtype=np.float64)
    X = np.fft.fft(x)
    n = len(X)

    dc = X[0]

    # Для вещественного сигнала:
    # - при чётной длине есть Nyquist-bin;
    # - при нечётной длине его нет.
    if n % 2 == 0:
        nyquist = X[n // 2]
        positive = X[1:n // 2]
    else:
        nyquist = None
        positive = X[1:(n + 1) // 2]

    # Делим без потери остатка
    p1, p2, p3, p4 = np.array_split(positive, 4)

    # Искажённый порядок был [C B D A]
    # Значит восстановление: [A B C D] = [p4, p2, p1, p3]
    restored_positive = np.concatenate([p4, p2, p1, p3])

    # Достраиваем полный спектр
    if nyquist is None:
        X_restored = np.concatenate([
            [dc],
            restored_positive,
            np.conj(restored_positive[::-1])
        ])
    else:
        X_restored = np.concatenate([
            [dc],
            restored_positive,
            [nyquist],
            np.conj(restored_positive[::-1])
        ])

    # Приводим длину к исходной
    X_restored = X_restored[:n] if len(X_restored) > n else np.pad(
        X_restored, (0, n - len(X_restored)), mode="constant"
    )

    return np.fft.ifft(X_restored).real


def save_wav_like_original(path_in: str, path_out: str):
    fs, data = wavfile.read(path_in)

    if data.ndim == 1:
        xr = inverse_quarter_permutation_spectrum(data)

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            xr = np.clip(np.rint(xr), info.min, info.max).astype(data.dtype)
        else:
            xr = xr.astype(data.dtype)

    else:
        channels = []
        for ch in range(data.shape[1]):
            xr_ch = inverse_quarter_permutation_spectrum(data[:, ch])
            channels.append(xr_ch)

        xr = np.stack(channels, axis=1)

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            xr = np.clip(np.rint(xr), info.min, info.max).astype(data.dtype)
        else:
            xr = xr.astype(data.dtype)

    wavfile.write(path_out, fs, xr)
    return fs, data, xr


input_file = "data/test5.wav"
output_file = "test5_restored.wav"

fs, original_data, restored_data = save_wav_like_original(input_file, output_file)

print(f"Готово. Файл сохранён как: {output_file}")
print(f"Частота дискретизации: {fs} Гц")
print(f"Длина сигнала: {len(original_data)} отсчётов")