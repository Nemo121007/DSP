from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags


# ==========================================================
# Метрики качества
# ==========================================================

def rmse(x_true, x_pred):
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    return np.sqrt(np.mean((x_true - x_pred) ** 2))


def snr_db(clean, estimate):
    clean = np.asarray(clean)
    estimate = np.asarray(estimate)

    noise = clean - estimate

    return 10 * np.log10(
        np.sum(clean ** 2) /
        (np.sum(noise ** 2) + 1e-12)
    )


# ==========================================================
# Загрузка данных
# ==========================================================

def load_signal(path):

    arr = np.loadtxt(path)

    arr = np.asarray(arr)

    if arr.ndim == 0:
        return arr.reshape(1)

    if arr.ndim == 1:
        return arr.astype(float)

    return arr[:, -1].astype(float)


# ==========================================================
# Предобработка
# ==========================================================

def normalize(x):
    x = np.asarray(x)

    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def remove_mean(x):
    return x - np.mean(x)


# ==========================================================
# Оценка задержки между каналами
# ==========================================================

def estimate_lag(signal1, signal2):

    x = normalize(signal1)
    y = normalize(signal2)

    corr = correlate(x, y, mode="full")

    lags = correlation_lags(
        len(x),
        len(y),
        mode="full"
    )

    lag = lags[np.argmax(np.abs(corr))]

    return int(lag)


def shift_signal(x, lag, target_len):

    x = np.asarray(x)

    if lag > 0:

        out = np.r_[np.zeros(lag), x]

        return out[:target_len]

    elif lag < 0:

        lag = -lag

        out = np.r_[x[lag:], np.zeros(lag)]

        return out[:target_len]

    else:

        return x[:target_len]


# ==========================================================
# Leaky NLMS adaptive filter
# ==========================================================

def leaky_nlms(
        desired,
        reference,
        n_taps=64,
        mu=0.01,
        leak=1e-4,
        eps=1e-8
):

    desired = remove_mean(desired)
    reference = remove_mean(reference)

    n = min(len(desired), len(reference))

    desired = desired[:n]
    reference = reference[:n]

    weights = np.zeros(n_taps)

    buffer = np.zeros(n_taps)

    noise_est = np.zeros(n)

    cleaned = np.zeros(n)

    for k in range(n):

        buffer[1:] = buffer[:-1]
        buffer[0] = reference[k]

        noise_est[k] = np.dot(weights, buffer)

        cleaned[k] = desired[k] - noise_est[k]

        norm = np.dot(buffer, buffer)

        step = mu / (norm + eps)

        weights = (
                (1 - leak) * weights +
                step * cleaned[k] * buffer
        )

    return cleaned, noise_est, weights


# ==========================================================
# Основной код
# ==========================================================

def main():

    corrupted_path = Path("data files/CorruptedSignal.txt")
    accel_path = Path("data files/Acceleration.txt")
    clean_path = Path("data files/Signal.txt")

    corrupted = load_signal(corrupted_path)
    accel = load_signal(accel_path)
    clean = load_signal(clean_path)

    n = min(
        len(corrupted),
        len(accel),
        len(clean)
    )

    corrupted = corrupted[:n]
    accel = accel[:n]
    clean = clean[:n]

    # -------------------------------------
    # Компенсация задержки
    # -------------------------------------

    lag = estimate_lag(
        corrupted,
        accel
    )

    accel = shift_signal(
        accel,
        lag,
        n
    )

    print("Оценённый лаг:", lag)

    # -------------------------------------
    # Адаптивная фильтрация
    # -------------------------------------

    corrected, estimated_noise, weights = leaky_nlms(
        desired=corrupted,
        reference=accel,

        n_taps=128,
        mu=0.005,
        leak=1e-4
    )

    np.savetxt(
        "CorrectedSignal.txt",
        corrected
    )

    print(
        "Сохранено:",
        Path("CorrectedSignal.txt").resolve()
    )

    # -------------------------------------
    # Исключаем этап адаптации
    # -------------------------------------

    burn = 256

    clean_eval = clean[burn:]
    corrupted_eval = corrupted[burn:]
    corrected_eval = corrected[burn:]

    rmse_before = rmse(
        clean_eval,
        corrupted_eval
    )

    rmse_after = rmse(
        clean_eval,
        corrected_eval
    )

    snr_before = snr_db(
        clean_eval,
        corrupted_eval
    )

    snr_after = snr_db(
        clean_eval,
        corrected_eval
    )

    print()

    print(
        "RMSE до:",
        rmse_before
    )

    print(
        "RMSE после:",
        rmse_after
    )

    print()

    print(
        "SNR до:",
        snr_before,
        "dB"
    )

    print(
        "SNR после:",
        snr_after,
        "dB"
    )

    print(
        "Прирост:",
        snr_after - snr_before,
        "dB"
    )

    # ==================================================
    # Графики
    # ==================================================

    t = np.arange(n)

    plt.figure(
        figsize=(15, 10)
    )

    plt.subplot(411)

    plt.plot(clean)

    plt.title(
        "Эталонный сигнал"
    )

    plt.grid()

    plt.subplot(412)

    plt.plot(corrupted)

    plt.title(
        "Искажённый сигнал"
    )

    plt.grid()

    plt.subplot(413)

    plt.plot(accel)

    plt.title(
        "Акселерометр"
    )

    plt.grid()

    plt.subplot(414)

    plt.plot(corrected)

    plt.title(
        "Очищенный сигнал"
    )

    plt.grid()

    plt.tight_layout()

    plt.show()

    # ==================================================
    # Отдельное сравнение
    # ==================================================

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        clean,
        label="Эталон"
    )

    plt.plot(
        corrupted,
        alpha=0.7,
        label="Искажённый"
    )

    plt.plot(
        corrected,
        label="После NLMS"
    )

    plt.legend()

    plt.grid()

    plt.title(
        "Сравнение сигналов"
    )

    plt.show()


if __name__ == "__main__":
    main()