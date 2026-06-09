from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags

# ==========================================================
# Метрики качества
# ==========================================================


def rmse(x_true, x_pred):
    """Вычисляет среднеквадратичную ошибку (RMSE) между двумя сигналами.

    Args:
        x_true (array_like): Истинный (эталонный) сигнал.
        x_pred (array_like): Предсказанный (оцениваемый) сигнал.

    Returns:
        float: Значение RMSE.
    """
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    return np.sqrt(np.mean((x_true - x_pred) ** 2))


def snr_db(clean, estimate):
    """Вычисляет отношение сигнал/шум (SNR) в децибелах (дБ).

    Args:
        clean (array_like): Чистый (эталонный) сигнал.
        estimate (array_like): Очищенный (оцениваемый) сигнал с остаточным шумом.

    Returns:
        float: Значение SNR в децибелах.
    """
    clean = np.asarray(clean)
    estimate = np.asarray(estimate)

    noise = clean - estimate

    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-12))


# ==========================================================
# Загрузка данных
# ==========================================================


def load_signal(path):
    """Загружает сигнал из текстового файла.

    Если файл содержит несколько столбцов, возвращает последний столбец.

    Args:
        path (str | Path): Путь к текстовому файлу с данными.

    Returns:
        np.ndarray: Одномерный массив загруженных данных типа float.
    """

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
    """Нормализует сигнал (вычитает среднее и делит на стандартное отклонение).

    Args:
        x (array_like): Входной сигнал.

    Returns:
        np.ndarray: Нормализованный сигнал.
    """
    x = np.asarray(x)

    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def remove_mean(x):
    """Удаляет постоянную составляющую (среднее значение) из сигнала.

    Args:
        x (array_like): Входной сигнал.

    Returns:
        np.ndarray: Сигнал с нулевым средним.
    """
    return x - np.mean(x)


# ==========================================================
# Оценка задержки между каналами
# ==========================================================


def estimate_lag(signal1, signal2):
    """Оценивает задержку (лаг) между двумя сигналами с помощью кросс-корреляции.

    Args:
        signal1 (array_like): Первый сигнал (опорный).
        signal2 (array_like): Второй сигнал (сдвинутый).

    Returns:
        int: Оцененная задержка в отсчетах. Положительное значение означает,
        что signal2 отстает от signal1.
    """

    x = normalize(signal1)
    y = normalize(signal2)

    corr = correlate(x, y, mode="full")

    lags = correlation_lags(len(x), len(y), mode="full")

    lag = lags[np.argmax(np.abs(corr))]

    return int(lag)


def shift_signal(x, lag, target_len):
    """Сдвигает сигнал на заданное количество отсчетов.

    Дополняет сигнал нулями при задержке или обрезает начало при опережении.

    Args:
        x (array_like): Входной сигнал.
        lag (int): Величина сдвига в отсчетах.
        target_len (int): Требуемая длина выходного сигнала.

    Returns:
        np.ndarray: Сдвинутый сигнал заданной длины.
    """

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


def leaky_nlms(desired, reference, n_taps=64, mu=0.01, leak=1e-4, eps=1e-8):
    """Адаптивный фильтр NLMS (Normalized Least Mean Squares) с утечкой (Leakage).

    Применяется для активного подавления шума.

    Args:
        desired (array_like): Искаженный сигнал (полезный сигнал + шум).
        reference (array_like): Опорный сигнал (сигнал шума, коррелирующий с шумом в desired).
        n_taps (int, optional): Порядок фильтра (количество весов). По умолчанию 64.
        mu (float, optional): Шаг адаптации. По умолчанию 0.01.
        leak (float, optional): Коэффициент утечки, предотвращающий неограниченный рост весов. По умолчанию 1e-4.
        eps (float, optional): Малая константа для предотвращения деления на ноль. По умолчанию 1e-8.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Очищенный сигнал (ошибка адаптации).
            - Оцененный шум.
            - Итоговые веса фильтра.
    """

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

        weights = (1 - leak) * weights + step * cleaned[k] * buffer

    return cleaned, noise_est, weights


# ==========================================================
# Основной код
# ==========================================================


def main():
    """Главная функция для выполнения очистки сигнала алгоритмом Leaky NLMS.

    Считывает файлы, оценивает задержку между акселерометром и искаженным сигналом,
    применяет адаптивный фильтр, вычисляет метрики RMSE и SNR,
    сохраняет очищенный сигнал и строит графики для визуального анализа.
    """

    corrupted_path = Path("data files/CorruptedSignal.txt")
    accel_path = Path("data files/Acceleration.txt")
    clean_path = Path("data files/Signal.txt")

    corrupted = load_signal(corrupted_path)
    accel = load_signal(accel_path)
    clean = load_signal(clean_path)

    n = min(len(corrupted), len(accel), len(clean))

    corrupted = corrupted[:n]
    accel = accel[:n]
    clean = clean[:n]

    # -------------------------------------
    # Компенсация задержки
    # -------------------------------------

    lag = estimate_lag(corrupted, accel)

    accel = shift_signal(accel, lag, n)

    print("Оценённый лаг:", lag)

    # -------------------------------------
    # Адаптивная фильтрация
    # -------------------------------------

    corrected, estimated_noise, weights = leaky_nlms(
        desired=corrupted, reference=accel, n_taps=128, mu=0.005, leak=1e-4
    )

    np.savetxt("CorrectedSignal.txt", corrected)

    print("Сохранено:", Path("CorrectedSignal.txt").resolve())

    # -------------------------------------
    # Исключаем этап адаптации
    # -------------------------------------

    burn = 256

    clean_eval = clean[burn:]
    corrupted_eval = corrupted[burn:]
    corrected_eval = corrected[burn:]

    rmse_before = rmse(clean_eval, corrupted_eval)

    rmse_after = rmse(clean_eval, corrected_eval)

    snr_before = snr_db(clean_eval, corrupted_eval)

    snr_after = snr_db(clean_eval, corrected_eval)

    print()

    print("RMSE до:", rmse_before)

    print("RMSE после:", rmse_after)

    print()

    print("SNR до:", snr_before, "dB")

    print("SNR после:", snr_after, "dB")

    print("Прирост:", snr_after - snr_before, "dB")

    # ==================================================
    # Графики
    # ==================================================

    t = np.arange(n)

    plt.figure(figsize=(15, 10))
    plt.subplot(411)
    plt.plot(clean)
    plt.title("Эталонный сигнал")
    plt.grid()

    plt.subplot(412)
    plt.plot(corrupted)
    plt.title("Искажённый сигнал")
    plt.grid()

    plt.subplot(413)
    plt.plot(accel)
    plt.title("Акселерометр")
    plt.grid()

    plt.subplot(414)
    plt.plot(corrected)
    plt.title("Очищенный сигнал")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # ==================================================
    # Отдельное сравнение
    # ==================================================

    plt.figure(figsize=(14, 5))
    plt.plot(clean, label="Эталон")
    plt.plot(corrupted, alpha=0.7, label="Искажённый")
    plt.plot(corrected, label="После NLMS")
    plt.legend()
    plt.grid()
    plt.title("Сравнение сигналов")
    plt.show()


if __name__ == "__main__":
    main()
