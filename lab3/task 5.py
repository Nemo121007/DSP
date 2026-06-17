from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# Метрики качества
# ==========================================================
def snr_db(clean, estimate):
    """Вычисляет отношение сигнал/шум (SNR) в децибелах."""
    clean = np.asarray(clean, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    noise = clean - estimate
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-12))


# ==========================================================
# Загрузка данных
# ==========================================================
def load_signal(path):
    """Загружает сигнал из txt-файла. Если столбцов несколько, берёт последний."""
    arr = np.loadtxt(path)
    arr = np.asarray(arr)

    if arr.ndim == 0:
        return arr.reshape(1).astype(float)

    if arr.ndim == 1:
        return arr.astype(float)

    return arr[:, -1].astype(float)


# ==========================================================
# Предобработка
# ==========================================================
def remove_mean(x):
    """Удаляет постоянную составляющую."""
    x = np.asarray(x, dtype=float)
    return x - np.mean(x)


# ==========================================================
# Leaky NLMS адаптивный фильтр
# ==========================================================

def leaky_nlms(desired, reference, n_taps=128, mu=0.005, leak=1e-4, eps=1e-8):
    """
    Адаптивный фильтр NLMS с утечкой.

    desired   — искажённый сигнал PPG
    reference — сигнал акселерометра
    n_taps    — число коэффициентов FIR-фильтра
    mu        — базовый шаг адаптации
    leak      — коэффициент утечки
    eps       — защита от деления на ноль
    """
    desired = remove_mean(desired)
    reference = remove_mean(reference)

    n = min(len(desired), len(reference))
    desired = desired[:n]
    reference = reference[:n]

    weights = np.zeros(n_taps, dtype=float)
    buffer = np.zeros(n_taps, dtype=float)

    noise_est = np.zeros(n, dtype=float)
    cleaned = np.zeros(n, dtype=float)

    for k in range(n):
        # Сдвигаем окно опорного сигнала
        buffer[1:] = buffer[:-1]
        buffer[0] = reference[k]

        # Оценка шума по текущим весам
        noise_est[k] = np.dot(weights, buffer)

        # Ошибка адаптации — это и есть очищенный сигнал
        cleaned[k] = desired[k] - noise_est[k]

        # NLMS-нормировка шага
        norm = np.dot(buffer, buffer)
        step = mu / (norm + eps)

        # Обновление весов с утечкой
        weights = (1.0 - leak) * weights + step * cleaned[k] * buffer

    return cleaned, noise_est, weights


# ==========================================================
# Основной код
# ==========================================================
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

# Адаптивная фильтрация
corrected, estimated_noise, weights = leaky_nlms(
        desired=corrupted,
        reference=accel,
        n_taps=128,
        mu=0.005,
        leak=1e-4,
)

out_path = Path("CorrectedSignal.txt")
np.savetxt(out_path, corrected, fmt="%.10f")
print("Сохранено:", out_path.resolve())

# Оценка качества, пропуская начальный переходный участок
burn = 256

clean_eval = clean[burn:]
corrupted_eval = corrupted[burn:]
corrected_eval = corrected[burn:]


rmse_before = np.sqrt(np.mean((clean_eval - corrupted_eval) ** 2))
rmse_after = np.sqrt(np.mean((clean_eval - corrected_eval) ** 2))
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

plt.figure(figsize=(14, 5))
plt.plot(clean, label="Эталон")
plt.plot(corrupted, alpha=0.7, label="Искажённый")
plt.plot(corrected, label="После NLMS")
plt.legend()
plt.grid()
plt.title("Сравнение сигналов")
plt.show()
