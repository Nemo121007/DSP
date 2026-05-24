import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


FS = 8000.0
NUMTAPS = 401  # нечётное число => тип I, линейная ФЧХ


# ------------------------------------------------------------
# Спецификация
# ------------------------------------------------------------
# Полосы:
# 0..50     -> 0
# 50..150   -> 2
# 150..350  -> 0
# 350..750  -> 1
# 750..900  -> 0
# 900..1500 -> 0.5
# 1500..4000-> 0
BANDS_2D = np.array([
    [0, 50],
    [50, 150],
    [150, 350],
    [350, 750],
    [750, 900],
    [900, 1500],
    [1500, FS / 2],
], dtype=float)

DESIRED_2D = np.array([
    [0, 0],
    [2, 2],
    [0, 0],
    [1, 1],
    [0, 0],
    [0.5, 0.5],
    [0, 0],
], dtype=float)

# Вес можно оставить единичным для чистого LS-сравнения
WEIGHTS = np.ones(len(BANDS_2D), dtype=float)


# ------------------------------------------------------------
# Собственная реализация МНК для симметричного КИХ типа I
# ------------------------------------------------------------
def design_linear_phase_fir_ls(numtaps, fs, bands_2d, desired_2d, grid=30001, band_weights=None):
    """
    МНК-проектирование КИХ-фильтра типа I с линейной ФЧХ.

    Решается задача:
        min || W^(1/2) (B h_half - d) ||_2

    где:
        B(f) = [2 cos(ωM), 2 cos(ω(M-1)), ..., 2 cos(ω), 1]
        h_half = [h[0], ..., h[M]]^T
        numtaps = 2M + 1
    """
    if numtaps % 2 == 0:
        raise ValueError("Для типа I нужен нечётный numtaps.")

    if band_weights is None:
        band_weights = np.ones(len(bands_2d), dtype=float)

    M = (numtaps - 1) // 2
    f = np.linspace(0, fs / 2, grid)
    d = np.zeros_like(f, dtype=float)
    w = np.zeros_like(f, dtype=float)

    # Формируем кусочно-постоянную желаемую АЧХ и веса по полосам
    for i, ((f1, f2), (d1, d2), bw) in enumerate(zip(bands_2d, desired_2d, band_weights)):
        if i < len(bands_2d) - 1:
            mask = (f >= f1) & (f < f2)
        else:
            mask = (f >= f1) & (f <= f2)

        d[mask] = d1
        w[mask] = bw

    omega = 2.0 * np.pi * f / fs

    # Матрица базисных функций из лекции:
    # A(ω) = h[M] + Σ_{n=0}^{M-1} 2 h[n] cos(ω(M-n))
    B = np.column_stack(
        [2.0 * np.cos(omega * (M - n)) for n in range(M)] + [np.ones_like(omega)]
    )

    sw = np.sqrt(w)
    Bh = B * sw[:, None]
    dh = d * sw

    # Численно устойчивее, чем прямые нормальные уравнения
    h_half, *_ = np.linalg.lstsq(Bh, dh, rcond=None)

    # Восстановление полной симметричной ИХ:
    # [h0, h1, ..., hM, h(M-1), ..., h1, h0]
    h_full = np.concatenate([h_half[:-1], h_half[-1:], h_half[-2::-1]])
    return h_full


# ------------------------------------------------------------
# Встроенная функция SciPy
# ------------------------------------------------------------
h_custom = design_linear_phase_fir_ls(
    NUMTAPS, FS, BANDS_2D, DESIRED_2D, grid=30001, band_weights=WEIGHTS
)

h_builtin = signal.firls(
    NUMTAPS,
    BANDS_2D,
    DESIRED_2D,
    weight=WEIGHTS,
    fs=FS
)

# ------------------------------------------------------------
# Сравнение
# ------------------------------------------------------------
w, H_custom = signal.freqz(h_custom, worN=16384, fs=FS)
_, H_builtin = signal.freqz(h_builtin, worN=16384, fs=FS)


def desired_response(freq_hz):
    d = np.zeros_like(freq_hz, dtype=float)
    d[(freq_hz >= 50) & (freq_hz <= 150)] = 2.0
    d[(freq_hz >= 350) & (freq_hz <= 750)] = 1.0
    d[(freq_hz >= 900) & (freq_hz <= 1500)] = 0.5
    return d


D = desired_response(w)

err_custom = np.mean((np.abs(H_custom) - D) ** 2)
err_builtin = np.mean((np.abs(H_builtin) - D) ** 2)

print(f"Max |h_custom - h_builtin| = {np.max(np.abs(h_custom - h_builtin)):.3e}")
print(f"MSE амплитуды (собственная)          = {err_custom:.6e}")
print(f"MSE амплитуды (встроенная)           = {err_builtin:.6e}")

# Проверка симметрии и линейной ФЧХ
print(f"Макс ошибка симметрии (собственная)  = {np.max(np.abs(h_custom - h_custom[::-1])):.3e}")
print(f"Макс ошибка симметрии (встроенная)   = {np.max(np.abs(h_builtin - h_builtin[::-1])):.3e}")

# ------------------------------------------------------------
# График
# ------------------------------------------------------------
plt.figure(figsize=(20, 10))
plt.plot(w, np.abs(H_custom), label="Собственная реализация МНК")
plt.plot(w, np.abs(H_builtin), "--", label="scipy.signal.firls")
plt.plot(w, D, "k:", label="Желаемая АЧХ")
plt.xlim(0, FS / 2)
plt.ylim(-0.2, 2.3)
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.title("Проектирование КИХ-фильтра методом наименьших квадратов с линейной ФЧХ")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
