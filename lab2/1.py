import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Синтетический сигнал x(t)
# ==========================================

fs = 100  # частота дискретизации
T = 1.0  # длительность
t = np.arange(0, T, 1 / fs)

# Исходный сигнал
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)

# ==========================================
# ДПФ исходного сигнала
# ==========================================

X = np.fft.rfft(x)

A = np.abs(X)  # амплитудный спектр
phi = np.angle(X)  # фазовый спектр

# ------------------------------------------
# Скрываем сообщение: 101
# ------------------------------------------

bits = np.array([1, 0, 1])

phi_d = np.where(bits == 0, np.pi / 2, -np.pi / 2)

phi_tilde = phi.copy()

# пропускаем DC-компоненту (бин 0)
phi_tilde[1 : 1 + len(bits)] = phi_d

# ==========================================
# Новый спектр
# ==========================================

X_tilde = A * np.exp(1j * phi_tilde)

# Восстановление сигнала
x_tilde = np.fft.irfft(X_tilde)

A_tilde = np.abs(X_tilde)
phi_tilde = np.angle(X_tilde)

freq = np.fft.rfftfreq(len(x), d=1 / fs)

# ==========================================
# Графики
# ==========================================

plt.figure(figsize=(14, 10))

# -----------------------------
# x(t)
# -----------------------------
plt.subplot(221)

plt.plot(t, x, label="Исходный")
plt.plot(t, x_tilde, "--", label="Стего")

plt.title("Сигнал во времени x(t)")
plt.xlabel("t, сек")
plt.ylabel("Амплитуда")

plt.legend()
plt.grid()

# -----------------------------
# |X(f)|
# -----------------------------
plt.subplot(222)

plt.stem(freq, A, label="Исходный")
plt.stem(freq, A_tilde, linefmt="r--", markerfmt="ro", basefmt=" ", label="Стего")

plt.title("Амплитудный спектр |X(f)|")
plt.xlabel("f, Гц")
plt.ylabel("Амплитуда")

plt.legend()
plt.grid()

# -----------------------------
# φ(f)
# -----------------------------
plt.subplot(223)

plt.plot(freq, phi, "o-", label="Исходный")
plt.plot(freq, phi_tilde, "o--", label="Стего")

plt.title("Фазовый спектр φ(f)")
plt.xlabel("f, Гц")
plt.ylabel("Фаза, рад")

plt.legend()
plt.grid()

# -----------------------------
# Разность сигналов
# -----------------------------
plt.subplot(224)

plt.plot(t, x_tilde - x)

plt.title("Изменение сигнала")
plt.xlabel("t, сек")
plt.ylabel("Ошибка")

plt.grid()

plt.tight_layout()
plt.show()
