import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Параметры дискретизации
# -----------------------------
dt = 0.01
L = 12.0
N = int(2 * L / dt) + 1
t = np.linspace(-L, L, N)

# Достаточно плотная частотная сетка
nfft = 2**15


# -----------------------------
# Сигналы
# -----------------------------
def x1(t):
    """Вычисляет значение сигнала x1(t) = exp(-t^2).

    Args:
        t (float или np.ndarray): Момент(ы) времени.

    Returns:
        float или np.ndarray: Значение сигнала в заданные моменты времени.
    """
    return np.exp(-(t**2))


def x2(t):
    """Вычисляет значение сигнала x2(t).

    Сигнал равен cos(pi * t / 2) при |t| <= 1.0 и 0 иначе.

    Args:
        t (float или np.ndarray): Момент(ы) времени.

    Returns:
        float или np.ndarray: Значение сигнала в заданные моменты времени.
    """
    return np.where(np.abs(t) <= 1.0, np.cos(np.pi * t / 2.0), 0.0)


# -----------------------------
# Численное CTFT через FFT
# -----------------------------
def ctft_via_fft(x, dt, nfft):
    """Приближенное вычисление непрерывного преобразования Фурье (CTFT) через FFT.

    Формула: X(ω) = ∫ x(t)e^{-iωt}dt через FFT на равномерной сетке.

    Args:
        x (np.ndarray): Значения сигнала во временной области.
        dt (float): Шаг дискретизации во времени.
        nfft (int): Количество точек для FFT.

    Returns:
        tuple: Кортеж, содержащий:
            - np.ndarray: Массив частот (omega).
            - np.ndarray: Комплексные значения спектра X(ω).
    """
    X = dt * np.fft.fftshift(np.fft.fft(x, n=nfft))
    omega = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
    return omega, X


# -----------------------------
# Теоретические спектры
# -----------------------------
def X1_theory(omega):
    """
    Расчет теоретического спектра для сигнала:

        x1(t) = exp(-t^2)

    через определение преобразования Фурье:
        X1(ω)=∫ exp(-t^2)e^{-iωt}dt
    """

    t = np.linspace(-10, 10, 5000)

    x = x1(t)

    X = []

    for w in np.atleast_1d(omega):
        integrand = x * np.exp(-1j * w * t)

        X.append(np.trapezoid(integrand, t))

    return np.array(X)


def X2_theory(omega):
    t = np.linspace(-1, 1, 5000)

    x = x2(t)

    X = []

    for w in np.atleast_1d(omega):
        integrand = x * np.exp(-1j * w * t)
        X.append(np.trapezoid(integrand, t))

    return np.array(X)


# -----------------------------
# Расчёт
# -----------------------------
x1_s = x1(t)
x2_s = x2(t)

omega1, X1_num = ctft_via_fft(x1_s, dt, nfft)
omega2, X2_num = ctft_via_fft(x2_s, dt, nfft)

X1_th = X1_theory(omega1)
X2_th = X2_theory(omega2)

# Для сравнения берём умеренный диапазон частот
wmax = 15.0
m1 = np.abs(omega1) <= wmax
m2 = np.abs(omega2) <= wmax

err1 = np.max(np.abs(np.abs(X1_num[m1]) - X1_th[m1]))
err2 = np.max(np.abs(np.abs(X2_num[m2]) - np.abs(X2_th[m2])))

print(f"Max abs error x1(t) = exp(-t^2): {err1:.6e}")
print(f"Max abs error x2(t): {err2:.6e}")


# -----------------------------
# Графики
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(
    omega1[m1], np.abs(X1_num[m1]), label=r"Численно: $|\mathrm{FFT}|\cdot \Delta t$"
)
plt.plot(omega1[m1], np.abs(X1_th[m1]), "--", label="Теория")
plt.title(r"Сигнал $x_1(t)=e^{-t^2}$")
plt.xlabel(r"$\omega$")
plt.ylabel(r"$|X(\omega)|$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(
    omega2[m2], np.abs(X2_num[m2]), label=r"Численно: $|\mathrm{FFT}|\cdot \Delta t$"
)
plt.plot(omega2[m2], np.abs(X2_th[m2]), "--", label="Теория")
plt.title(r"Сигнал $x_2(t)=\cos(\pi t/2),\ |t|\leq 1$")
plt.xlabel(r"$\omega$")
plt.ylabel(r"$|X(\omega)|$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
