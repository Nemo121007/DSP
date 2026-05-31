import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin2, freqz

Fs = 44_000.0
Fc = 30_000.0


def desired_compensation(f_norm: np.ndarray) -> np.ndarray:
    """Вычисляет желаемую амплитудно-частотную характеристику (АЧХ) для компенсации.

    Эта функция вычисляет желаемую АЧХ на интервале нормализованных частот [0, 1],
    где 1 соответствует частоте Найквиста (Fs/2).

    Args:
        f_norm (np.ndarray): Массив нормализованных частот в диапазоне [0.0, 1.0].

    Returns:
        np.ndarray: Массив значений желаемой амплитуды для соответствующих частот.
    """
    F = f_norm * (Fs / 2.0)  # analog frequency in Hz
    return 1.0 / (1.0 - F / Fc)


# Choose FIR length: odd => linear-phase Type I
numtaps = 101

# Dense grid for magnitude approximation
f_norm = np.linspace(0.0, 1.0, 2049)
gain = desired_compensation(f_norm)

# FIR design
h = firwin2(numtaps=numtaps, freq=f_norm, gain=gain, window="hann")

print("FIR coefficients:")
print(np.array2string(h, precision=12, separator=", "))

# Check frequency response
w, H = freqz(h, worN=4096)
f_hz = w * Fs / (2 * np.pi)

plt.figure(figsize=(8, 4))
plt.plot(f_hz, np.abs(H), label="FIR magnitude")
plt.plot(f_norm * Fs / 2.0, gain, "--", label="Desired magnitude")
plt.xlabel("Frequency, Hz")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
