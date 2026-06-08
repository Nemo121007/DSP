import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin2, freqz

Fs = 44_000.0
Fc = 30_000.0


def mic_response(f_norm: np.ndarray) -> np.ndarray:
    """
    Исходная АЧХ микрофона.
    """
    F = f_norm * (Fs / 2.0)  # Hz
    return 1.0 - F / Fc


def desired_compensation(f_norm: np.ndarray) -> np.ndarray:
    """
    Желаемая АЧХ компенсации: обратная к АЧХ микрофона
    в полосе, доступной после дискретизации.
    """
    return 1.0 / mic_response(f_norm)


# FIR Тип I
numtaps = 101

f_norm = np.linspace(0.0, 1.0, 2049)

# АЧХ микрофона
H_mic = mic_response(f_norm)

gain = desired_compensation(f_norm)

# FIR
h = firwin2(numtaps=numtaps, freq=f_norm, gain=gain, window="hann")

print("FIR коэффициенты:")
print(np.array2string(h, precision=12, separator=", "))

w, H = freqz(h, worN=4096)
f_hz = w * Fs / (2 * np.pi)

f_target_hz = f_norm * Fs / 2.0

plt.figure(figsize=(10, 5))
plt.plot(f_target_hz, H_mic, label="До коррекции: H_a(F)")
plt.plot(f_target_hz, gain, "--", label="Желаемая компенсация: 1 / H_a(F)")
plt.plot(f_hz, np.abs(H), label="АЧХ FIR")
plt.xlabel("Частота, Hz")
plt.ylabel("Амплитуда")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
