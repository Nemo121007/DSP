import numpy as np
from scipy.signal import firwin2, freqz
import matplotlib.pyplot as plt

Fs = 44_000.0
Fc = 30_000.0


def desired_compensation(f_norm: np.ndarray) -> np.ndarray:
    """
    Desired magnitude response on [0, 1],
    where 1 corresponds to Nyquist (Fs/2).
    """
    F = f_norm * (Fs / 2.0)          # analog frequency in Hz
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
f_hz = w * Fs / (2*np.pi)

plt.figure(figsize=(8, 4))
plt.plot(f_hz, np.abs(H), label="FIR magnitude")
plt.plot(f_norm * Fs / 2.0, gain, "--", label="Desired magnitude")
plt.xlabel("Frequency, Hz")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
