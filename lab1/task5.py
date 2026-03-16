import numpy as np
from scipy.signal import correlate
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# =========================
# Constants
# =========================

FS = 100000  # Hz
C = 1125     # ft/s


# =========================
# Load data
# =========================

def load_data():
    s = np.loadtxt("data_files/Transmitter.txt")  # shape: (4, N)
    r = np.loadtxt("data_files/Receiver.txt")     # shape: (N,)
    return s, r


# =========================
# Estimate delays
# =========================

def estimate_delays(s, r):

    delays = []

    for i in range(s.shape[0]):
        corr = correlate(r, s[i], mode='full')

        lag = np.argmax(corr) - (len(s[i]) - 1)

        T = lag / FS
        delays.append(T)

    return np.array(delays)


# =========================
# Multilateration
# =========================

def residuals(p, speakers, distances):

    res = []

    for i in range(len(speakers)):
        dist = np.linalg.norm(p - speakers[i])
        res.append(dist - distances[i])

    return np.array(res)


def estimate_position(distances):

    speakers = np.array([
        [0, 0, 10],
        [20, 0, 10],
        [0, 20, 10],
        [20, 20, 10]
    ])

    x0 = np.array([10, 10, 5])  # начальное приближение

    res = least_squares(
        residuals,
        x0,
        args=(speakers, distances)
    )

    return res.x


# =========================
# Main
# =========================

def main():

    s, r = load_data()

    # 1. задержки
    T = estimate_delays(s, r)

    # 2. расстояния
    R = C * T

    # 3. позиция
    pos = estimate_position(R)

    print("Estimated position:", pos)


if __name__ == "__main__":
    main()