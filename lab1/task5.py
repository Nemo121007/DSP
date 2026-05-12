from typing import Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import correlate, correlation_lags

# =========================
# Constants
# =========================

FS = 100000.0   # Hz
C = 1125.0      # ft/s


# =========================
# Load data
# =========================
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает данные передатчиков и приёмника из файлов.

    Returns:
        s: массив сигналов передатчиков формы (4, N)
        r: массив сигнала приёмника формы (N,)
    """
    s = np.loadtxt("data_files/Transmitter.txt", dtype=np.float64)
    r = np.loadtxt("data_files/Receiver.txt", dtype=np.float64)

    if s.ndim != 2:
        raise ValueError(f"Transmitter.txt must be 2D, got shape {s.shape}")
    if r.ndim != 1:
        raise ValueError(f"Receiver.txt must be 1D, got shape {r.shape}")
    if s.shape[0] != 4:
        raise ValueError(f"Expected 4 transmitter signals, got shape {s.shape}")

    return s, r


# =========================
# Estimate delays
# =========================
def estimate_delays(transmitters: np.ndarray, received: np.ndarray) -> np.ndarray:
    """
    Оценивает задержки между каждым сигналом передатчика и сигналом приёмника.
    Returns:
        delays: массив задержек в секундах, shape (4,)
    """
    delays = np.zeros(transmitters.shape[0], dtype=np.float64)

    for i, transmitter in enumerate(transmitters):
        corr = correlate(received, transmitter, mode="full")
        lags = correlation_lags(len(received), len(transmitter), mode="full")

        peak_index = np.argmax(corr)
        lag_samples = lags[peak_index]

        delays[i] = lag_samples / FS

    return delays


# =========================
# Multilateration
# =========================
def residuals(p: np.ndarray, speakers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Вектор невязок для задачи multilateration.

    p: (3,) -> [x, y, z]
    speakers: (4, 3)
    distances: (4,)
    """
    return np.linalg.norm(speakers - p, axis=1) - distances


def estimate_position(distances: np.ndarray) -> np.ndarray:
    """
    Оценивает 3D-позицию источника по расстояниям до четырёх передатчиков.

    distances: shape (4,)
    returns: shape (3,)
    """
    speakers = np.array(
        [
            [0.0, 0.0, 10.0],
            [20.0, 0.0, 10.0],
            [20.0, 20.0, 10.0],
            [0.0, 20.0, 10.0],
        ],
        dtype=np.float64,
    )

    x0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    result = least_squares(residuals, x0, args=(speakers, distances))
    return result.x


# =========================
# Main
# =========================
def main() -> None:
    transmitters, received = load_data()

    # Задержки в секундах
    delays = estimate_delays(transmitters, received)

    # Расстояния в футах
    distances = C * delays

    # Оценка позиции
    pos = estimate_position(distances)

    print("Estimated position:", pos)


if __name__ == "__main__":
    main()