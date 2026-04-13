from typing import Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import correlate

# =========================
# Constants
# =========================

FS = 100000  # Hz - частота дискретизации
C = 332  # ft/s - скорость звука в метрах в секунду


# =========================
# Load data
# =========================
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает данные передатчиков и приёмника из файлов.

    Returns:
        Кортеж содержащий:
        - s: Массив сигналов передатчиков размером (4, N)
        - r: Массив сигнала приёмника размером (N,)
    """
    s = np.loadtxt("data_files/Transmitter.txt")  # shape: (4, N)
    r = np.loadtxt("data_files/Receiver.txt")  # shape: (N,)
    return s, r


# =========================
# Estimate delays
# =========================
def estimate_delays(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Оценивает задержки между сигналами передатчиков и приёмника.
    
    Args:
        s: Массив сигналов передатчиков размером (4, N)
        r: Массив сигнала приёмника размером (N,)
        
    Returns:
        Массив оцененных задержек в секундах для каждого передатчика
    """
    delays: list = []

    for i in range(s.shape[0]):
        # correlate(s, r) ищет, на сколько нужно сдвинуть s, чтобы получить r.
        # Для физической задачи r(t) ~ s(t - T), пик будет на положительном лаге +T.
        corr = correlate(s[i], r, mode="full")

        # Индекс пика корреляции
        lag_index = np.argmax(corr)

        # Нулевой лаг находится в индексе len(r) - 1
        zero_lag_index = len(r) - 1

        # Вычисляем сдвиг в отсчетах
        lag = lag_index - zero_lag_index

        T = lag / FS
        delays.append(T)

    return np.array(delays)


def residuals(p: np.ndarray, speakers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Вычисляет вектор невязок.

    Каждый элемент вектора представляет разницу между расчётным расстоянием
    от точки до передатчика и измеренным расстоянием.

    Args:
        p: Оцениваемая позиция размером (3,) [x, y, z]
        speakers: Позиции передатчиков размером (4, 3)
        distances: Измеренные расстояния до передатчиков размером (4,)

    Returns:
        Вектор невязок размером (4,)
    """
    res = []

    for i, speaker in enumerate(speakers):
        # f_i(x)
        dist = np.linalg.norm(p - speaker)
        # y_i - f_i(x)
        res.append(dist - distances[i])

    return np.array(res)


def estimate_position(distances: np.ndarray) -> np.ndarray:
    """
    Оценивает трёхмерную позицию на основе расстояний до четырёх передатчиков.

    Решает задачу многолатерации (multilateration) методом наименьших квадратов,
    исходя из известных позиций четырёх передатчиков и расстояний до них.

    Args:
        distances: Расстояния до четырёх передатчиков размером (4,)

    Returns:
        Оценённая позиция размером (3,) [x, y, z] в футах
    """
    speakers = np.array(
        [[0, 0, 10], [20, 0, 10], [0, 20, 10], [20, 20, 10]], dtype=float
    )

    x0 = np.array([10, 10, 5], dtype=float)  # начальное приближение

    res = least_squares(residuals, x0, args=(speakers, distances))

    return res.x


# =========================
# Main
# =========================
def main() -> None:
    """
    Основная функция для локализации источника звука.

    Выполняет следующие шаги:
    1. Загружает данные с передатчиков и приёмника
    2. Оценивает временные задержки между сигналами
    3. Вычисляет расстояния на основе задержек
    4. Определяет позицию источника
    5. Выводит результат
    """
    s, r = load_data()

    # Оценка задержек
    T = estimate_delays(s, r)

    # Вычисление расстояний на основе задержек
    # y_i = f(x) + ξ
    R = C * T

    # Определение позиции источника
    pos = estimate_position(R)

    print("Estimated position:", pos)


if __name__ == "__main__":
    main()
