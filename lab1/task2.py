from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Utils
# =========================
def wrap_to_pi(theta: float) -> float:
    """
    Нормализует угол(ы) в диапазон [-pi, pi].
    """
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def make_spd(P: np.ndarray) -> np.ndarray:
    """
    Приводит матрицу к виду симметричной положительно определенной.
    """
    P = 0.5 * (P + P.T)

    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals[eigvals < 1e-8] = 1e-8

    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# =========================
# Motion model
# =========================
def motion_model(
    state: Tuple[float, float, float], u: Tuple[float, float, float]
) -> np.ndarray:
    """
    Модель движения для робота с управлением в виде [dr1, dt, dr2], где:
    Args:
        state: модель движения в виде [x, y, theta]
        u: модель управления в виде [dr1, dt, dr2]
    Returns:
        Список нового состояния [x_new, y_new, theta_new] после применения модели движения
    """
    x, y, theta = state
    dr1, dt, dr2 = u

    x_new = x + dt * np.cos(theta + dr1)
    y_new = y + dt * np.sin(theta + dr1)
    theta_new = wrap_to_pi(theta + dr1 + dr2)

    return np.array([x_new, y_new, theta_new])


def jacobian_motion(
    state: Tuple[float, float, float], u: Tuple[float, float, float]
) -> np.ndarray:
    """
    Якобиан модели движения

    Args:
        state: модель движения в виде [x, y, theta]
            - x - координата по оси X
            - y - координата по оси Y
            - theta - угол ориентации робота
        u: модель управления в виде [dr1, dt, dr2]
            - dr1 - поворот перед движением
            - dt - пройденное расстояние
            - dr2 - поворот после завершения движения
    """
    _, _, theta = state
    dr1, dt, _ = u

    angle = theta + dr1

    return np.array(
        [[1, 0, -dt * np.sin(angle)], [0, 1, dt * np.cos(angle)], [0, 0, 1]]
    )  # (25, 32)


# =========================
# Measurement model (RANGE + BEARING)
# =========================
def measurement_model(
    state: Tuple[float, float, float], landmark: Tuple[float, float]
) -> np.ndarray:
    """Модель измерения для RANGE + BEARING
    Args:
        state: модель движения в виде [x, y, theta]
        - x - координата по оси X
        - y - координата по оси Y
        - theta - угол ориентации робота
        landmark: координаты ориентира в виде [lx, ly]
        - lx - координата ориентира по оси X
        - ly - координата ориентира по оси Y
    Returns:
        Список измерения [range, bearing] для данного состояния и ориентира
    """
    x, y, theta = state
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    r = np.sqrt(dx**2 + dy**2)
    bearing = wrap_to_pi(np.arctan2(dy, dx) - theta)

    return np.array([r, bearing])


def jacobian_measurement(
    state: Tuple[float, float, float], landmark: Tuple[float, float]
) -> np.ndarray:
    """
    Якобиан модели измерения для RANGE + BEARING
    Args:
        state: модель движения в виде [x, y, theta]
            - x - координата по оси X
            - y - координата по оси Y
            - theta - угол ориентации робота
        landmark: координаты ориентира в виде [lx, ly]
            - lx - координата ориентира по оси X
            - ly - координата ориентира по оси Y
    Returns:
        Матрица Якобиана H для измерения [range, bearing]
    """
    x, y, _ = state
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    q = dx**2 + dy**2
    r = np.sqrt(q) + 1e-9

    return np.array([[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]])


# =========================
# Noise
# =========================
# Ковариационная матрица шума движения
Q = np.diag([0.2, 0.2, 0.1])
# Ковариационная матрица шума измерений
R = np.diag([0.2, 0.1])  # [range, bearing]


# =========================
# EKF
# =========================
def ekf_step(
    state: Tuple[float, float, float],
    P: np.ndarray,
    u: Tuple[float, float, float],
    measurements: List,
    landmarks: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Шаг EKF для модели движения и измерения RANGE + BEARING
    Args:
        state: модель движения в виде [x, y, theta]
            - x - координата по оси X
            - y - координата по оси Y
            - theta - угол ориентации робота
        P: ковариационная матрица оценки состояния
        u: модель управления в виде [dr1, dt, dr2]
            - dr1 - поворот перед движением
            - dt - пройденное расстояние
            - dr2 - поворот после завершения движения
        measurements: список измерений в виде [(landmark_id, [range, bearing]), ...]
        landmarks: словарь с координатами ориентиров в виде {landmark_id: [lx, ly], ...}
    Returns:
        Список обновленных состояния и ковариационной матрицы после применения EKF
    """
    # Prediction
    F = jacobian_motion(state, u)
    state = motion_model(state, u)
    P = F @ P @ F.T + Q

    # Update: одно групповое обновление по всем ориентирам сразу
    if len(measurements) == 0:
        return state, P

    z_list = []
    z_pred_list = []
    H_list = []

    for lm_id, z in measurements:
        landmark = landmarks[lm_id]

        # Предсказанное значение для данного ориентира
        z_pred = measurement_model(state, landmark)
        H_i = jacobian_measurement(state, landmark)

        z_list.append(np.asarray(z, dtype=float))
        z_pred_list.append(z_pred)
        H_list.append(H_i)

    # Обобщённый вектор измерений и предсказаний
    z = np.concatenate(z_list)  # shape: (2 * m,)
    z_pred = np.concatenate(z_pred_list)  # shape: (2 * m,)

    # Обобщённая матрица Якобиана
    H = np.vstack(H_list)  # shape: (2 * m, 3)

    # Инновация
    innovation = z - z_pred

    # Для каждого ориентира отдельно нормализуем угол
    # Формат измерения: [range, bearing], [range, bearing], ...
    innovation[1::2] = np.array([wrap_to_pi(float(a)) for a in innovation[1::2]])

    # Обобщённая ковариация шума измерений
    R_big = np.kron(np.eye(len(measurements)), R)

    # Ковариационная матрица инновации
    S = H @ P @ H.T + R_big

    # Усиление Калмана
    K = P @ H.T @ np.linalg.inv(S)

    # Обновление состояния
    state = state + K @ innovation
    state[2] = wrap_to_pi(float(state[2]))

    # Обновление ковариации — ровно в том же виде, как у тебя
    P = (np.eye(3) - K @ H) @ P

    return state, P


# =========================
# UKF
# =========================
def generate_sigma_points(
    x: Tuple[float, float, float],
    P: np.ndarray,
    alpha: float = 0.5,
    beta: int = 2,
    kappa: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерация сигма-точек для UKF
    Args:
        x: текущее состояние в виде [x, y, theta]
        P: текущая ковариация состояния
        alpha: параметр, определяющий разброс сигма-точек вокруг среднего (обычно 0.5-1)
        beta: параметр, учитывающий априорную информацию о распределении (обычно 2 для гауссовского)
        kappa: параметр, обычно 0 или 3-n, где n - размерность состояния
    Returns:
        Список сигма-точек, веса для среднего и веса для ковариации
    """
    n = len(x)
    lam = alpha**2 * (n + kappa) - n  # (45)

    scale = n + lam
    if scale <= 0:
        scale = 1e-3

    P = make_spd(P)
    P = P + 1e-5 * np.eye(n)

    sqrt_P = np.linalg.cholesky(scale * P)  # (40)

    sigma_points = [x]
    for i in range(n):  # (39, 40, 41)
        sigma_points.append(x + sqrt_P[:, i])
        sigma_points.append(x - sqrt_P[:, i])

    sigma_points = np.array(sigma_points)

    Wm = np.full(2 * n + 1, 1 / (2 * scale))
    Wc = np.full(2 * n + 1, 1 / (2 * scale))

    Wm[0] = lam / scale
    Wc[0] = lam / scale + (1 - alpha**2 + beta)  # (45)

    return sigma_points, Wm, Wc


def ukf_predict(
    state: Tuple[float, float, float], P: np.ndarray, u: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Предсказание состояния и ковариации с помощью UKF для модели движения RANGE + BEARING
    Args:
        state: модель движения в виде [x, y, theta]
        P: текущая ковариация состояния
        u: модель управления в виде [dr1, dt, dr2]
    Returns:
        Список предсказанных состояния и ковариационной матрицы после применения UKF
    """
    sigma_points, Wm, Wc = generate_sigma_points(state, P)

    propagated = np.array([motion_model(sp, u) for sp in sigma_points])

    x_pred = np.sum(Wm[:, None] * propagated, axis=0)  # (51)
    x_pred[2] = wrap_to_pi(x_pred[2])  # (52)

    P_pred = np.zeros((3, 3))
    for i, prop in enumerate(propagated):
        diff = prop - x_pred
        diff[2] = wrap_to_pi(diff[2])
        P_pred += Wc[i] * np.outer(diff, diff)

    P_pred += Q

    return x_pred, P_pred


def ukf_update(
    state: Tuple[float, float, float], P: np.ndarray, measurement: List, landmark: List
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Усиление состояния и ковариации с помощью UKF для модели измерения RANGE + BEARING
    Args:
        state: модель движения в виде [x, y, theta]
        P: текущая ковариация состояния
        measurement: cписок измерений в виде [range, bearing]
        landmark: список координат ориентира в виде [lx, ly]

    Returns:
        Список обновленных состояния и ковариационной матрицы после применения UKF
    """
    sigma_points, Wm, Wc = generate_sigma_points(state, P)

    Z = np.array([measurement_model(sp, landmark) for sp in sigma_points])

    z_pred = np.sum(Wm[:, None] * Z, axis=0)  # (57)
    z_pred[1] = wrap_to_pi(z_pred[1])

    S = np.zeros((2, 2))
    Pxz = np.zeros((3, 2))

    for i, (z, sp) in enumerate(zip(Z, sigma_points)):
        dz = z - z_pred
        dz[1] = wrap_to_pi(dz[1])

        dx = sp - state
        dx[2] = wrap_to_pi(dx[2])

        S += Wc[i] * np.outer(dz, dz)  # (58)
        Pxz += Wc[i] * np.outer(dx, dz)  # (59)

    # Добавляем шум И jitter сразу
    S += R + 1e-6 * np.eye(2)

    # Усиление Калмана (насколько необходимо сместить оценку в зависимости от нового измерения)
    K = Pxz @ np.linalg.inv(S)  # (60)

    innovation = measurement - z_pred
    innovation[1] = wrap_to_pi(innovation[1])

    state = state + K @ innovation
    state[2] = wrap_to_pi(state[2])

    P = P - K @ S @ K.T  # (12), (36)
    P = make_spd(P)

    return state, P


def ukf_step(
    state: Tuple[float, float, float],
    P: np.ndarray,
    u: Tuple[float, float, float],
    measurements: List,
    landmarks: List,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Шаг фильтра UKF для модели движения и измерения RANGE + BEARING
    Args:
        state: модель состояния в виде [x, y, theta]
        P: текущая ковариация состояния
        u: модель управления в виде [dr1, dt, dr2]
        measurements: список измерений в виде [(landmark_id, [range, bearing]), ...]
        landmarks: список координат ориентиров в виде {landmark_id: [lx, ly], ...}

    Returns:
        Список обновленных состояния и ковариационной матрицы после применения UKF
    """
    state, P = ukf_predict(state, P, u)

    for lm_id, z in measurements:
        landmark = landmarks[lm_id]
        state, P = ukf_update(state, P, z, landmark)

    return state, P


# =========================
# Data loading
# =========================


def load_landmarks(path: Path) -> Dict[int, np.ndarray]:
    """
    Загружает координаты ориентиров из файла.
    Файл должен содержать строки в формате:
    landmark_id x y
    Строки, начинающиеся с "#", пропускаются как комментарии.

    Args:
        path: Путь к файлу с координатами ориентиров

    Returns:
        Словарь где ключи - ID ориентиров (int),
        значения - массивы координат (x, y) размером (2,)
        Формат: {landmark_id: np.array([x, y]), ...}
    """
    landmarks = {}
    with path.open("r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            landmarks[int(parts[0])] = np.array([float(parts[1]), float(parts[2])])
    return landmarks


def load_sensor_data(
    path: Path,
) -> Tuple[List[np.ndarray], List[List[Tuple[int, np.ndarray]]]]:
    """
    Загружает данные одометрии и датчиков (дальномера) из файла.

    Файл должен содержать строки с тегами ODOMETRY и SENSOR в формате:
    - ODOMETRY dr1 dt dr2 (dr1, dt, dr2 - поворот перед, расстояние, поворот после)
    - SENSOR landmark_id range bearing (измерение дальности и угла до ориентира)

    Строки, начинающиеся с "#", пропускаются как комментарии.

    Каждой команде ODOMETRY соответствует последовательность измерений SENSOR,
    полученных после выполнения этой команды.

    Args:
        path: Путь к файлу с данными одометрии и датчиков

    Returns:
        Кортеж содержащий:
        - odometry_data: список команд управления размером (N,),
                         каждая команда - массив [dr1, dt, dr2]:
                         * dr1: поворот перед движением (rad)
                         * dt: пройденное расстояние
                         * dr2: поворот после завершения движения (rad)
        - measurements_seq: список последовательностей измерений размером (N,),
                            каждая последовательность содержит кортежи:
                            [(landmark_id, [range, bearing]), ...]
                            * landmark_id: целочисленный ID ориентира
                            * range: расстояние до ориентира
                            * bearing: угловое направление до ориентира (rad)
    """
    odometry_data = []
    measurements_seq = []
    current_measurements = []

    with path.open("r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "ODOMETRY":
                if odometry_data:
                    measurements_seq.append(current_measurements)
                    current_measurements = []

                u: np.ndarray = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
                odometry_data.append(u)

            elif tag == "SENSOR":
                lm_id: int = int(parts[1])
                r: float = float(parts[2])
                b: float = float(parts[3])

                current_measurements.append((lm_id, np.array([r, b])))

    if current_measurements:
        measurements_seq.append(current_measurements)

    return odometry_data, measurements_seq


# =========================
# Main
# =========================


def run_filter(
    odometry_data: List[np.ndarray],
    measurements_seq: List[List[Tuple[int, np.ndarray]]],
    landmarks: Dict[int, np.ndarray],
    use_ukf: bool = False,
) -> np.ndarray:
    """
    Запускает фильтр EKF или UKF для оценки траектории робота.

    Фильтр обрабатывает последовательность управляющих команд (одометрия)
    и соответствующих им измерений ориентиров (дальность и угол),
    оценивая при этом траекторию робота.

    Args:
        odometry_data: Список команд управления размером (N,),
                       каждая команда - [dr1, dt, dr2]:
                       - dr1: поворот перед движением (rad)
                       - dt: пройденное расстояние
                       - dr2: поворот после завершения движения (rad)
        measurements_seq: Список последовательностей измерений размером (N,),
                         каждая последовательность содержит измерения вида:
                         [(landmark_id, [range, bearing]), ...]
                         - landmark_id: ID ориентира (int)
                         - range: расстояние до ориентира
                         - bearing: угловое направление до ориентира (rad)
        landmarks: Словарь координат ориентиров вида {landmark_id: [x, y], ...}
                   где x, y - координаты ориентира в мировой СК
        use_ukf: Флаг выбора фильтра:
                 - False: использовать EKF (Extended Kalman Filter)
                 - True: использовать UKF (Unscented Kalman Filter)

    Returns:
        Массив оценённой траектории размером (N, 3),
        каждая строка содержит [x, y, theta]:
        - x: координата по оси X
        - y: координата по оси Y
        - theta: угол ориентации робота (rad)
    """
    state: np.ndarray = np.array([0.0, 0.0, 0.0])
    P: np.ndarray = np.eye(3)

    trajectory: List[np.ndarray] = []

    for u, measurements in zip(odometry_data, measurements_seq):
        if use_ukf:
            state, P = ukf_step(state, P, u, measurements, landmarks)
        else:
            state, P = ekf_step(state, P, u, measurements, landmarks)

        trajectory.append(state.copy())

    return np.array(trajectory)


if __name__ == "__main__":

    path_landmarks = Path("data_files/landmarks.dat")
    path_data = Path("data_files/sensor_data_ekf.dat")

    landmarks = load_landmarks(path_landmarks)
    odometry_data, measurements_seq = load_sensor_data(path_data)

    traj_ekf = run_filter(odometry_data, measurements_seq, landmarks, False)
    traj_ukf = run_filter(odometry_data, measurements_seq, landmarks, True)

    plt.figure(figsize=(10, 8))

    plt.plot(traj_ekf[:, 0], traj_ekf[:, 1], label="EKF")
    plt.plot(traj_ukf[:, 0], traj_ukf[:, 1], label="UKF", linestyle="--")

    lm_x = [lm[0] for lm in landmarks.values()]
    lm_y = [lm[1] for lm in landmarks.values()]
    plt.scatter(lm_x, lm_y, marker="x", label="Landmarks")

    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("Range + Bearing: EKF vs UKF")

    plt.show()
