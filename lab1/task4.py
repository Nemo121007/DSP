import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Constants
# =========================

G = np.array([0.0, 0.0, -9.81])

SIGMA_ACC = 0.02
SIGMA_GYRO = 0.02

SIGMA_GNSS = 0.1
SIGMA_LIDAR = 0.1

C_LIDAR = np.array(
    [
        [0.99376, -0.09722, 0.05466],
        [0.09971, 0.99401, -0.04475],
        [-0.04998, 0.04992, 0.9975],
    ],
    dtype=float,
)

T_LIDAR = np.array([0.5, 0.1, 0.5], dtype=float)


# =========================
# Math utils
# =========================

# (77, s 50)
def skew(a: np.ndarray) -> np.ndarray:
    """
    Преобразует вектор в кососимметричную матрицу (skew-symmetric matrix).

    Args:
        a: Входной вектор размером (3,)

    Returns:
        Кососимметричная матрица размером (3, 3)
    """
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


# (77, s 51)
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Умножение двух кватернионов.

    Args:
        q1: Первый кватернион формата [w, x, y, z]
        q2: Второй кватернион формата [w, x, y, z]

    Returns:
        Результат умножения кватернионов [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_exp(theta: np.ndarray) -> np.ndarray:
    """
    Вычисляет экспоненту вектора угловой скорости (кватернион).

    Args:
        theta: Вектор угловой скорости размером (3,)

    Returns:
        Кватернион формата [w, x, y, z]
    """
    angle = np.linalg.norm(theta)

    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = theta / angle
    return np.hstack([np.cos(angle / 2.0), axis * np.sin(angle / 2.0)])


# (79, 83. s 51, s 55)
def quat_to_R(q: np.ndarray) -> np.ndarray:
    """
    Преобразует кватернион в матрицу ориентации (матрица вращения).

    Args:
        q: Кватернион формата [w, x, y, z]

    Returns:
        Матрица ориентации размером (3, 3)
    """
    q0, q1, q2, q3 = q

    return np.array(
        [
            [
                2 * q0 * q0 - 1 + 2 * q1 * q1,
                2 * q1 * q2 - 2 * q0 * q3,
                2 * q1 * q3 + 2 * q0 * q2,
            ],
            [
                2 * q1 * q2 + 2 * q0 * q3,
                2 * q0 * q0 - 1 + 2 * q2 * q2,
                2 * q2 * q3 - 2 * q0 * q1,
            ],
            [
                2 * q1 * q3 - 2 * q0 * q2,
                2 * q2 * q3 + 2 * q0 * q1,
                2 * q0 * q0 - 1 + 2 * q3 * q3,
            ],
        ],
        dtype=float,
    )


# =========================
# Data loading
# =========================


def load_data(path: str | Path) -> Dict[str, Any]:
    """
    Загружает данные из pickle файла.

    Args:
        path: Путь к файлу данных (относительный или абсолютный)

    Returns:
        Словарь содержащий загруженные данные (imu_f, imu_w, gnss, lidar, gt)
    """
    data_path = Path(path)

    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    data_files_dir = data_path.parent.parent
    if str(data_files_dir) not in sys.path:
        sys.path.insert(0, str(data_files_dir))

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return data


# =========================
# Prepare sequences
# =========================


def prepare_sequences(
    data: Dict[str, Any],
) -> Tuple[
    List[Tuple[float, np.ndarray, np.ndarray]],
    List[Tuple[float, np.ndarray]],
    List[Tuple[float, np.ndarray]],
]:
    """
    Подготавливает последовательности управления и наблюдений для фильтра.

    Args:
        data: Словарь загруженных данных содержащий:
              - imu_f: ускорения IMU
              - imu_w: угловые скорости IMU
              - gnss: GNSS наблюдения
              - lidar: LiDAR наблюдения

    Returns:
        Кортеж из трёх списков:
        - control: список (время, ускорение, угловая_скорость)
        - obs_gnss: список (время, позиция_GNSS)
        - obs_lidar: список (время, позиция_LiDAR)
    """
    imu_f = data["imu_f"]
    imu_w = data["imu_w"]
    gnss = data["gnss"]
    lidar = data["lidar"]

    control = []
    for i in range(1, len(imu_f.data)):
        control.append(
            (
                float(imu_f.t[i]),
                imu_f.data[i - 1].astype(float),
                imu_w.data[i - 1].astype(float),
            )
        )

    obs_gnss = []
    for i, _ in enumerate(gnss.t):
        obs_gnss.append((float(gnss.t[i]), gnss.data[i].astype(float)))

    obs_lidar = []
    for i, _ in enumerate(lidar.t):
        z = C_LIDAR @ lidar.data[i] + T_LIDAR       # (6, s9)
        obs_lidar.append((float(lidar.t[i]), z.astype(float)))

    return control, obs_gnss, obs_lidar


# =========================
# ESKF
# =========================


class ESKF:
    """
    Error-State Kalman Filter (ESKF) для оценки состояния.

    Фильтр оценивает позицию (p), скорость (v) и ориентацию (q).
    Ошибочное состояние описывается 9-мерным вектором ошибок.

    Attributes:
        p: Позиция (3,)
        v: Скорость (3,)
        q: Кватернион ориентации (4,) [w, x, y, z]
        P: Матрица ковариации ошибочного состояния (9, 9)
    """

    def __init__(self, p0: np.ndarray, v0: np.ndarray, q0: np.ndarray) -> None:
        """
        Инициализирует ESKF с начальными условиями.

        Args:
            p0: Начальная позиция (3,)
            v0: Начальная скорость (3,)
            q0: Начальный кватернион ориентации (4,)
        """
        self.p: np.ndarray = p0.astype(float).copy()
        self.v: np.ndarray = v0.astype(float).copy()
        self.q: np.ndarray = q0.astype(float).copy()

        self.P: np.ndarray = np.eye(9, dtype=float)

    # (2)
    def predict(self, f: np.ndarray, w: np.ndarray, dt: float) -> None:
        """
        Предсказательный шаг фильтра (update номинального состояния и ковариации).

        Args:
            f: Ускорение IMU в координатах тела (3,)
            w: Угловая скорость IMU в координатах тела (3,)
            dt: Временной шаг (сек)
        """
        if dt <= 0:
            return

        R = quat_to_R(self.q)
        acc_world = R @ f + G

        # nominal state update
        self.p += self.v * dt + 0.5 * acc_world * dt**2
        self.v += acc_world * dt

        dq = quat_exp(w * dt)   # (82, s 54)
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # error dynamics update
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -skew(R @ f) * dt

        L = np.zeros((9, 6))
        L[3:6, 0:3] = np.eye(3)
        L[6:9, 3:6] = np.eye(3)

        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = SIGMA_ACC**2 * np.eye(3)
        Q[3:6, 3:6] = SIGMA_GYRO**2 * np.eye(3)
        Q *= dt**2

        self.P = F @ self.P @ F.T + L @ Q @ L.T     # (32)

    # (2)
    def update(self, z: np.ndarray, R_meas: np.ndarray) -> None:
        """
        Коррекционный шаг фильтра (обновление состояния на основе измерения).

        Args:
            z: Измерение позиции (3,)
            R_meas: Матрица ковариации измерения (3, 3)
        """
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)

        y = z - self.p

        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y

        self.p += dx[0:3]
        self.v += dx[3:6]

        dtheta = dx[6:9]
        self.q = quat_mul(quat_exp(dtheta), self.q)
        self.q /= np.linalg.norm(self.q)

        self.P = (np.eye(9) - K @ H) @ self.P


# =========================
# Filter run
# =========================


def run_filter(data: Dict[str, Any]) -> np.ndarray:
    """
    Запускает ESKF фильтр на входных данных.

    Args:
        data: Словарь загруженных данных содержащий IMU, GNSS, LiDAR измерения и ground truth

    Returns:
        Массив оценённых позиций траектории размером (N, 3)
    """
    control, obs_gnss, obs_lidar = prepare_sequences(data)

    eskf = ESKF(
        p0=data["gt"].p[0], v0=data["gt"].v[0], q0=np.array([1.0, 0.0, 0.0, 0.0])
    )

    traj = []

    gnss_idx = 0
    lidar_idx = 0
    EPS = 0.05

    for i in range(1, len(control)):

        t, f, w = control[i]
        prev_t = control[i - 1][0]

        dt = t - prev_t

        eskf.predict(f, w, dt)

        # GNSS update
        while gnss_idx < len(obs_gnss) and abs(obs_gnss[gnss_idx][0] - t) < EPS:
            eskf.update(obs_gnss[gnss_idx][1], SIGMA_GNSS**2 * np.eye(3))
            gnss_idx += 1

        # LiDAR update
        while lidar_idx < len(obs_lidar) and abs(obs_lidar[lidar_idx][0] - t) < EPS:
            eskf.update(obs_lidar[lidar_idx][1], SIGMA_LIDAR**2 * np.eye(3))
            lidar_idx += 1

        traj.append(eskf.p.copy())

    return np.array(traj)


# =========================
# Visualization
# =========================


def plot_trajectory(traj: np.ndarray, gt: Any) -> None:
    """
    Визуализирует траекторию оценки фильтра и ground truth.

    Args:
        traj: Массив оценённых позиций размером (N, 3)
        gt: Ground truth данные содержащие поле p с траекторией
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="ESKF")
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label="Ground Truth")

    ax.legend()
    ax.set_title("Trajectory")

    plt.show()


# =========================
# Main
# =========================


def main() -> None:
    """
    Основная функция для запуска фильтра и визуализации результатов.
    """
    data = load_data("data_files/data/data.pkl")

    traj = run_filter(data)

    plot_trajectory(traj, data["gt"])


if __name__ == "__main__":
    main()
