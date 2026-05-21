import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Constants
# =========================
# Вектор ускорения свободного падения в мировой системе координат
G = np.array([0.0, 0.0, -9.81])

# Среднеквадратичное отклонение f_k
SIGMA_ACC = 0.02
# Среднеквадратичное отклонение w_k
SIGMA_GYRO = 0.02

# Среднеквадратичное отклонение ошибки позиционирования GNSS
SIGMA_GNSS = 0.1
# Среднеквадратичное отклонение ошибки локализации по LiDAR.
SIGMA_LIDAR = 0.01

# Параметры калибровки LiDAR (Условие задачи)
# Матрица поворота лидара относительно IMU
C_LIDAR = np.array(
    [
        [0.99376, -0.09722, 0.05466],
        [0.09971, 0.99401, -0.04475],
        [-0.04998, 0.04992, 0.9975],
    ],
    dtype=float,
)
# Вектор смещения центра лидара относительно IMU.
T_LIDAR = np.array([0.5, 0.1, 0.5], dtype=float)


# =========================
# Math utils
# =========================
def skew(a: np.ndarray) -> np.ndarray:
    """
    Вычисляет кососимметричную матрицу для 3D вектора.
    Args:
        a (np.ndarray): Входной 3D вектор формы (3,).

    Returns:
        np.ndarray: Кососимметричная матрица формы (3, 3).
    """
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Выполняет умножение двух кватернионов.
    
    Args:
        q1 (np.ndarray): Первый кватернион [w, x, y, z].
        q2 (np.ndarray): Второй кватернион [w, x, y, z].

    Returns:
        np.ndarray: Результат перемножения кватернионов [w, x, y, z].
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
    Вычисляет экспоненту от вектора угла поворота, возвращая кватернион.
    
    Args:
        theta (np.ndarray): Вектор угла поворота (3,) (ось поворота, умноженная на угол).

    Returns:
        np.ndarray: Кватернион [w, x, y, z], представляющий данный поворот.
    """
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = theta / angle
    return np.hstack([np.cos(angle / 2.0), axis * np.sin(angle / 2.0)])


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """
    Преобразует кватернион в матрицу поворота (косинусов направлений).
    
    Args:
        q (np.ndarray): Кватернион [w, x, y, z].

    Returns:
        np.ndarray: Матрица поворота формы (3, 3).
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
        path (str | Path): Путь к файлу данных (.pkl).

    Returns:
        Dict[str, Any]: Словарь с загруженными данными (GNSS, LiDAR, IMU и Ground Truth).
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
    Подготавливает последовательности измерений для подавления в фильтр.
    
    Args:
        data (Dict[str, Any]): Исходный словарь данных.

    Returns:
        Tuple[List, List, List]: Кортеж, содержащий:
            - control: список управления (dt, акселерометр, гироскоп)
            - obs_gnss: измерения GNSS (время, координаты)
            - obs_lidar: измерения LiDAR (время, координаты с учетом калибровки)
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
        z = C_LIDAR @ lidar.data[i] + T_LIDAR
        obs_lidar.append((float(lidar.t[i]), z.astype(float)))

    return control, obs_gnss, obs_lidar


# =========================
# ESKF
# =========================
class ESKF:
    """
    Класс, реализующий фильтр Калмана с состояниями ошибки (Error-State Kalman Filter) 
    для объединения данных IMU, GNSS и LiDAR.
    """
    def __init__(self, p0: np.ndarray, v0: np.ndarray, q0: np.ndarray) -> None:
        self.p: np.ndarray = p0.astype(float).copy()
        self.v: np.ndarray = v0.astype(float).copy()
        self.q: np.ndarray = q0.astype(float).copy()
        self.P: np.ndarray = np.eye(9, dtype=float)

    def predict(self, f: np.ndarray, w: np.ndarray, dt: float) -> None:
        """
        Шаг предсказания (Prediction step) ESKF на основе данных IMU.
        
        Args:
            f (np.ndarray): Измерения акселерометра (3,).
            w (np.ndarray): Измерения гироскопа (3,).
            dt (float): Изменение времени (дельта t) с прошлого шага.
        """
        if dt <= 0:
            return

        # Получаем матрицу поворота из кватерниона
        R = quat_to_R(self.q)
        acc_world = R @ f + G

        # Изменение положения (условия задачи)
        self.p += self.v * dt + 0.5 * acc_world * dt**2
        # Изменение скорости (условия задачи)
        self.v += acc_world * dt

        # Умножение кватернионов
        dq = quat_exp(w * dt)
        # Обновление ориентации
        self.q = quat_mul(self.q, dq)
        # Нормализация кватерниона
        self.q /= np.linalg.norm(self.q)

        # Матрица Якоби системы (F)
        #     | I_3     dt*I_3             0         |
        # F = |  0       I_3       -skew(R @ f) * dt |
        #     |  0        0               I_3        |
        F = np.eye(9)
        # Связь ошибки позиции с ошибкой скорости
        F[0:3, 3:6] = np.eye(3) * dt
        # Связь ошибки скорости с ошибкой ориентации
        F[3:6, 6:9] = -skew(R @ f) * dt

        # Матрица Якоби по шуму (L)
        #     |  0    0  |
        # F = | I_3   0  |
        #     |  0   I_3 |
        L = np.zeros((9, 6))
        # Шум акселерометра влияет на скорость
        L[3:6, 0:3] = np.eye(3)
        # Шум гироскопа влияет на ориентацию
        L[6:9, 3:6] = np.eye(3)

        # Масштабирование описано в условии
        Q = np.zeros((6, 6))
        # Дисперсия шума акселерометра
        Q[0:3, 0:3] = SIGMA_ACC**2 * np.eye(3)
        # Дисперсия шума гироскопа
        Q[3:6, 3:6] = SIGMA_GYRO**2 * np.eye(3)
        # Масштабирование шума (дискретизация)
        Q *= dt**2

        # Матрица Якоби по шуму "переносит" шум из пространства датчиков в пространство состояния (8)
        self.P = F @ self.P @ F.T + L @ Q @ L.T

    def update(self, z: np.ndarray, R_meas: np.ndarray) -> None:
        """
        Шаг обновления (Update step) ESKF для одного типа измерений по позиции.
        
        Args:
            z (np.ndarray): Вектор измерений позиции (3,).
            R_meas (np.ndarray): Матрица ковариации шума измерений формы (3, 3).
        """
        # ДЛя анализа выделяем только компоненты координат
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)

        # Вычисление невязки
        y = z - self.p
        # Вычисление ковариации невязки
        S = H @ self.P @ H.T + R_meas
        # Вычисление коэффициента Калмана
        K = self.P @ H.T @ np.linalg.inv(S)

        # Оценка вектора ошибки
        # dx[0:3] - ошибка в позиции
        # dx[3:6] - ошибка в скорости
        # dx[6:9] - ошибка в ориентации
        dx = K @ y

        # Добавляем оценку ошибки к состоянию
        self.p += dx[0:3]
        self.v += dx[3:6]

        # Добавление нормализованного вектора, полученного из кватерниона
        dtheta = dx[6:9]
        self.q = quat_mul(quat_exp(dtheta), self.q)
        self.q /= np.linalg.norm(self.q)

        # Обновление ковариации состояния
        self.P = (np.eye(9) - K @ H) @ self.P

    def update_combined(
            self, z_gnss: np.ndarray, z_lidar: np.ndarray, R_gnss: np.ndarray, R_lidar: np.ndarray
    ) -> None:
        """
        Комбинированный шаг обновления ESKF, использующий одновременно измерения от GNSS и LiDAR.
        
        Args:
            z_gnss (np.ndarray): Измерения позиции от GNSS (3,).
            z_lidar (np.ndarray): Измерения позиции от LiDAR (3,).
            R_gnss (np.ndarray): Матрица ковариации GNSS (3, 3).
            R_lidar (np.ndarray): Матрица ковариации LiDAR (3, 3).
        """
        # Вектор измерений: [gnss_x, gnss_y, gnss_z, lidar_x, lidar_y, lidar_z]
        z = np.concatenate((z_gnss, z_lidar))

        # Ожидаемое измерение: текущая позиция дублируется
        h = np.concatenate((self.p, self.p))

        # Невязка
        y = z - h

        # Матрица наблюдения H (6x9)
        # Производные по позиции для обеих частей равны I
        # H =
        # [
        #  [I, 0, 0],
        #  [I, 0, 0]
        # ]
        H = np.zeros((6, 9))
        H[0:3, 0:3] = np.eye(3)  # Для GNSS
        H[3:6, 0:3] = np.eye(3)  # Для LiDAR

        # Матрица ковариации измерений R (6x6)
        # R = [
        #   [R_gnss, 0],
        #   [0, R_lidar]
        # ]
        R_meas = np.zeros((6, 6))
        R_meas[0:3, 0:3] = R_gnss
        R_meas[3:6, 3:6] = R_lidar

        # Стандартные уравнения Калмана
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y

        # Внедрение ошибки в номинальное состояние
        self.p += dx[0:3]
        self.v += dx[3:6]

        dtheta = dx[6:9]
        self.q = quat_mul(quat_exp(dtheta), self.q)
        self.q /= np.linalg.norm(self.q)

        # Обновление ковариации
        self.P = (np.eye(9) - K @ H) @ self.P


# =========================
# Filter run
# =========================
def run_filter(data: Dict[str, Any]) -> np.ndarray:
    """
    Запускает процесс фильтрации (ESKF) по заданным данным.
    
    Args:
        data (Dict[str, Any]): Подготовленный словарь с данными (IMU, GNSS, LiDAR, GT).

    Returns:
        np.ndarray: Массив оцененных позиций на каждом шаге (N, 3).
    """
    control, obs_gnss, obs_lidar = prepare_sequences(data)

    eskf = ESKF(
        p0=data["gt"].p[0], v0=data["gt"].v[0], q0=np.array([1.0, 0.0, 0.0, 0.0])
    )

    traj = []

    gnss_idx = 0
    lidar_idx = 0
    EPS = 0.05  # Допуск для сравнения времени измерений
    count_combine, count_gnss, count_lidar = 0, 0, 0

    R_gnss = SIGMA_GNSS**2 * np.eye(3)
    R_lidar = SIGMA_LIDAR**2 * np.eye(3)

    for i in range(1, len(control)):

        t, f, w = control[i]
        prev_t = control[i - 1][0]

        dt = t - prev_t

        # Шаг предсказания
        eskf.predict(f, w, dt)

        # Проверяем наличие измерений
        has_gnss = gnss_idx < len(obs_gnss) and abs(obs_gnss[gnss_idx][0] - t) < EPS
        has_lidar = lidar_idx < len(obs_lidar) and abs(obs_lidar[lidar_idx][0] - t) < EPS

        if has_gnss and has_lidar:
            # Логика объединения (combined update)
            eskf.update_combined(
                obs_gnss[gnss_idx][1], obs_lidar[lidar_idx][1], R_gnss, R_lidar
            )
            gnss_idx += 1
            lidar_idx += 1

            count_combine += 1
        elif has_gnss:
            # Обычное обновление GNSS
            eskf.update(obs_gnss[gnss_idx][1], R_gnss)
            gnss_idx += 1

            count_gnss += 1
        elif has_lidar:
            # Обычное обновление LiDAR
            eskf.update(obs_lidar[lidar_idx][1], R_lidar)
            lidar_idx += 1

            count_lidar += 1

        traj.append(eskf.p.copy())

    print(f"count_combine: {count_combine}, count_gnss: {count_gnss}, count_lidar: {count_lidar}")
    return np.array(traj)


# =========================
# Visualization
# =========================
def plot_trajectory(traj: np.ndarray, gt: Any) -> None:
    """
    Отрисовывает 3D-график оцененной траектории и истинной (Ground Truth).
    Args:
        traj (np.ndarray): Оцененная траектория формы (N, 3).
        gt (Any): Объект Ground Truth, содержащий истинную позицию.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="ESKF Combined")
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label="Ground Truth")

    ax.legend()
    ax.set_title("Trajectory (Combined Update Logic)")

    plt.show()


# =========================
# Main
# =========================
def main() -> None:
    data = load_data("data_files/data/data.pkl")
    traj = run_filter(data)
    plot_trajectory(traj, data["gt"])


if __name__ == "__main__":
    main()
