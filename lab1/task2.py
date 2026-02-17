import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict


# Вспомогательные функции
def normalize_angle(angle: float) -> float:
    """Приведение угла к диапазону [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _symmetrize(A: np.ndarray) -> np.ndarray:
    """Приведение матрицы к симметричному виду для устранения ошибок округления."""
    return 0.5 * (A + A.T)


def _ensure_spd(P: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """
    Гарантирует, что матрица является симметричной положительно-определенной (SPD).
    Критически важно для устойчивости UKF (разложение Холецкого).
    """
    P = _symmetrize(P)
    w, V = np.linalg.eigh(P)
    w = np.maximum(w, min_eig)  # Обрезаем отрицательные собственные значения
    P = V @ np.diag(w) @ V.T
    return _symmetrize(P)


def motion_model(state: np.ndarray, odometry: np.ndarray) -> np.ndarray:
    """
    Модель движения робота (одометрия).
    Args:
        state: вектор состояния [x,y, theta]
        odometry: вектор управления
    Return:
        Новый вектор состояния после применения управления
    """
    x, y, theta = state
    delta_r1, delta_trans, delta_r2 = odometry

    x_new = x + delta_trans * np.cos(theta + delta_r1)
    y_new = y + delta_trans * np.sin(theta + delta_r1)
    theta_new = theta + delta_r1 + delta_r2

    return np.array([x_new, y_new, normalize_angle(theta_new)])


def measurement_model(state: np.ndarray, landmarks: List[np.ndarray]) -> np.ndarray:
    """
    Модель измерений (Range-Only).
    Возвращает вектор предсказанных расстояний до списка ориентиров.
    Args:
        state: вектор состояния [x,y, theta]
        landmarks: список координат ориентиров [[lm1_x, lm1_y], [lm2_x, lm2_y], ...]
    Return:
        Вектор предсказанных расстояний до ориентиров
    """
    x, y, _ = state
    predictions = []
    for lm in landmarks:
        dist = np.sqrt((lm[0] - x) ** 2 + (lm[1] - y) ** 2)
        predictions.append(dist)
    return np.array(predictions)


def load_landmarks(path: Path) -> Dict[int, np.ndarray]:
    """Загрузка координат ориентиров из файла."""
    landmarks = {}
    if not path.exists():
        raise FileNotFoundError(f"Файл ориентиров не найден: {path}")

    with path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and not line.startswith('#'):
                lm_id = int(parts[0])
                lm_x = float(parts[1])
                lm_y = float(parts[2])
                landmarks[lm_id] = np.array([lm_x, lm_y])
    return landmarks


def load_sensor_data(path: Path, landmarks: Dict[int, np.ndarray]):
    """
    Загрузка данных одометрии и измерений.
    Args:
        path: путь к файлу данных
        landmarks: словарь ориентиров для проверки наличия id в измерениях
    Returns:
        odometry_data: список векторов управления [r1, trans, r2]
        measurements_seq: список списков кортежей (id, distance)
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    odometry_data = []
    measurements_seq = []

    current_measurements = []

    with path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or line.startswith('#'):
                continue

            tag = parts[0].upper()

            if tag == 'ODOMETRY':
                # Сохраняем измерения для предыдущего шага (если есть)
                if current_measurements or odometry_data:
                    pass

                # Если это не первый блок, сохраняем накопленные замеры как завершение предыдущего шага
                # Начинаем новый шаг
                # Если уже был шаг, добавляем его измерения в список
                if odometry_data:
                    measurements_seq.append(current_measurements)
                    current_measurements = []

                u = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                odometry_data.append(u)

            elif tag == 'SENSOR':
                # Формат: SENSOR id range bearing (bearing игнорируем)
                lm_id = int(parts[1])
                rng = float(parts[2])
                # bearing = float(parts[3]) # Не используется в Range-Only
                current_measurements.append((lm_id, rng))

    if current_measurements:
        measurements_seq.append(current_measurements)

    # Выравнивание длин списков
    while len(measurements_seq) < len(odometry_data):
        measurements_seq.append([])

    return odometry_data, measurements_seq


class EKF:
    """Расширенный фильтр Калмана для локализации робота с Range-Only измерениями."""
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: float) -> None:
        """
        Args:
            x0: начальный вектор состояния [x, y, theta]
            P0: начальная ковариация состояния
            Q: ковариация процесса (одометрия)
            R: дисперсия измерений (Range-Only)
        """
        self.m = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R

    def predict(self, odometry: np.ndarray) -> None:
        """
        Метод предсказания состояния на основе одометрии.
        Args:
            odometry: вектор управления для предсказания нового состояния
        """
        self.m = motion_model(self.m, odometry)     # (31)

        # Якобиан F
        x, y, theta = self.m
        delta_r1, delta_trans, delta_r2 = odometry

        F = np.array([
            [1, 0, -delta_trans * np.sin(theta + delta_r1)],
            [0, 1, delta_trans * np.cos(theta + delta_r1)],
            [0, 0, 1]
        ])  # (25)

        self.P = F @ self.P @ F.T + self.Q
        self.P = _symmetrize(self.P)

    def update(self, observed_lms: List[np.ndarray], measurements: List[float]) -> None:
        """
        Метод, обновляющий оценку состояния на основе измерений до ориентиров.
        Args:
            observed_lms: список координат ориентиров, для которых есть измерения на этом шаге
            measurements: список измеренных расстояний до этих ориентиров
        """
        if len(measurements) == 0:
            return

        z_pred = measurement_model(self.m, observed_lms)
        n = len(measurements)

        # Якобиан H
        H = np.zeros((n, 3))
        x, y, _ = self.m

        for i, lm in enumerate(observed_lms):
            mx, my = lm
            dx = mx - x
            dy = my - y
            q = dx ** 2 + dy ** 2
            dist = np.sqrt(q)

            H[i, 0] = -dx / dist
            H[i, 1] = -dy / dist
            H[i, 2] = 0

        y_k = np.array(measurements) - z_pred

        # Ковариация корректировки
        R_mat = np.eye(n) * self.R
        S = H @ self.P @ H.T + R_mat        # (33)

        # Коэффициент Калмана (через solve для устойчивости)
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T        # (34)

        # Обновление
        self.m = self.m + K @ y_k       # (35)
        self.m[2] = normalize_angle(self.m[2])

        self.P = self.P - K @ S @ K.T   # (36)
        self.P = _symmetrize(self.P)


class UKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: float,
                 alpha: float = 1e-3, beta: float = 2, kappa: float = 0) -> None:
        """
        Args:
            x0: начальный вектор состояния [x, y, theta]
            P0: начальная ковариация состояния
            Q: ковариация процесса (одометрия)
            R: дисперсия измерений (Range-Only)
            alpha, beta, kappa: параметры для генерации сигма-точек
        """
        self.m = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R
        self.n = len(x0)

        self.lambda_ = alpha ** 2 * (self.n + kappa) - self.n

        # Веса
        self.W_m = np.zeros(2 * self.n + 1)
        self.W_c = np.zeros(2 * self.n + 1)

        self.W_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.W_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha ** 2 + beta)
        self.W_m[1:] = 1 / (2 * (self.n + self.lambda_))
        self.W_c[1:] = 1 / (2 * (self.n + self.lambda_))

    def sigma_points(self, m: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Метод генерации сигма-точек для UKF. Использует адаптивный jitter для обеспечения численной
        устойчивости при разложении Холецкого.
        Args:
            m: вектор состояния
            P: ковариация состояния
        Return:
            Матрица сигма-точек размером (n, 2n+1
        """
        n = self.n
        P = _ensure_spd(P)  # Гарантируем SPD
        L = np.linalg.cholesky(P )

        if L is None:
            P2 = _ensure_spd(P + 1e-6 * np.eye(n), min_eig=1e-9)
            L = np.linalg.cholesky(P2)

        sigmas = np.zeros((n, 2 * n + 1))
        sigmas[:, 0] = m            # (47-49)
        gamma = np.sqrt(n + self.lambda_)

        for i in range(n):
            sigmas[:, i + 1] = m + gamma * L[:, i]
            sigmas[:, i + 1 + n] = m - gamma * L[:, i]

        return sigmas

    def predict(self, odometry: np.ndarray) -> None:
        """
        Метод предсказания состояния на основе одометрии.
        Генерирует сигма-точки, пропускает их через модель движения и вычисляет новое среднее и ковариацию.
        Args:
            odometry: вектор управления для предсказания нового состояния
        """
        X = self.sigma_points(self.m, self.P)
        X_pred = np.zeros_like(X)

        for i in range(X.shape[1]):
            X_pred[:, i] = motion_model(X[:, i], odometry)      # (50)

        self.m = np.sum(self.W_m * X_pred, axis=1)      # (51)

        self.P = np.zeros((self.n, self.n))
        for i in range(X_pred.shape[1]):
            diff = X_pred[:, i] - self.m
            diff[2] = normalize_angle(diff[2])
            self.P += self.W_c[i] * np.outer(diff, diff)        # (52)

        self.P += self.Q
        self.P = _ensure_spd(self.P)
        self.m[2] = normalize_angle(self.m[2])

    def update(self, observed_lms: List[np.ndarray], measurements: List[float]) -> None:
        """Метод обновления оценки состояния на основе измерений до ориентиров.
        Args:
            observed_lms: список координат ориентиров, для которых есть измерения на этом шаге
            measurements: список измеренных расстояний до этих ориентиров
        """
        if len(measurements) == 0:
            return

        X = self.sigma_points(self.m, self.P)
        n_meas = len(measurements)

        Y = np.zeros((n_meas, X.shape[1]))
        for i in range(X.shape[1]):
            Y[:, i] = measurement_model(X[:, i], observed_lms)      # (56)

        mu = np.sum(self.W_m * Y, axis=1)           # (57)

        S = np.zeros((n_meas, n_meas))
        C = np.zeros((self.n, n_meas))

        for i in range(X.shape[1]):
            y_diff = Y[:, i] - mu
            x_diff = X[:, i] - self.m
            x_diff[2] = normalize_angle(x_diff[2])

            S += self.W_c[i] * np.outer(y_diff, y_diff)     # (58)
            C += self.W_c[i] * np.outer(x_diff, y_diff)     # (59)

        S += np.eye(n_meas) * self.R
        S = _symmetrize(S)

        K = np.linalg.solve(S, C.T).T       # (60)

        self.m = self.m + K @ (np.array(measurements) - mu)     # (61)
        self.m[2] = normalize_angle(self.m[2])

        self.P = self.P - K @ S @ K.T       # (32)
        self.P = _ensure_spd(self.P)


def run_filters():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data files'

    landmarks_path = data_dir / 'landmarks.dat'
    sensor_path = data_dir / 'sensor_data_ekf.dat'

    print(f"Загрузка ориентиров из: {landmarks_path}")
    print(f"Загрузка данных из: {sensor_path}")

    try:
        landmarks_dict = load_landmarks(landmarks_path)
        odometry_data, measurements_seq = load_sensor_data(sensor_path, landmarks_dict)
    except Exception as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return

    # Параметры шума из условия
    Q = np.diag([0.2, 0.2, 0.2])
    R = 0.2

    # Начальные условия
    m0 = np.array([0.0, 0.0, 0.0])
    P0 = np.diag([0.1, 0.1, 0.1])

    ekf = EKF(m0.copy(), P0.copy(), Q, R)
    ukf = UKF(m0.copy(), P0.copy(), Q, R, alpha=0.001, beta=2, kappa=0)

    ekf_path = [m0.copy()]
    ukf_path = [m0.copy()]

    print("Запуск фильтрации...")

    # Цикл по данным
    # Структура: одометрия предсказывает состояние на шаг t, измерения корректируют его.
    for k in range(len(odometry_data)):
        u = odometry_data[k]
        meas = measurements_seq[k]

        # Подготовка данных для Update
        # meas это список кортежей (id, range)
        obs_lms = []
        ranges = []
        for lm_id, r in meas:
            if lm_id in landmarks_dict:
                obs_lms.append(landmarks_dict[lm_id])
                ranges.append(r)

        # EKF
        ekf.predict(u)
        if len(ranges) > 0:
            ekf.update(obs_lms, ranges)
        ekf_path.append(ekf.m.copy())

        # UKF
        ukf.predict(u)
        if len(ranges) > 0:
            ukf.update(obs_lms, ranges)
        ukf_path.append(ukf.m.copy())

    ekf_path = np.array(ekf_path)
    ukf_path = np.array(ukf_path)

    # Визуализация ориентиров
    lm_coords = np.array(list(landmarks_dict.values()))

    plt.figure(figsize=(12, 8))
    plt.plot(ekf_path[:, 0], ekf_path[:, 1], 'r-', label='EKF (Range-Only)', linewidth=2, alpha=0.7)
    plt.plot(ukf_path[:, 0], ukf_path[:, 1], 'g--', label='UKF (Range-Only)', linewidth=2, alpha=0.7)

    if lm_coords.size > 0:
        plt.scatter(lm_coords[:, 0], lm_coords[:, 1], c='b', marker='^', s=100, label='Ориентиры')

    plt.title('Локализация мобильного робота (EKF vs UKF)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    run_filters()