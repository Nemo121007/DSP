import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, TypeAlias


StateVec: TypeAlias = np.ndarray  # shape: (3,)
OdomVec: TypeAlias = np.ndarray   # shape: (3,)
Mat: TypeAlias = np.ndarray       # generic matrix
Landmarks: TypeAlias = np.ndarray  # shape: (N, 2)
Measurements: TypeAlias = np.ndarray  # shape: (N, 2) [range, bearing]


# Вспомогательные функции
def normalize_angle(angle: float) -> float:
    """Приведение угла к диапазону [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _symmetrize(A: Mat) -> Mat:
    """Приведение матрицы к симметричной форме"""
    return 0.5 * (A + A.T)


def _ensure_spd(P: Mat, min_eig: float = 1e-12) -> Mat:
    """Гарантирует симметричность и PSD/SPD (через подрезание собственных значений)."""
    # Обеспечение симметричности
    P = _symmetrize(P)
    # Вычисление собственных значений и векторов
    w, V = np.linalg.eigh(P)
    # Замена отрицательных или слишком маленьких собственных значений на min_eig
    w = np.maximum(w, min_eig)
    # Восстановление матрицы с гарантией SPD (собственное разложение матрицы)
    P = V @ np.diag(w) @ V.T
    return _symmetrize(P)


def motion_model(state: StateVec, odometry: OdomVec) -> StateVec:
    """
    Модель движения для робота.
    Args:
        state: [x, y,theta] - текущее состояние робота
        odometry: [delta_r1, delta_trans, delta_r2] - одометрические измерения
    Returns:
        new_state: [x_new, y_new, theta_new] - новое состояние после применения модели движения
    """
    x, y, theta = state
    delta_r1, delta_trans, delta_r2 = odometry

    x_new = x + delta_trans * np.cos(theta + delta_r1)
    y_new = y + delta_trans * np.sin(theta + delta_r1)
    theta_new = theta + delta_r1 + delta_r2

    return np.array([x_new, y_new, normalize_angle(theta_new)])


def measurement_model(state: StateVec, landmarks: Iterable[np.ndarray]) -> Measurements:
    """
    Модель измерения, которая предсказывает расстояние и пеленг до каждого ориентира.
    Args:
        state: [x, y, theta] - текущее состояние робота
        landmarks: массив (N, 2) с координатами ориентиров [[lm1_x, lm1_y], [lm2_x, lm2_y], ...]
    Returns:
        predictions: массив (N, 2) с предсказанными измерениями [[dist1, bearing1], [dist2, bearing2], ...]
         - dist: расстояние до ориентира
         - bearing: угол на ориентир в локальной системе координат робота (нормализованный в [-pi, pi])
    """
    x, y, theta = state
    predictions = []
    for lm in landmarks:
        dx = lm[0] - x
        dy = lm[1] - y
        dist = np.sqrt(dx**2 + dy**2)
        # Пеленг: угол на ориентир в локальной системе координат робота
        angle = np.arctan2(dy, dx) - theta
        predictions.append([dist, normalize_angle(angle)])
    return np.array(predictions, dtype=float)


def load_landmarks_dat(path: str | Path) -> dict[int, np.ndarray]:
    """
    Метод загрузки ориентиров из landmarks.dat. Предполагается, что файл имеет строки вида: ind, x, y
     - ind: целочисленный идентификатор ориентира
     - x, y: координаты ориентира в мировой системе координат
    Args:
        path: путь к файлу landmarks.dat
    Returns:
        Словарь, где ключ - идентификатор ориентира (int), а значение - массив [x, y] с координатами ориентира.
    """
    path = Path(path)
    lm_by_id: dict[int, np.ndarray] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lm_id = int(parts[0])
            lm_by_id[lm_id] = np.array([float(parts[1]), float(parts[2])], dtype=float)
    if not lm_by_id:
        raise ValueError(f"Не удалось прочитать ориентиры из {path}")
    return lm_by_id


def load_sensor_data_ekf_dat(
        path: str | Path,
        landmarks_by_id: dict[int, np.ndarray],
) -> tuple[list[OdomVec], list[list[np.ndarray]], list[Measurements]]:
    """
    Метод загрузки данных из sensor_data_ekf.dat. Предполагается, что файл содержит строки двух типов:
     - ODOMETRY delta_r1 delta_trans delta_r2
        - delta_r1: изменение угла переднего колеса (рад)
        - delta_trans: изменение расстояния (м)
        - delta_r2: изменение угла заднего колеса (рад)
     - SENSOR lm_id range bearing
        - lm_id: идентификатор наблюдаемого ориентира (целое число)
        - range: измеренное расстояние до ориентира (м)
        - bearing: измеренный угол на ориентир в локальной системе координат робота (рад)
    Args:
        path: путь к файлу sensor_data_ekf.dat
        landmarks_by_id: идентификаторы ориентиров и их координаты, загруженные из landmarks.dat.
        Ключ - целочисленный идентификатор, значение - массив [x, y].
    Returns:
        Кортеж списка одометрических измерений, списка наблюдаемых ориентиров и списка измерений для каждого шага:
            - odometry_data: список массивов [delta_r1, delta_trans, delta_r2] для каждого шага
            - observed_landmarks: список списков массивов [x, y] с координатами наблюдаемых ориентиров для каждого шага
            - measurements: список массивов (N, 2) с измерениями [range, bearing] для каждого шага,
            где N - количество наблюдаемых ориентиров на этом шаге
    """
    path = Path(path)
    odometry_data: list[OdomVec] = []
    observed_landmarks: list[list[np.ndarray]] = []
    measurements: list[Measurements] = []

    current_lms: list[np.ndarray] | None = None
    current_meas: list[list[float]] | None = None

    def flush_current() -> None:
        nonlocal current_lms, current_meas
        if current_lms is None or current_meas is None:
            return
        observed_landmarks.append(current_lms)
        measurements.append(np.asarray(current_meas, dtype=float))
        current_lms = None
        current_meas = None

    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tag = parts[0].upper()

            if tag == 'ODOMETRY':
                if odometry_data:
                    flush_current()
                if len(parts) < 4:
                    raise ValueError(f"Некорректная строка ODOMETRY: {raw!r}")
                odom = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                odometry_data.append(odom)
                current_lms = []
                current_meas = []

            elif tag == 'SENSOR':
                if current_lms is None:
                    continue
                if len(parts) < 4:
                    continue
                lm_id = int(parts[1])
                rng = float(parts[2])
                bearing = float(parts[3])

                lm_xy = landmarks_by_id.get(lm_id)
                if lm_xy is None:
                    raise KeyError(f"Unknown landmark id {lm_id}")

                current_lms.append(lm_xy)
                current_meas.append([rng, bearing])

    flush_current()
    if len(observed_landmarks) != len(odometry_data):
        raise ValueError("Data mismatch")
    return odometry_data, observed_landmarks, measurements


# Расширенный фильтр Калмана (EKF)
class EKF:
    """EKF для локализации робота по данным range+bearing."""

    def __init__(self, x0: StateVec, P0: Mat, Q: Mat, R_dist: float, R_angle: float) -> None:
        """
        Args:
            x0: начальное состояние [x, y, theta]
            P0: матрица ковариации начального состояния (3x3)
            Q: матрица ковариации шума процесса (3x3)
            R_dist: размерность шума расстояния (предполагаемый, так как в условии не дан явно)
            R_angle: размерность шума угла (предполагаемый, так как в условии не дан явно)
        """
        self.m: StateVec = x0
        self.P: Mat = P0
        self.Q: Mat = Q
        self.R_dist: float = float(R_dist)
        self.R_angle: float = float(R_angle)

    def predict(self, odometry: OdomVec) -> None:
        """
        Метод, выполняющий этап предсказания EKF
        Args:
            odometry: массив [delta_r1, delta_trans, delta_r2] с одометрическими измерениями для текущего шага
        """
        self.m = motion_model(self.m, odometry)
        x, y, theta = self.m
        delta_r1, delta_trans, delta_r2 = odometry

        Fx = np.array([
            [1, 0, -delta_trans * np.sin(theta + delta_r1)],
            [0, 1,  delta_trans * np.cos(theta + delta_r1)],
            [0, 0, 1]
        ])
        self.P = Fx @ self.P @ Fx.T + self.Q

    def update(self, observed_landmarks: list[np.ndarray], measurements: Measurements) -> None:
        """
        Метод, выполняющий этап обновления EKF на основе наблюдаемых ориентиров и соответствующих измерений
        Args:
            observed_landmarks: список массивов [x, y] с координатами наблюдаемых ориентиров для текущего шага
            measurements: список массивов (N, 2) с измерениями [range, bearing]
            для каждого наблюдаемого ориентира на текущем шаге
        """
        if len(measurements) == 0:
            return
        n_landmarks = len(measurements)

        # Предсказанные измерения (N, 2)
        z_pred = measurement_model(self.m, observed_landmarks)

        # Вектор невязки (2N, 1)
        y_k = (measurements - z_pred).flatten()
        # Нормализация углов в невязке (каждый второй элемент)
        for i in range(1, len(y_k), 2):
            y_k[i] = normalize_angle(y_k[i])

        # Строим большой Якобиан H (2N x 3)
        H_all = np.zeros((2 * n_landmarks, 3))

        # Строим блочно-диагональную матрицу R (2N x 2N)
        R_all = np.zeros((2 * n_landmarks, 2 * n_landmarks))

        x, y, theta = self.m

        for i, lm in enumerate(observed_landmarks):
            mx, my = lm
            dx = mx - x
            dy = my - y
            q = dx**2 + dy**2
            dist = np.sqrt(q)

            # Якобиан для одного измерения (2x3)
            H_i = np.array([
                [-dx / dist, -dy / dist, 0],      # производные расстояния
                [dy / q,    -dx / q,    -1]      # производные пеленга
            ])

            H_all[2*i:2*i+2, :] = H_i

            # Шумы для одного измерения
            R_all[2*i, 2*i] = self.R_dist
            R_all[2*i+1, 2*i+1] = self.R_angle

        # Стандартные уравнения Калмана
        S_k = H_all @ self.P @ H_all.T + R_all
        K_k = self.P @ H_all.T @ np.linalg.inv(S_k)

        self.m = self.m + K_k @ y_k
        self.m[2] = normalize_angle(self.m[2])

        self.P = self.P - K_k @ S_k @ K_k.T


# Сигма-точечный фильтр Калмана (UKF)
class UKF:
    """UKF для локализации робота по данным range+bearing"""
    def __init__(
        self,
        x0: StateVec,
        P0: Mat,
        Q: Mat,
        R_dist: float,
        R_angle: float,
        alpha: float = 1e-3,
        beta: float = 2,
        kappa: float = 0,
    ) -> None:
        """
        Args:
            x0: начальное состояние [x, y, theta]
            P0: начальная ковариационная матрица (3x3)
            Q: матрица ковариации шума процесса (3x3)
            R_dist: размерность шума расстояния (предполагаемый, так как в условии не дан явно)
            R_angle: размерность шума угла (предполагаемый, так как в условии не дан явно)
            alpha: параметр, определяющий разброс сигма-точек (обычно маленькое значение, например 1e-3)
            beta: параметр, учитывающий априорные знания о распределении
                (для гауссовского распределения оптимально beta=2)
            kappa: параметр, который можно использовать для дополнительной настройки (обычно 0 или 3-n)
        """
        self.m: StateVec = x0
        self.P: Mat = P0
        self.Q: Mat = Q
        self.R_dist: float = float(R_dist)
        self.R_angle: float = float(R_angle)

        self.n: int = int(len(x0))
        self.lambda_: float = float(alpha**2 * (self.n + kappa) - self.n)

        self.W_m: np.ndarray = np.zeros(2 * self.n + 1)
        self.W_c: np.ndarray = np.zeros(2 * self.n + 1)

        self.W_m[0] = self.lambda_ / (self.n + self.lambda_)
        # В стандартной формуле UKF добавка именно +beta (а не +beta^2)
        self.W_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        self.W_m[1:] = 1 / (2 * (self.n + self.lambda_))
        self.W_c[1:] = 1 / (2 * (self.n + self.lambda_))

    def sigma_points(self, m: StateVec, P: Mat) -> Mat:
        """
        Метод, генерирующий сигма-точки для данного среднего вектора состояния и ковариационной матрицы.
        Args:
            m: вектор среднего состояния (3,)
            P: матрица, которую нужно гарантировать как SPD перед генерацией сигма-точек
        Returns:
            Матрица, содержащая сигма-точки в виде столбцов, размером (3, 2n+1)
        """
        n = self.n
        sigma_points = np.zeros((n, 2 * n + 1))

        P = _ensure_spd(P)

        # Адаптивный diagonal loading: увеличиваем jitter, пока не получится Cholesky.
        # Это защищает от вырождения/не-SPD из-за численной ошибки.
        jitter = 1e-12
        # Масштабируем по величине P, чтобы jitter был релевантен.
        scale = float(np.trace(P) / n) if np.isfinite(np.trace(P)) and np.trace(P) > 0 else 1.0
        jitter *= max(scale, 1.0)

        L = None
        for _ in range(10):
            try:
                L = np.linalg.cholesky(P + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0

        if L is None:
            # Последняя попытка: жёстко приводим к SPD и пробуем ещё раз.
            P2 = _ensure_spd(P + jitter * np.eye(n), min_eig=1e-9)
            L = np.linalg.cholesky(P2)

        sigma_points[:, 0] = m
        gamma = np.sqrt(n + self.lambda_)

        for i in range(n):
            col = L[:, i]
            sigma_points[:, i + 1] = m + gamma * col
            sigma_points[:, i + 1 + n] = m - gamma * col
        return sigma_points

    def predict(self, odometry: OdomVec) -> None:
        """
        Метод предсказания UKF, который генерирует сигма-точки,
        пропускает их через модель движения и затем восстанавливает предсказанное среднее и ковариацию.
        Args:
            odometry:

        Returns:

        """
        # Держим ковариацию в SPD перед генерацией сигма-точек
        self.P = _ensure_spd(self.P)

        X = self.sigma_points(self.m, self.P)
        X_pred = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_pred[:, i] = motion_model(X[:, i], odometry)

        self.m = np.sum(self.W_m * X_pred, axis=1)

        self.P = np.zeros((self.n, self.n))
        for i in range(X_pred.shape[1]):
            diff = X_pred[:, i] - self.m
            diff[2] = normalize_angle(diff[2])
            self.P += self.W_c[i] * np.outer(diff, diff)

        self.P += self.Q
        self.P = _ensure_spd(self.P)
        self.m[2] = normalize_angle(self.m[2])

    def update(self, observed_landmarks: list[np.ndarray], measurements: Measurements) -> None:
        """
        Метод обновления UKF
        Args:
            observed_landmarks: список массивов [x, y] с координатами наблюдаемых ориентиров для текущего шага
            measurements: список массивов (N, 2) с измерениями [range, bearing]
            для каждого наблюдаемого ориентира на текущем шаге
        """
        if len(measurements) == 0:
            return

        self.P = _ensure_spd(self.P)

        X = self.sigma_points(self.m, self.P)
        n_meas = len(measurements) * 2

        Y = np.zeros((n_meas, X.shape[1]))
        for i in range(X.shape[1]):
            preds = measurement_model(X[:, i], observed_landmarks)
            Y[:, i] = preds.flatten()

        mu = np.sum(self.W_m * Y, axis=1)

        S = np.zeros((n_meas, n_meas))
        C = np.zeros((self.n, n_meas))

        R_full = np.zeros((n_meas, n_meas))
        for k in range(len(measurements)):
            R_full[2 * k, 2 * k] = self.R_dist
            R_full[2 * k + 1, 2 * k + 1] = self.R_angle

        for i in range(X.shape[1]):
            y_diff = Y[:, i] - mu
            for k in range(len(measurements)):
                y_diff[2 * k + 1] = normalize_angle(y_diff[2 * k + 1])

            x_diff = X[:, i] - self.m
            x_diff[2] = normalize_angle(x_diff[2])

            S += self.W_c[i] * np.outer(y_diff, y_diff)
            C += self.W_c[i] * np.outer(x_diff, y_diff)

        S += R_full
        S = _symmetrize(S)

        # Вместо явного inverse — solve (численно устойчивее)
        K = (np.linalg.solve(S, C.T)).T  # C @ inv(S)

        meas_diff = measurements.flatten() - mu
        for k in range(len(measurements)):
            meas_diff[2 * k + 1] = normalize_angle(meas_diff[2 * k + 1])

        self.m = self.m + K @ meas_diff
        self.m[2] = normalize_angle(self.m[2])

        self.P = self.P - K @ S @ K.T
        self.P = _ensure_spd(self.P)


# Запуск
def run_filters() -> None:
    """Метод, загружающий данные, запускающий EKF и UKF и визуализирующий результаты."""
    # Параметры шума
    Q = np.diag([0.2, 0.2, 0.2])   # Шум процесса
    R_dist = 0.2                    # Шум расстояния (из условия)
    R_angle = 0.1                   # Шум угла (предполагаемый, так как в условии не дан явно)

    m0 = np.array([0.0, 0.0, 0.0])
    P0 = np.diag([0.1, 0.1, 0.1])

    base_dir = Path(__file__).resolve().parent
    landmarks_path = base_dir / 'data files' / 'landmarks.dat'
    sensor_path = base_dir / 'data files' / 'sensor_data_ekf.dat'

    try:
        landmarks_by_id = load_landmarks_dat(landmarks_path)
        odometry_data, observed_landmarks_seq, measurements_seq = load_sensor_data_ekf_dat(
            sensor_path, landmarks_by_id
        )
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    landmarks_xy = np.stack([landmarks_by_id[k] for k in sorted(landmarks_by_id.keys())], axis=0)

    ekf = EKF(m0.copy(), P0.copy(), Q, R_dist, R_angle)
    ukf = UKF(m0.copy(), P0.copy(), Q, R_dist, R_angle, alpha=0.001, beta=2, kappa=0)

    ekf_path: list[StateVec] = [ekf.m.copy()]
    ukf_path: list[StateVec] = [ukf.m.copy()]

    print("Запуск фильтрации...")
    for k in range(len(odometry_data)):
        u_k = odometry_data[k]
        lms_k = observed_landmarks_seq[k]
        z_k = measurements_seq[k] # Теперь это массив Nx2

        ekf.predict(u_k)
        ekf.update(lms_k, z_k)
        ekf_path.append(ekf.m.copy())

        ukf.predict(u_k)
        ukf.update(lms_k, z_k)
        ukf_path.append(ukf.m.copy())

    ekf_path = np.asarray(ekf_path)
    ukf_path = np.asarray(ukf_path)

    plt.figure(figsize=(10, 8))
    plt.plot(ekf_path[:, 0], ekf_path[:, 1], 'r--', label='EKF (Range+Bearing)')
    plt.plot(ukf_path[:, 0], ukf_path[:, 1], 'g-.', label='UKF (Range+Bearing)')
    plt.scatter(landmarks_xy[:, 0], landmarks_xy[:, 1], c='b', marker='^', s=100, label='Ориентиры')

    plt.legend()
    plt.grid(True)
    plt.title('Локализация с использованием расстояния и пеленга')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_filters()