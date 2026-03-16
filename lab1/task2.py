from typing import Dict

import numpy as np
from pathlib import Path


# =========================
# Utils
# =========================

def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


# =========================
# Motion model
# =========================

def motion_model(state, u):
    x, y, theta = state
    dr1, dt, dr2 = u

    x_new = x + dt * np.cos(theta + dr1)
    y_new = y + dt * np.sin(theta + dr1)
    theta_new = theta + dr1 + dr2

    theta_new = wrap_to_pi(theta_new)

    return np.array([x_new, y_new, theta_new])


def jacobian_motion(state, u):
    _, _, theta = state
    dr1, dt, _ = u

    angle = theta + dr1

    return np.array([
        [1, 0, -dt * np.sin(angle)],
        [0, 1,  dt * np.cos(angle)],
        [0, 0, 1]
    ])


# =========================
# Measurement model
# =========================

def measurement_model(state, landmark):
    x, y, _ = state
    lx, ly = landmark

    dx = x - lx
    dy = y - ly

    return np.sqrt(dx**2 + dy**2)


def jacobian_measurement(state, landmark):
    x, y, _ = state
    lx, ly = landmark

    dx = x - lx
    dy = y - ly

    r = np.sqrt(dx**2 + dy**2) + 1e-9  # защита от деления на 0

    return np.array([[dx / r, dy / r, 0]])


# =========================
# Noise
# =========================

Q = np.diag([0.2, 0.2, 0.2])
R = 0.2


# =========================
# EKF
# =========================

def ekf_step(state, P, u, measurements, landmarks):

    # ---- Prediction ----
    F = jacobian_motion(state, u)
    state_pred = motion_model(state, u)

    P_pred = F @ P @ F.T + Q

    state = state_pred
    P = P_pred

    # ---- Update (по каждому ориентиру) ----
    for lm_id, z in measurements:

        landmark = landmarks[lm_id]

        z_pred = measurement_model(state, landmark)
        H = jacobian_measurement(state, landmark)

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        innovation = z - z_pred

        state = state + (K.flatten() * innovation)
        state[2] = wrap_to_pi(state[2])

        P = (np.eye(3) - K @ H) @ P

    return state, P


# =========================
# UKF
# =========================

def generate_sigma_points(x, P, alpha=0.5, beta=2, kappa=0):
    n = len(x)
    lam = alpha**2 * (n + kappa) - n

    scale = n + lam
    if scale <= 0:
        scale = 1e-3

    eps = 1e-6
    P = P + eps * np.eye(n)

    sqrt_P = np.linalg.cholesky(scale * P)

    sigma_points = [x]

    for i in range(n):
        sigma_points.append(x + sqrt_P[:, i])
        sigma_points.append(x - sqrt_P[:, i])

    sigma_points = np.array(sigma_points)

    Wm = np.full(2*n+1, 1/(2*scale))
    Wc = np.full(2*n+1, 1/(2*scale))

    Wm[0] = lam/scale
    Wc[0] = lam/scale + (1 - alpha**2 + beta)

    return sigma_points, Wm, Wc


def ukf_predict(state, P, u):

    sigma_points, Wm, Wc = generate_sigma_points(state, P)

    propagated = np.array([motion_model(sp, u) for sp in sigma_points])

    # mean
    x_pred = np.sum(Wm[:, None] * propagated, axis=0)
    x_pred[2] = wrap_to_pi(x_pred[2])

    # covariance
    P_pred = np.zeros((3, 3))
    for i in range(len(sigma_points)):
        diff = propagated[i] - x_pred
        diff[2] = wrap_to_pi(diff[2])
        P_pred += Wc[i] * np.outer(diff, diff)

    P_pred += Q

    return x_pred, P_pred, propagated, Wm, Wc


def ukf_update(state, P, sigma_points, x_pred, Wm, Wc, measurement, landmark):

    Z = np.array([measurement_model(sp, landmark) for sp in sigma_points])
    z_pred = np.sum(Wm * Z)

    S = 0
    Pxz = np.zeros((3,))

    for i in range(len(Z)):
        dz = Z[i] - z_pred
        dx = sigma_points[i] - x_pred
        dx[2] = wrap_to_pi(dx[2])

        S += Wc[i] * dz * dz
        Pxz += Wc[i] * dx * dz

    S = max(S + R, 1e-9)

    K = Pxz / S

    innovation = measurement - z_pred

    state = state + K * innovation
    state[2] = wrap_to_pi(state[2])

    P = P - np.outer(K, K) * S
    P = 0.5 * (P + P.T)

    return state, P


def ukf_step(state, P, u, measurements, landmarks):

    state, P, _, _, _ = ukf_predict(state, P, u)

    for lm_id, z in measurements:
        landmark = landmarks[lm_id]

        sigma_points, Wm, Wc = generate_sigma_points(state, P)

        state, P = ukf_update(state, P, sigma_points, state, Wm, Wc, z, landmark)

    return state, P

# =========================
# Main
# =========================

def run_filter(odometry_data, measurements_seq, landmarks, use_ukf=False):

    state = np.array([0.0, 0.0, 0.0])
    P = np.eye(3)

    trajectory = []

    for u, measurements in zip(odometry_data, measurements_seq):

        if use_ukf:
            state, P = ukf_step(state, P, u, measurements, landmarks)
        else:
            state, P = ekf_step(state, P, u, measurements, landmarks)

        trajectory.append(state.copy())

    return np.array(trajectory)


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

import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ---- пути к файлам ----
    path_landmarks = Path(__file__).parent / "data_files" / "landmarks.dat"
    path_data = Path(__file__).parent / "data_files" / "sensor_data_ekf.dat"

    # ---- загрузка данных ----
    landmarks = load_landmarks(path_landmarks)
    odometry_data, measurements_seq = load_sensor_data(path_data, landmarks)

    print(f"Загружено шагов: {len(odometry_data)}")
    print(f"Ориентиров: {len(landmarks)}")

    # ---- запуск EKF ----
    traj_ekf = run_filter(
        odometry_data,
        measurements_seq,
        landmarks,
        use_ukf=False
    )

    # ---- запуск UKF ----
    traj_ukf = run_filter(
        odometry_data,
        measurements_seq,
        landmarks,
        use_ukf=True
    )

    # ---- визуализация ----
    plt.figure(figsize=(10, 8))

    # траектории
    plt.plot(traj_ekf[:, 0], traj_ekf[:, 1], label="EKF", linewidth=2)
    plt.plot(traj_ukf[:, 0], traj_ukf[:, 1], label="UKF", linestyle="--")

    # ориентиры
    lm_x = [lm[0] for lm in landmarks.values()]
    lm_y = [lm[1] for lm in landmarks.values()]

    plt.scatter(lm_x, lm_y, marker='x', label="Landmarks")

    # подписи ориентиров
    for lm_id, (lx, ly) in landmarks.items():
        plt.text(lx + 0.1, ly + 0.1, str(lm_id), fontsize=8)

    # оформление
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Robot Localization (EKF vs UKF)")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    plt.show()