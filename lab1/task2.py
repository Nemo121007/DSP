from typing import Dict
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# Utils
# =========================

def wrap_to_pi(theta) -> float:
    """
    Нормализует угол(ы) в диапазон [-pi, pi].

    """
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def make_spd(P: np.NDarray) -> np.ndarray:
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

def motion_model(state, u):
    """
    Модель движения в виде [dr1, dt, dr2]
        - dr1 - поворот перед движением
        - dt - пройденное расстояние
        - dr2 - поворот после завершения движения
    """
    x, y, theta = state
    dr1, dt, dr2 = u

    x_new = x + dt * np.cos(theta + dr1)
    y_new = y + dt * np.sin(theta + dr1)
    theta_new = wrap_to_pi(theta + dr1 + dr2)

    return np.array([x_new, y_new, theta_new])


def jacobian_motion(state, u):
    """
    Якобиан
    """
    _, _, theta = state
    dr1, dt, _ = u

    angle = theta + dr1

    return np.array([
        [1, 0, -dt * np.sin(angle)],
        [0, 1,  dt * np.cos(angle)],
        [0, 0, 1]
    ])


# =========================
# Measurement model (RANGE + BEARING)
# =========================

def measurement_model(state, landmark):
    x, y, theta = state
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    r = np.sqrt(dx**2 + dy**2)
    bearing = wrap_to_pi(np.arctan2(dy, dx) - theta)

    return np.array([r, bearing])


def jacobian_measurement(state, landmark):
    x, y, theta = state
    lx, ly = landmark

    dx = lx - x
    dy = ly - y

    q = dx**2 + dy**2
    r = np.sqrt(q) + 1e-9

    return np.array([
        [-dx / r, -dy / r, 0],
        [ dy / q, -dx / q, -1]
    ])


# =========================
# Noise
# =========================

Q = np.diag([0.2, 0.2, 0.1])
R = np.diag([0.2, 0.1])  # [range, bearing]


# =========================
# EKF
# =========================

def ekf_step(state, P, u, measurements, landmarks):

    # Prediction
    F = jacobian_motion(state, u)
    state = motion_model(state, u)
    P = F @ P @ F.T + Q

    # Update
    for lm_id, z in measurements:
        landmark = landmarks[lm_id]

        z_pred = measurement_model(state, landmark)
        H = jacobian_measurement(state, landmark)

        innovation = z - z_pred
        innovation[1] = wrap_to_pi(innovation[1])

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        state = state + K @ innovation
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

    P = make_spd(P)
    P = P + 1e-5 * np.eye(n)

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

    x_pred = np.sum(Wm[:, None] * propagated, axis=0)
    x_pred[2] = wrap_to_pi(x_pred[2])

    P_pred = np.zeros((3, 3))
    for i in range(len(propagated)):
        diff = propagated[i] - x_pred
        diff[2] = wrap_to_pi(diff[2])
        P_pred += Wc[i] * np.outer(diff, diff)

    P_pred += Q

    return x_pred, P_pred


def ukf_update(state, P, measurement, landmark):
    sigma_points, Wm, Wc = generate_sigma_points(state, P)

    Z = np.array([measurement_model(sp, landmark) for sp in sigma_points])

    z_pred = np.sum(Wm[:, None] * Z, axis=0)
    z_pred[1] = wrap_to_pi(z_pred[1])

    S = np.zeros((2, 2))
    Pxz = np.zeros((3, 2))

    for i in range(len(Z)):
        dz = Z[i] - z_pred
        dz[1] = wrap_to_pi(dz[1])

        dx = sigma_points[i] - state
        dx[2] = wrap_to_pi(dx[2])

        S += Wc[i] * np.outer(dz, dz)
        Pxz += Wc[i] * np.outer(dx, dz)

    # ✅ добавляем шум И jitter сразу
    S += R + 1e-6 * np.eye(2)

    K = Pxz @ np.linalg.inv(S)

    innovation = measurement - z_pred
    innovation[1] = wrap_to_pi(innovation[1])

    state = state + K @ innovation
    state[2] = wrap_to_pi(state[2])

    P = P - K @ S @ K.T
    P = make_spd(P)

    return state, P


def ukf_step(state, P, u, measurements, landmarks):
    state, P = ukf_predict(state, P, u)

    for lm_id, z in measurements:
        landmark = landmarks[lm_id]
        state, P = ukf_update(state, P, z, landmark)

    return state, P


# =========================
# Data loading
# =========================

def load_landmarks(path: Path) -> Dict[int, np.ndarray]:
    landmarks = {}
    with path.open('r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            landmarks[int(parts[0])] = np.array([float(parts[1]), float(parts[2])])
    return landmarks


def load_sensor_data(path: Path):
    odometry_data = []
    measurements_seq = []
    current_measurements = []

    with path.open('r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == 'ODOMETRY':
                if odometry_data:
                    measurements_seq.append(current_measurements)
                    current_measurements = []

                u = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                odometry_data.append(u)

            elif tag == 'SENSOR':
                lm_id = int(parts[1])
                r = float(parts[2])
                b = float(parts[3])

                current_measurements.append((lm_id, np.array([r, b])))

    if current_measurements:
        measurements_seq.append(current_measurements)

    return odometry_data, measurements_seq


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
    plt.scatter(lm_x, lm_y, marker='x', label="Landmarks")

    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("Range + Bearing: EKF vs UKF")

    plt.show()