import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


# =========================
# Constants
# =========================

G = np.array([0.0, 0.0, -9.81])

SIGMA_ACC = 0.02
SIGMA_GYRO = 0.02

SIGMA_GNSS = 0.1
SIGMA_LIDAR = 0.1

C_LIDAR = np.array([
    [0.99376, -0.09722, 0.05466],
    [0.09971, 0.99401, -0.04475],
    [-0.04998, 0.04992, 0.9975]
], dtype=float)

T_LIDAR = np.array([0.5, 0.1, 0.5], dtype=float)


# =========================
# Math utils
# =========================

def skew(a):
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0]
    ])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_exp(theta):
    angle = np.linalg.norm(theta)

    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = theta / angle
    return np.hstack([
        np.cos(angle / 2.0),
        axis * np.sin(angle / 2.0)
    ])


def quat_to_R(q):
    q0, q1, q2, q3 = q

    return np.array([
        [2*q0*q0 - 1 + 2*q1*q1, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 2*q0*q0 - 1 + 2*q2*q2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 2*q0*q0 - 1 + 2*q3*q3]
    ], dtype=float)


# =========================
# Data loading
# =========================

def load_data(path):
    data_path = Path(path)

    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    data_files_dir = data_path.parent.parent
    if str(data_files_dir) not in sys.path:
        sys.path.insert(0, str(data_files_dir))

    import data.data  # noqa
    import data.utils  # noqa

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


# =========================
# Prepare sequences
# =========================

def prepare_sequences(data):

    imu_f = data['imu_f']
    imu_w = data['imu_w']
    gnss = data['gnss']
    lidar = data['lidar']

    control = []
    for i in range(1, len(imu_f.data)):
        control.append((
            float(imu_f.t[i]),
            imu_f.data[i - 1].astype(float),
            imu_w.data[i - 1].astype(float)
        ))

    obs_gnss = []
    for i in range(len(gnss.t)):
        obs_gnss.append((
            float(gnss.t[i]),
            gnss.data[i].astype(float)
        ))

    obs_lidar = []
    for i in range(len(lidar.t)):
        z = C_LIDAR @ lidar.data[i] + T_LIDAR
        obs_lidar.append((
            float(lidar.t[i]),
            z.astype(float)
        ))

    return control, obs_gnss, obs_lidar


# =========================
# ESKF
# =========================

class ESKF:

    def __init__(self, p0, v0, q0):
        self.p = p0.astype(float).copy()
        self.v = v0.astype(float).copy()
        self.q = q0.astype(float).copy()

        self.P = np.eye(9, dtype=float)

    def predict(self, f, w, dt):

        if dt <= 0:
            return

        R = quat_to_R(self.q)
        acc_world = R @ f + G

        # nominal
        self.p += self.v * dt + 0.5 * acc_world * dt**2
        self.v += acc_world * dt

        dq = quat_exp(w * dt)
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # error dynamics
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

        self.P = F @ self.P @ F.T + L @ Q @ L.T

    def update(self, z, R_meas):

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

def run_filter(data):

    control, obs_gnss, obs_lidar = prepare_sequences(data)

    eskf = ESKF(
        p0=data['gt'].p[0],
        v0=data['gt'].v[0],
        q0=np.array([1.0, 0.0, 0.0, 0.0])
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

        # GNSS
        while (
            gnss_idx < len(obs_gnss)
            and abs(obs_gnss[gnss_idx][0] - t) < EPS
        ):
            eskf.update(
                obs_gnss[gnss_idx][1],
                SIGMA_GNSS**2 * np.eye(3)
            )
            gnss_idx += 1

        # LiDAR
        while (
            lidar_idx < len(obs_lidar)
            and abs(obs_lidar[lidar_idx][0] - t) < EPS
        ):
            eskf.update(
                obs_lidar[lidar_idx][1],
                SIGMA_LIDAR**2 * np.eye(3)
            )
            lidar_idx += 1

        traj.append(eskf.p.copy())

    return np.array(traj)


# =========================
# Visualization
# =========================

def plot_trajectory(traj, gt):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="ESKF")
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label="Ground Truth")

    ax.legend()
    ax.set_title("Trajectory")

    plt.show()


# =========================
# Main
# =========================

def main():

    data = load_data("data_files/data/data.pkl")

    traj = run_filter(data)

    plot_trajectory(traj, data['gt'])


if __name__ == "__main__":
    main()