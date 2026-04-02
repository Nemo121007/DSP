import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/ubuntu/PycharmProjects/DSP/lab1/data_files")

# =========================
# Constants
# =========================

G = np.array([0.0, 0.0, -9.81])

SIGMA_ACC = 0.02
SIGMA_GYRO = 0.02

SIGMA_GNSS = 0.1
SIGMA_LIDAR = 0.1

EPS_SYNC = 0.05

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
# Utils
# =========================

def skew(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_exp(theta):
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.array([1, 0, 0, 0])
    axis = theta / angle
    return np.hstack([np.cos(angle / 2), axis * np.sin(angle / 2)])


def quat_to_R(q):
    q0, q1, q2, q3 = q
    return np.array([
        [2*q0*q0-1+2*q1*q1, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
        [2*q1*q2+2*q0*q3, 2*q0*q0-1+2*q2*q2, 2*q2*q3-2*q0*q1],
        [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*q0*q0-1+2*q3*q3]
    ])

# =========================
# Data adapters
# =========================

def get_time(obj):
    return np.asarray(obj.t if hasattr(obj, "t") else obj._t)


def get_data(obj):
    return np.asarray(obj.data, dtype=float)


def nearest_idx(t_ref, t):
    return np.argmin(np.abs(t_ref - t))


def align(t_ref, t_meas, z_meas):
    aligned = [[] for _ in range(len(t_ref))]
    for t, z in zip(t_meas, z_meas):
        i = nearest_idx(t_ref, t)
        if abs(t_ref[i] - t) < EPS_SYNC:
            aligned[i].append(z)
    return aligned

# =========================
# Prepare
# =========================

def prepare(data):
    imu_f = data["imu_f"]
    imu_w = data["imu_w"]
    gnss = data["gnss"]
    lidar = data["lidar"]

    t = get_time(imu_f)
    f = get_data(imu_f)
    w = get_data(imu_w)

    gnss_aligned = align(t, get_time(gnss), get_data(gnss))

    lidar_z = get_data(lidar)
    lidar_z = (C_LIDAR @ lidar_z.T).T + T_LIDAR
    lidar_aligned = align(t, get_time(lidar), lidar_z)

    return t, f, w, gnss_aligned, lidar_aligned

# =========================
# ESKF
# =========================

class ESKF:
    def __init__(self, p, v, q):
        self.p = p.copy()
        self.v = v.copy()
        self.q = q.copy()
        self.P = np.eye(9)

    def predict(self, f, w, dt):
        R = quat_to_R(self.q)
        a = R @ f + G

        self.p += self.v * dt + 0.5 * a * dt**2
        self.v += a * dt

        dq = quat_exp(w * dt)
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -skew(R @ f) * dt

        Q = np.eye(6)
        Q[:3, :3] *= SIGMA_ACC**2
        Q[3:, 3:] *= SIGMA_GYRO**2
        Q *= dt**2

        L = np.zeros((9, 6))
        L[3:6, :3] = np.eye(3)
        L[6:9, 3:] = np.eye(3)

        self.P = F @ self.P @ F.T + L @ Q @ L.T

    def update(self, z, H, R):
        x = np.hstack([self.p, self.v, np.zeros(3)])
        y = z - H @ x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y

        self.p += dx[:3]
        self.v += dx[3:6]

        dtheta = dx[6:9]
        self.q = quat_mul(quat_exp(dtheta), self.q)
        self.q /= np.linalg.norm(self.q)

        self.P = (np.eye(9) - K @ H) @ self.P

# =========================
# Run
# =========================


def run_filter(data):
    t, f, w, gnss_a, lidar_a = prepare(data)

    eskf = ESKF(
        data["gt"].p[0],
        data["gt"].v[0],
        np.array([1.0, 0.0, 0.0, 0.0])
    )

    traj = []

    for i in range(1, len(t) - 1):
        dt = t[i] - t[i - 1]

        eskf.predict(f[i - 1], w[i - 1], dt)

        gnss_list = gnss_a[i]
        lidar_list = lidar_a[i]

        has_gnss = len(gnss_list) > 0
        has_lidar = len(lidar_list) > 0

        # ===== JOINT UPDATE (6D) =====
        if has_gnss and has_lidar:
            z = np.hstack([gnss_list[0], lidar_list[0]])

            H = np.zeros((6, 9))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 0:3] = np.eye(3)

            R = np.zeros((6, 6))
            R[0:3, 0:3] = SIGMA_GNSS**2 * np.eye(3)
            R[3:6, 3:6] = SIGMA_LIDAR**2 * np.eye(3)

            eskf.update(z, H, R)

        # ===== GNSS ONLY =====
        elif has_gnss:
            z = gnss_list[0]

            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)

            R = SIGMA_GNSS**2 * np.eye(3)

            eskf.update(z, H, R)

        # ===== LIDAR ONLY =====
        elif has_lidar:
            z = lidar_list[0]

            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)

            R = SIGMA_LIDAR**2 * np.eye(3)

            eskf.update(z, H, R)

        traj.append(eskf.p.copy())

    return np.array(traj)

# =========================
# Main
# =========================


def main():
    with open("lab1/data_files/data/data.pkl", "rb") as f:
        data = pickle.load(f)

    traj = run_filter(data)
    gt = data["gt"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="ESKF")
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label="GT")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()