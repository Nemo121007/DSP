from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Constants
# =========================

MU0 = 4 * np.pi * 1e-7
CONST = MU0 / (4 * np.pi)

h = 0.5
Q_SCALE = 1e5


# =========================
# Sensors (GRID)
# =========================

def generate_sensors(grid_size: int = 5, spacing: float = 1.0):
    sensors = []
    offset = (grid_size - 1) / 2

    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - offset) * spacing
            y = (j - offset) * spacing
            sensors.append(np.array([x, y, h]))

    return sensors


# =========================
# G(p)
# =========================

def compute_G(p, sensors):
    G = []
    px, py = p

    for r in sensors:
        dx = r[0] - px
        dy = r[1] - py
        dz = r[2]

        d2 = dx**2 + dy**2 + dz**2
        d3 = d2**1.5 + 1e-12

        g1 = CONST * (dy / d3)
        g2 = CONST * (-dx / d3)

        G.append([g1, g2])

    return np.array(G)


# =========================
# Data generation (CIRCLE)
# =========================

def generate_data(T, sensors, delta_q, R, radius=1.5, omega=0.1):
    L = len(sensors)

    theta = 0.0

    p_true = []
    q_true = [np.array([Q_SCALE, 0.5 * Q_SCALE])]
    y_data = []

    for _ in range(T):
        theta += omega

        p = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta)
        ])

        q_new = q_true[-1] + np.random.randn(2) * delta_q * Q_SCALE

        p_true.append(p)
        q_true.append(q_new)

        G = compute_G((float(p[0]), float(p[1])), sensors)
        noise = np.random.multivariate_normal(np.zeros(L), R)

        y = G @ q_new + noise
        y_data.append(y)

    return np.array(p_true), np.array(q_true), np.array(y_data)


# =========================
# Particle
# =========================

class Particle:
    def __init__(self, p, theta, q_mean, q_cov, weight):
        self.p = p
        self.theta = theta
        self.q_mean = q_mean
        self.q_cov = q_cov
        self.weight = weight


# =========================
# Init
# =========================

def init_particles(n, radius):
    particles = []
    for _ in range(n):
        theta = np.random.uniform(0, 2*np.pi)

        p = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta)
        ])

        q_mean = np.random.randn(2) * Q_SCALE
        q_cov = (Q_SCALE**2) * 0.5 * np.eye(2)

        particles.append(Particle(p, theta, q_mean, q_cov, 1.0 / n))

    return particles


# =========================
# Normalize
# =========================

def normalize_weights(particles):
    weights = np.array([p.weight for p in particles])
    weights += 1e-300
    weights /= np.sum(weights)

    for i, p in enumerate(particles):
        p.weight = weights[i]


# =========================
# ESS
# =========================

def effective_sample_size(particles):
    w = np.array([p.weight for p in particles])
    return 1.0 / np.sum(w**2)


# =========================
# Resampling
# =========================

def systematic_resample(particles):
    N = len(particles)
    weights = np.array([p.weight for p in particles])

    positions = (np.arange(N) + np.random.rand()) / N
    cumulative = np.cumsum(weights)

    indexes = np.zeros(N, dtype=int)
    i, j = 0, 0

    while i < N:
        if positions[i] < cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    new_particles = []
    for idx in indexes:
        p = particles[idx]
        new_particles.append(
            Particle(p.p.copy(), p.theta, p.q_mean.copy(), p.q_cov.copy(), 1.0 / N)
        )

    return new_particles


# =========================
# RBPF step
# =========================

def rbpf_step(particles, y, sensors, delta_q, R, radius, omega, sigma_theta=0.02, accumulate=True):
    L = len(sensors)

    for p in particles:

        # --- propagate (движение по кругу)
        p.theta = p.theta + omega + np.random.randn() * sigma_theta

        p.p = np.array([
            radius * np.cos(p.theta),
            radius * np.sin(p.theta)
        ])

        # --- KF predict
        q_mean_pred = p.q_mean
        q_cov_pred = p.q_cov + (delta_q**2) * (Q_SCALE**2) * np.eye(2)

        # --- observation model
        G = compute_G((float(p.p[0]), float(p.p[1])), sensors)
        S = G @ q_cov_pred @ G.T + R

        try:
            inv_S = np.linalg.inv(S)
            innovation = y - G @ q_mean_pred

            exponent = -0.5 * innovation.T @ inv_S @ innovation
            sign, logdet_S = np.linalg.slogdet(S)

            log_likelihood = (
                exponent
                - 0.5 * logdet_S
                - 0.5 * L * np.log(2 * np.pi)
            )

            likelihood = np.exp(log_likelihood)

        except np.linalg.LinAlgError:
            likelihood = 1e-300
            innovation = np.zeros(L)

        if accumulate:
            p.weight *= likelihood
        else:
            p.weight = likelihood

        # --- KF update
        K = q_cov_pred @ G.T @ np.linalg.inv(S)
        p.q_mean = q_mean_pred + K @ innovation
        p.q_cov = (np.eye(2) - K @ G) @ q_cov_pred

    normalize_weights(particles)

    if effective_sample_size(particles) < len(particles) / 2:
        particles = systematic_resample(particles)

    return particles


# =========================
# Run
# =========================

def run_rbpf(y_data, sensors, N=200, accumulate=True):
    delta_q = 0.01

    radius = 1.5
    omega = 0.1

    L = len(sensors)
    R = 1e-9 * np.eye(L)

    particles = init_particles(N, radius)

    est_p = []

    for y in y_data:
        particles = rbpf_step(
            particles, y, sensors, delta_q, R,
            sigma_theta=0.1,
            radius=radius,
            omega=omega,
            accumulate=accumulate
        )

        p_mean = np.sum([p.weight * p.p for p in particles], axis=0)
        est_p.append(p_mean)

    return np.array(est_p)


# =========================
# Main
# =========================

if __name__ == "__main__":

    np.random.seed(0)

    sensors = generate_sensors(grid_size=5, spacing=1.0)

    T = 200
    delta_q = 0.05

    L = len(sensors)
    R = 1e-7 * np.eye(L)

    p_true, q_true, y_data = generate_data(
        T, sensors, delta_q, R,
        radius=1.5,
        omega=0.1
    )

    est_acc = run_rbpf(y_data, sensors, accumulate=True)
    est_noacc = run_rbpf(y_data, sensors, accumulate=False)

    # Plot
    plt.figure(figsize=(10, 8))

    plt.plot(p_true[:, 0], p_true[:, 1], label="True", linewidth=2)
    plt.plot(est_acc[:, 0], est_acc[:, 1], "--", label="RBPF (accumulate)")
    plt.plot(est_noacc[:, 0], est_noacc[:, 1], ":", label="RBPF (no accumulate)")

    sx = [s[0] for s in sensors]
    sy = [s[1] for s in sensors]
    plt.scatter(sx, sy, marker="x", label="Sensors")

    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("RBPF (CIRCULAR MOTION)")
    plt.show()