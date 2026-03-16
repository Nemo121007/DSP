import numpy as np
import matplotlib.pyplot as plt

# =========================
# Constants (PHYSICAL)
# =========================

MU0 = 4 * np.pi * 1e-7
CONST = MU0 / (4 * np.pi)   # 1e-7

h = 0.5
L = 8

Q_SCALE = 1e5   # масштаб диполя


# =========================
# Sensors
# =========================

def generate_sensors(radius=2.0):
    sensors = []
    for i in range(L):
        angle = 2 * np.pi * i / L
        sensors.append(np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            h
        ]))
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
        d3 = d2 ** 1.5 + 1e-12

        g1 = CONST * (dy / d3)
        g2 = CONST * (-dx / d3)

        G.append([g1, g2])

    return np.array(G)


# =========================
# Data generation
# =========================

def generate_data(T, sensors, lambda_p, delta_q, R):
    p_true = [np.array([-1.0, -1.0])]
    q_true = [np.array([Q_SCALE, 0.5 * Q_SCALE])]

    y_data = []

    for _ in range(T):
        p_new = p_true[-1] + np.random.randn(2) * lambda_p
        q_new = q_true[-1] + np.random.randn(2) * delta_q * Q_SCALE

        p_true.append(p_new)
        q_true.append(q_new)

        G = compute_G(p_new, sensors)
        noise = np.random.multivariate_normal(np.zeros(L), R)

        y = G @ q_new + noise
        y_data.append(y)

    return np.array(p_true), np.array(q_true), np.array(y_data)


# =========================
# Particle
# =========================

class Particle:
    def __init__(self, p, q_mean, q_cov, weight):
        self.p = p
        self.q_mean = q_mean
        self.q_cov = q_cov
        self.weight = weight


# =========================
# Init
# =========================

def init_particles(N):
    particles = []
    for _ in range(N):
        p = np.random.randn(2) * 0.5
        q_mean = np.random.randn(2) * Q_SCALE
        q_cov = (Q_SCALE**2) * 0.5 * np.eye(2)

        particles.append(Particle(p, q_mean, q_cov, 1.0 / N))
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
            Particle(p.p.copy(), p.q_mean.copy(), p.q_cov.copy(), 1.0 / N)
        )

    return new_particles


# =========================
# RBPF step
# =========================

def rbpf_step(particles, y, sensors, lambda_p, delta_q, R, accumulate=True):

    for p in particles:

        # --- propagate
        p.p = p.p + np.random.randn(2) * lambda_p

        # --- KF predict
        q_mean_pred = p.q_mean
        q_cov_pred = p.q_cov + (delta_q**2) * (Q_SCALE**2) * np.eye(2)

        # --- model
        G = compute_G(p.p, sensors)

        S = G @ q_cov_pred @ G.T + R

        try:
            inv_S = np.linalg.inv(S)
            innovation = y - G @ q_mean_pred

            exponent = -0.5 * innovation.T @ inv_S @ innovation
            det_S = np.linalg.det(S)

            likelihood = np.exp(exponent) / np.sqrt((2 * np.pi)**L * det_S + 1e-300)
        except np.linalg.LinAlgError:
            likelihood = 1e-300

        # --- ключевая разница
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

    lambda_p = 0.05
    delta_q = 0.01

    # физический шум
    R = 1e-9 * np.eye(L)

    particles = init_particles(N)

    est_p = []

    for y in y_data:
        particles = rbpf_step(particles, y, sensors, lambda_p, delta_q, R, accumulate)

        p_mean = np.sum([p.weight * p.p for p in particles], axis=0)
        est_p.append(p_mean)

    return np.array(est_p)


# =========================
# Main
# =========================

if __name__ == "__main__":

    np.random.seed(0)

    sensors = generate_sensors()

    T = 200

    lambda_p = 0.05
    delta_q = 0.01
    R = 1e-9 * np.eye(L)

    p_true, q_true, y_data = generate_data(T, sensors, lambda_p, delta_q, R)

    est_acc = run_rbpf(y_data, sensors, accumulate=True)
    est_noacc = run_rbpf(y_data, sensors, accumulate=False)

    # =========================
    # Plot
    # =========================

    plt.figure(figsize=(10, 8))

    plt.plot(p_true[:, 0], p_true[:, 1], label="True", linewidth=2)
    plt.plot(est_acc[:, 0], est_acc[:, 1], '--', label="RBPF (accumulate)")
    plt.plot(est_noacc[:, 0], est_noacc[:, 1], ':', label="RBPF (no accumulate)")

    sx = [s[0] for s in sensors]
    sy = [s[1] for s in sensors]
    plt.scatter(sx, sy, marker='x', label="Sensors")

    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("RBPF (PHYSICAL SCALE)")

    plt.show()