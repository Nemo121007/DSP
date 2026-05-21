from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Constants
# =========================
# Магнитная постояннаяR
MU0 = 4 * np.pi * 1
CONST = 1

# Высота подъема датчиков
h = 0.5

# Масштаб магнитного момента (для удобного масштабирования MU0)
Q_SCALE = 1


# =========================
# Sensors (GRID)
# =========================
def generate_sensors(grid_size: int = 5, spacing: float = 1.0) -> List[np.ndarray]:
    """
    Создает квадратную сетку сенсоров.

    Args:
        grid_size (int): Размерность сетки (количество сенсоров по одной стороне).
        spacing (float): Расстояние между соседними сенсорами.

    Returns:
        List[np.ndarray]: Список координат сенсоров в виде массивов [x, y, z].
    """
    sensors = []
    offset = (grid_size - 1) / 2

    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - offset) * spacing
            y = (j - offset) * spacing
            sensors.append(np.array([x, y, h], dtype=float))

    return sensors


# =========================
# G(p)
# =========================
def compute_G(p: Tuple[float, float], sensors: List[np.ndarray]) -> np.ndarray:
    """
    Вычисляет матрицу наблюдений G(p) для заданного положения целевого объекта
    относительно всех сенсоров.

    Args:
        p (Tuple[float, float]): Координаты цели (x, y).
        sensors (List[np.ndarray]): Список координат сенсоров.

    Returns:
        np.ndarray: Матрица G формы (L, 2), где L — количество сенсоров.
    """
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

    return np.array(G, dtype=float)


# =========================
# Data generation (CIRCLE)
# =========================
def generate_data(
    T: int,
    sensors: List[np.ndarray],
    delta_q: float,
    R: np.ndarray,
    radius: float = 1.5,
    omega: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует синтетические данные (измерения) при движении объекта по окружности.

    Args:
        T (int): Количество временных шагов.
        sensors (List[np.ndarray]): Список координат сенсоров.
        delta_q (float): Коэффициент случайного блуждания магнитного момента (состояния).
        R (np.ndarray): Ковариационная матрица шума измерений.
        radius (float): Радиус круговой траектории.
        omega (float): Угловая скорость движения по окружности.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - p_true: Истинные позиции (x, y) на каждом шаге.
            - q_true: Истинные значения вектора магнитного момента.
            - y_data: Сгенерированные зашумленные измерения для сенсоров.
    """
    L = len(sensors)

    theta = 0.0
    p_true = []
    q_true = [np.array([Q_SCALE, 0.5 * Q_SCALE], dtype=float)]
    y_data = []

    for _ in range(T):
        theta += omega
        p = np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=float)

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
    """
    Класс, представляющий отдельную частицу в фильтре (RBPF).

    Attributes:
        p (np.ndarray): Оценка координат позиции (x, y).
        q_mean (np.ndarray): Вектор математического ожидания (для линейной подсистемы в KF).
        q_cov (np.ndarray): Ковариационная матрица оценки (для линейной подсистемы в KF).
        weight (float): Вес частицы.
    """

    def __init__(
        self,
        p: np.ndarray,
        q_mean: np.ndarray,
        q_cov: np.ndarray,
        weight: float,
    ) -> None:
        self.p = p.astype(float).copy()
        self.q_mean = q_mean.astype(float).copy()
        self.q_cov = q_cov.astype(float).copy()
        self.weight = float(weight)


# =========================
# Init
# =========================
def init_particles(
    n: int,
    init_center: np.ndarray,
    pos_std: float = 0.5,
) -> List[Particle]:
    """
    Инициализирует набор частиц для старта фильтрации.

    Args:
        n (int): Количество частиц.
        init_center (np.ndarray): Центр начального облака частиц, форма (2,).
        pos_std (float): Стандартное отклонение начального разброса по позиции.

    Returns:
        List[Particle]: Список созданных объектов Particle.
    """
    particles: List[Particle] = []
    for _ in range(n):
        p = init_center + np.random.randn(2) * pos_std
        q_mean = np.random.randn(2) * Q_SCALE
        q_cov = (Q_SCALE**2) * 0.5 * np.eye(2)
        particles.append(Particle(p, q_mean, q_cov, 1.0 / n))
    return particles


# =========================
# Normalize
# =========================
def normalize_weights(particles: List[Particle]) -> None:
    """
    Нормирует веса всех частиц так, чтобы их сумма равнялась 1.

    Args:
        particles (List[Particle]): Список частиц, чьи веса будут изменены in-place.
    """
    weights = np.array([p.weight for p in particles], dtype=float)
    weights += 1e-300
    weights /= np.sum(weights)

    for i, p in enumerate(particles):
        p.weight = float(weights[i])


# =========================
# ESS
# =========================
def effective_sample_size(particles: List[Particle]) -> float:
    """
    Вычисляет эффективный размер выборки,
    который служит индикатором необходимости ресэмплинга.

    Args:
        particles (List[Particle]): Список частиц фильтра.

    Returns:
        float: Значение ESS, от 1 до N.
    """
    w = np.array([p.weight for p in particles], dtype=float)
    return float(1.0 / np.sum(w**2))


# =========================
# Resampling
# =========================
def systematic_resample(particles: List[Particle]) -> List[Particle]:
    """
    Выполняет систематический ресэмплинг.

    Args:
        particles (List[Particle]): Список текущих частиц.

    Returns:
        List[Particle]: Новый список частиц после ресэмплинга.
    """
    N = len(particles)
    weights = np.array([p.weight for p in particles], dtype=float)
    cumulative = np.cumsum(weights)

    positions = (np.arange(N) + np.random.rand()) / N
    indexes = np.zeros(N, dtype=int)

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    new_particles: List[Particle] = []
    for idx in indexes:
        src = particles[idx]
        new_particles.append(
            Particle(
                p=src.p.copy(),
                q_mean=src.q_mean.copy(),
                q_cov=src.q_cov.copy(),
                weight=1.0 / N,
            )
        )

    return new_particles


# =========================
# RBPF step
# =========================
def rbpf_step(
    particles: List[Particle],
    y: np.ndarray,
    sensors: List[np.ndarray],
    delta_q: float,
    R: np.ndarray,
    sigma_pos: float = 0.1,
) -> List[Particle]:
    """
    Выполняет один шаг RBPF.

    Позиция частицы обновляется как случайное блуждание.
    Линейно-гауссовская часть q оценивается фильтром Калмана внутри каждой частицы.

    Args:
        particles (List[Particle]): Текущий набор частиц.
        y (np.ndarray): Вектор измерений на текущем шаге.
        sensors (List[np.ndarray]): Список сенсоров.
        delta_q (float): Параметр шума процесса для q.
        R (np.ndarray): Ковариационная матрица шума измерений.
        sigma_pos (float): Стандартное отклонение шума перехода по позиции.

    Returns:
        List[Particle]: Обновленный список частиц.
    """
    L = len(sensors)
    Q_p = (sigma_pos**2) * np.eye(2)

    for p in particles:
        p.p = p.p + np.random.multivariate_normal(np.zeros(2), Q_p)

        q_mean_pred = p.q_mean
        q_cov_pred = p.q_cov + (delta_q**2) * (Q_SCALE**2) * np.eye(2)

        G = compute_G((float(p.p[0]), float(p.p[1])), sensors)
        S = G @ q_cov_pred @ G.T + R

        try:
            innovation = y - G @ q_mean_pred
            inv_S = np.linalg.inv(S)

            exponent = -0.5 * innovation.T @ inv_S @ innovation
            sign, logdet_S = np.linalg.slogdet(S)

            if sign <= 0:
                likelihood = 1e-300
            else:
                log_likelihood = exponent - 0.5 * logdet_S - 0.5 * L * np.log(2 * np.pi)
                likelihood = float(np.exp(log_likelihood))
        except np.linalg.LinAlgError:
            likelihood = 1e-300
            innovation = np.zeros(L, dtype=float)

        p.weight *= likelihood

        # --- Kalman update for q
        try:
            K = q_cov_pred @ G.T @ np.linalg.inv(S)
            p.q_mean = q_mean_pred + K @ innovation
            p.q_cov = (np.eye(2) - K @ G) @ q_cov_pred
        except np.linalg.LinAlgError:
            p.q_mean = q_mean_pred
            p.q_cov = q_cov_pred

    normalize_weights(particles)

    if effective_sample_size(particles) < len(particles) / 2:
        particles = systematic_resample(particles)

    return particles


# =========================
# Run
# =========================
def run_rbpf(
    y_data: np.ndarray,
    sensors: List[np.ndarray],
    N: int = 200,
) -> np.ndarray:
    """
    Запускает RBPF фильтр для полного набора измерений и возвращает оценку траектории.

    Args:
        y_data (np.ndarray): Массив измерений для всех временных шагов.
        sensors (List[np.ndarray]): Список сенсоров.
        N (int): Количество используемых частиц.

    Returns:
        np.ndarray: Оценка позиций p (x, y) для каждого временного шага.
    """
    delta_q = 0.01
    R = 1e-2 * np.eye(len(sensors))
    particles = init_particles(N, init_center=np.array([0.0, 0.0]), pos_std=1.0)

    est_p = []

    for y in y_data:
        particles = rbpf_step(
            particles=particles,
            y=y,
            sensors=sensors,
            delta_q=delta_q,
            R=R,
            sigma_pos=0.5,
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
        T=T,
        sensors=sensors,
        delta_q=delta_q,
        R=R,
        radius=1.5,
        omega=0.1,
    )

    est_acc = run_rbpf(y_data, sensors, N=200)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(p_true[:, 0], p_true[:, 1], label="True", linewidth=2)
    plt.plot(est_acc[:, 0], est_acc[:, 1], "--", label="RBPF (particle motion)")

    sx = [s[0] for s in sensors]
    sy = [s[1] for s in sensors]
    plt.scatter(sx, sy, marker="x", label="Sensors")

    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("RBPF (CIRCULAR MOTION)")
    plt.show()
