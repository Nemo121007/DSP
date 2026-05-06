from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Constants
# =========================
# Магнитная постоянная
MU0 = 4 * np.pi * 1e-7
CONST = MU0 / (4 * np.pi)

# Высота подъема датчиков
h = 0.5
# Масштаб магнитного момента (для удобного масштабирования MU0)
Q_SCALE = 1e5


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
            sensors.append(np.array([x, y, h]))

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

    return np.array(G)


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
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Возвращает кортеж из трех массивов:
            - p_true: Истинные позиции (x, y) на каждом шаге.
            - q_true: Истинные значения вектора магнитного момента.
            - y_data: Сгенерированные зашумленные измерения для сенсоров.
    """
    L = len(sensors)

    theta = 0.0

    p_true = []
    q_true = [np.array([Q_SCALE, 0.5 * Q_SCALE])]
    y_data = []

    for _ in range(T):
        theta += omega

        p = np.array([radius * np.cos(theta), radius * np.sin(theta)])

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
        theta (float): Угол (фаза) на окружности для модели движения.
        q_mean (np.ndarray): Вектор математического ожидания (для линейной подсистемы в KF).
        q_cov (np.ndarray): Ковариационная матрица оценки (для линейной подсистемы в KF).
        weight (float): Вес частицы.
    """

    def __init__(
        self,
        p: np.ndarray,
        theta: float,
        q_mean: np.ndarray,
        q_cov: np.ndarray,
        weight: float,
    ) -> None:
        self.p = p
        self.theta = theta
        self.q_mean = q_mean
        self.q_cov = q_cov
        self.weight = weight


# =========================
# Init
# =========================
def init_particles(n: int, radius: float) -> List[Particle]:
    """
    Инициализирует набор частиц для старта фильтрации.

    Args:
        n (int): Количество частиц.
        radius (float): Радиус, вокруг которого распределяются начальные позиции.

    Returns:
        List[Particle]: Список созданных объектов Particle.
    """
    particles = []
    for _ in range(n):
        theta = np.random.uniform(0, 2 * np.pi)

        p = np.array([radius * np.cos(theta), radius * np.sin(theta)])

        q_mean = np.random.randn(2) * Q_SCALE
        q_cov = (Q_SCALE**2) * 0.5 * np.eye(2)

        particles.append(Particle(p, theta, q_mean, q_cov, 1.0 / n))

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
    weights = np.array([p.weight for p in particles])
    weights += 1e-300
    weights /= np.sum(weights)

    for i, p in enumerate(particles):
        p.weight = weights[i]


# =========================
# ESS
# =========================
def effective_sample_size(particles: List[Particle]) -> float:
    """
    Вычисляет эффективный размер выборки,
    который служит индикатором необходимости ресэмплинга (перевыборки).

    Args:
        particles (List[Particle]): Список частиц фильтра.

    Returns:
        float: Значение ESS, от 1 до N (количества частиц).
    """
    w = np.array([p.weight for p in particles])
    return 1.0 / np.sum(w**2)


# =========================
# Resampling
# =========================
def systematic_resample(particles: List[Particle]) -> List[Particle]:
    """
    Выполняет систематический ресэмплинг (перевыборку) с целью отбросить частицы с малым весом
    и размножить частицы с большим весом, сохраняя при этом общее число.

    Args:
        particles (List[Particle]): Список текущих частиц.

    Returns:
        List[Particle]: Новый список частиц после ресэмплинга со сброшенными (одинаковыми) весами.
    """
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
def rbpf_step(
    particles: List[Particle],
    y: np.ndarray,
    sensors: List[np.ndarray],
    delta_q: float,
    R: np.ndarray,
    radius: float,
    omega: float,
    sigma_theta: float = 0.02,
) -> List[Particle]:
    """
    Выполняет один шаг фильтра частиц Rao-Blackwellized Particle Filter.
    В нем позиция обновляется фильтром частиц, а линейно-гауссовская часть (q) — фильтром Калмана.

    Args:
        particles (List[Particle]): Текущий набор частиц.
        y (np.ndarray): Вектор измерений на текущем шаге.
        sensors (List[np.ndarray]): Список сенсоров.
        delta_q (float): Параметр шума процесса для q.
        R (np.ndarray): Ковариационная матрица шума измерений.
        radius (float): Радиус кругового движения (параметр модели).
        omega (float): Угловая скорость.
        sigma_theta (float): Стандартное отклонение шума по углу движения.
        accumulate (bool): Если True, веса умножаются на правдоподобие (накапливаются).

    Returns:
        List[Particle]: Обновленный список частиц.
    """
    L = len(sensors)

    for p in particles:

        # --- propagate (прогноз движения частицы в виде движения по кругу)
        p.theta = p.theta + omega + np.random.randn() * sigma_theta
        p.p = np.array([radius * np.cos(p.theta), radius * np.sin(p.theta)])

        # --- KF predict
        # Прогноз линейной части
        # Фильтр Калмана оценивает дипольный момент
        q_mean_pred = p.q_mean
        q_cov_pred = p.q_cov + (delta_q**2) * (Q_SCALE**2) * np.eye(2)

        # --- observation model
        # Вычисляет матрицу наблюдения H_k(u_k) для текущей позиции частицы
        G = compute_G((float(p.p[0]), float(p.p[1])), sensors)
        # Ковариация невязки (инновации)
        S = G @ q_cov_pred @ G.T + R

        try:
            # Вычисляется вероятность получить измерение y при данном состоянии
            inv_S = np.linalg.inv(S)
            innovation = y - G @ q_mean_pred

            exponent = -0.5 * innovation.T @ inv_S @ innovation
            sign, logdet_S = np.linalg.slogdet(S)

            log_likelihood = exponent - 0.5 * logdet_S - 0.5 * L * np.log(2 * np.pi)

            likelihood = np.exp(log_likelihood)

        except np.linalg.LinAlgError:
            likelihood = 1e-300
            innovation = np.zeros(L)

        p.weight *= likelihood

        # --- KF update
        # Обновление весов и линейной части
        K = q_cov_pred @ G.T @ np.linalg.inv(S)
        p.q_mean = q_mean_pred + K @ innovation
        p.q_cov = (np.eye(2) - K @ G) @ q_cov_pred

    # Нормализация весов
    normalize_weights(particles)

    # Ресемплинг
    if effective_sample_size(particles) < len(particles) / 2:
        particles = systematic_resample(particles)

    return particles


# =========================
# Run
# =========================
def run_rbpf(
    y_data: np.ndarray, sensors: List[np.ndarray], N: int = 200) -> np.ndarray:
    """
    Запускает RBPF фильтр для полного набора измерений и возвращает оценку траектории.

    Args:
        y_data (np.ndarray): Массив измерений для всех временных шагов.
        sensors (List[np.ndarray]): Список сенсоров.
        N (int): Количество используемых частиц.

    Returns:
        np.ndarray: Оценка позиций p (x, y) для каждого временного шага, усредненная по весам частиц.
    """
    delta_q = 0.01

    radius = 1.5
    omega = 0.1

    L = len(sensors)
    R = 1e-9 * np.eye(L)

    particles = init_particles(N, radius)

    est_p = []

    for y in y_data:
        particles = rbpf_step(
            particles,
            y,
            sensors,
            delta_q,
            R,
            sigma_theta=0.1,
            radius=radius,
            omega=omega,
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
        T, sensors, delta_q, R, radius=1.5, omega=0.1
    )

    est_acc = run_rbpf(y_data, sensors)

    # Plot
    plt.figure(figsize=(10, 8))

    plt.plot(p_true[:, 0], p_true[:, 1], label="True", linewidth=2)
    plt.plot(est_acc[:, 0], est_acc[:, 1], "--", label="RBPF (accumulate)")

    sx = [s[0] for s in sensors]
    sy = [s[1] for s in sensors]
    plt.scatter(sx, sy, marker="x", label="Sensors")

    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("RBPF (CIRCULAR MOTION)")
    plt.show()
