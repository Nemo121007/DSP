from typing import List, Tuple

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

def generate_sensors(radius: float = 2.0) -> List[np.ndarray]:
    """
    Генерация расположения датчиков на окружности.
    
    Датчики расположены на окружности радиуса radius на высоте h,
    равномерно распределены по углам (L датчиков всего).
    
    Args:
        radius: Радиус окружности, на которой расположены датчики (м). По умолчанию 2.0.
        
    Returns:
        List[np.ndarray]: Список 3D координат датчиков, каждый элемент - массив [x, y, z].
    """
    sensors = []
    for i in range(L):
        angle: float = 2 * np.pi * i / L
        sensors.append(np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            h
        ]))
    return sensors


# =========================
# G(p)
# =========================

def compute_G(p: Tuple[float, float], sensors: List[np.ndarray]) -> np.ndarray:
    """
    Вычисление матрицы Якобиана G - производные магнитного поля по дипольному моменту.
    
    Для каждого датчика вычисляются частные производные магнитного поля 
    по компонентам дипольного момента q в 2D пространстве (px, py).
    
    Физическая модель: магнитное поле от диполя следует закону Кулона с константой CONST.
    Градиент поля вычисляется аналитически на основе расстояния между источником и датчиком.
    
    Args:
        p: Кортеж (px, py) - 2D позиция источника в плоскости XY (м).
        sensors: Список 3D координат датчиков размерности (L, 3), где L - количество датчиков.
        
    Returns:
        np.ndarray: Матрица размер (L, 2) - якобиан, где каждая строка содержит 
                   градиенты для соответствующего датчика [dH/dpx, dH/dpy].
                   
    Notes:
        - Используется единица 1e-12 для избежания деления на ноль в знаменателе.
        - Формулы основаны на магнитостатике и аналитическом дифференцировании.
    """
    # H(k)/u(k)     (75, 76)
    G = []
    px, py = p  # (u_k)

    for r in sensors:
        dx = r[0] - px
        dy = r[1] - py
        dz = r[2]

        d2 = dx**2 + dy**2 + dz**2  # (x_k, дипольный момент)
        d3 = d2 ** 1.5 + 1e-12

        g1 = CONST * (dy / d3)
        g2 = CONST * (-dx / d3)

        G.append([g1, g2])

    return np.array(G)


# =========================
# Data generation
# =========================

def generate_data(T: int, 
                  sensors: List[np.ndarray], 
                  lambda_p: float, 
                  delta_q: float, 
                  R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирование синтетических данных временного ряда для задачи фильтрации.
    
    Генерирует T-шаговый временной ряд позиций источника, дипольных моментов 
    и наблюдений магнитного поля в датчиках с добавлением гауссова шума.
    
    Динамическая модель:
        - Позиция: p_k = p_{k-1} + w_p, где w_p ~ N(0, lambda_p^2 * I)
        - Момент: q_k = q_{k-1} + w_q, где w_q ~ N(0, delta_q^2 * Q_SCALE^2 * I)
    
    Модель наблюдения:
        - y_k = G(p_k) * q_k + v_k, где v_k ~ N(0, R)
    
    Args:
        T: Количество временных шагов (целое число > 0).
        sensors: Список 3D координат датчиков размерности (L, 3).
        lambda_p: Стандартное отклонение шума динамики позиции (м).
        delta_q: Коэффициент масштаба для шума динамики момента (безразмерный).
        R: Матрица ковариации шума наблюдения размер (L, L).
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Кортеж из трёх массивов:
            - p_true: Истинные позиции источника размер (T+1, 2).
            - q_true: Истинные дипольные моменты размер (T+1, 2).
            - y_data: Наблюдения магнитного поля размер (T, L).
    """
    p_true = [np.array([-1.0, -1.0])]
    q_true = [np.array([Q_SCALE, 0.5 * Q_SCALE])]

    y_data = []

    for _ in range(T):
        p_new = p_true[-1] + np.random.randn(2) * lambda_p
        q_new = q_true[-1] + np.random.randn(2) * delta_q * Q_SCALE

        p_true.append(p_new)
        q_true.append(q_new)

        G = compute_G((float(p_new[0]), float(p_new[1])), sensors)
        noise = np.random.multivariate_normal(np.zeros(L), R)

        y = G @ q_new + noise
        y_data.append(y)

    return np.array(p_true), np.array(q_true), np.array(y_data)


# =========================
# Particle
# =========================

class Particle:
    """
    Представление одной частицы в алгоритме Rao-Blackwellized Particle Filter.
    
    Содержит 2D позицию источника и гауссовский параметр для дипольного момента.
    Используется для представления гипотезы о состоянии системы.
    
    Attributes:
        p: 2D позиция источника в плоскости XY формы (2,).
        q_mean: Среднее значение дипольного момента формы (2,).
        q_cov: Матрица ковариации дипольного момента размер (2, 2).
        weight: Вес частицы (0 <= weight <= 1), используется в фильтре.
    """
    
    def __init__(self, p: np.ndarray, q_mean: np.ndarray, q_cov: np.ndarray, weight: float) -> None:
        """
        Инициализация частицы.
        
        Args:
            p: 2D позиция источника формы (2,).
            q_mean: Среднее значение дипольного момента формы (2,).
            q_cov: Матрица ковариации дипольного момента размер (2, 2).
            weight: Начальный вес частицы.
        """
        self.p: np.ndarray = p
        self.q_mean: np.ndarray = q_mean
        self.q_cov: np.ndarray = q_cov
        self.weight: float = weight


# =========================
# Init
# =========================

def init_particles(N: int) -> List[Particle]:
    """
    Инициализация множества частиц с одинаковыми начальными весами.
    
    Каждая частица получает случайную начальную позицию и параметры момента,
    равномерно распределённые вокруг нуля.
    
    Args:
        N: Количество частиц для инициализации.
        
    Returns:
        List[Particle]: Список инициализированных частиц с весом 1/N каждая.
    """
    particles: List[Particle] = []
    for _ in range(N):
        p = np.random.randn(2) * 0.5
        q_mean = np.random.randn(2) * Q_SCALE
        q_cov = (Q_SCALE**2) * 0.5 * np.eye(2)

        particles.append(Particle(p, q_mean, q_cov, 1.0 / N))
    return particles


# =========================
# Normalize
# =========================

def normalize_weights(particles: List[Particle]) -> None:
    """
    Нормализация весов всех частиц так, чтобы их сумма была равна 1.
    
    Добавляет малую константу 1e-300 для численной стабильности перед
    нормализацией, чтобы избежать деления на ноль.
    
    Args:
        particles: Список частиц, веса которых требуют нормализации.
        
    Returns:
        None: Функция модифицирует веса частиц на месте.
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
    Вычисление эффективного размера выборки для набора взвешенных частиц.
    
    Эффективный размер выборки (ESS) используется для определения необходимости
    переsampleния. Низкое значение ESS указывает на вырождение весов.
    
    ESS = 1 / sum(w_i^2)
    
    Args:
        particles: Список частиц с весами.
        
    Returns:
        float: Значение эффективного размера выборки (0 < ESS <= N).
    """
    w = np.array([p.weight for p in particles])
    return 1.0 / np.sum(w**2)


# =========================
# Resampling
# =========================

def systematic_resample(particles: List[Particle]) -> List[Particle]:
    """
    Систематическое переsampleние частиц на основе их весов.
    
    Реализует систематическое переsampleние, которое сохраняет разнообразие частиц
    лучше, чем многочисленное переsampleние. Частицы с высокими весами создаются
    в несколько копий, а с низкими весами удаляются.
    
    Args:
        particles: Список частиц с весами для переsampleния.
        
    Returns:
        List[Particle]: Новый список частиц, переsampleненных на основе весов,
                       все с одинаковым весом 1/N.
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
            Particle(p.p.copy(), p.q_mean.copy(), p.q_cov.copy(), 1.0 / N)
        )

    return new_particles


# =========================
# RBPF step
# =========================

def rbpf_step(particles: List[Particle], 
              y: np.ndarray, 
              sensors: List[np.ndarray], 
              lambda_p: float, 
              delta_q: float, 
              R: np.ndarray, 
              accumulate: bool = True) -> List[Particle]:
    """
    Один шаг Rao-Blackwellized Particle Filter.
    
    Выполняет генерацию частиц, предсказание и обновление Фильтра Калмана,
    вычисление весов, нормализацию и ресэмпл при необходимости.
    
    Процесс:
        1. Генерирует позиции частиц согласно модели движения
        2. Предсказывает распределение дипольного момента (KF predict)
        3. Вычисляет правдоподобие наблюдений (likelihood)
        4. Обновляет веса частиц
        5. Нормализует веса
        6. Ресэмпл при низком ESS (< N/2)
        7. Обновляет оценку момента (KF update)
    
    Args:
        particles: Список частиц с текущим состоянием.
        y: Вектор наблюдений магнитного поля размер (L,).
        sensors: Список 3D координат датчиков размер (L, 3).
        lambda_p: Стандартное отклонение шума движения позиции (м).
        delta_q: Коэффициент масштаба шума движения момента.
        R: Матрица ковариации шума наблюдения размер (L, L).
        accumulate: Если True, веса накапливаются; если False, переустанавливаются.
        
    Returns:
        List[Particle]: Обновленный список частиц после выполнения шага фильтра.
        
    Notes:
        - Используется матричное представление для вычисления правдоподобия
        - Добавляются небольшие константы (1e-12, 1e-300) для численной стабильности
    """

    for p in particles:

        # --- propagate
        p.p = p.p + np.random.randn(2) * lambda_p

        # --- KF predict
        q_mean_pred = p.q_mean
        q_cov_pred = p.q_cov + (delta_q**2) * (Q_SCALE**2) * np.eye(2)

        # --- model
        G = compute_G((float(p.p[0]), float(p.p[1])), sensors)

        S = G @ q_cov_pred @ G.T + R

        innovation = np.zeros(L)
        try:
            inv_S = np.linalg.inv(S)
            innovation = y - G @ q_mean_pred

            exponent = float(-0.5 * innovation.T @ inv_S @ innovation)
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

def run_rbpf(y_data: np.ndarray, sensors: List[np.ndarray], N: int = 200, accumulate: bool = True) -> np.ndarray:
    """
    Запуск Rao-Blackwellized Particle Filter на полной последовательности наблюдений.
    
    Инициализирует множество частиц и выполняет RBPF на каждом временном шаге,
    возвращая оценки позиций источника на основе средневзвешенных позиций частиц.
    
    Args:
        y_data: Матрица наблюдений магнитного поля размер (T, L), где T - количество шагов, L - количество датчиков.
        sensors: Список 3D координат датчиков размерности (L, 3).
        N: Количество частиц для фильтра (по умолчанию 200).
        accumulate: Если True, веса накапливаются; если False, переустанавливаются (по умолчанию True).
        
    Returns:
        np.ndarray: Матрица оценённых позиций источника размер (T, 2).
    """

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
