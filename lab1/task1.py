import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


def generate_trajectory_and_measurements(
    *,
    T: float,
    num_steps: int,
    x0: np.ndarray,
    W_L: float,
    W_R: float,
    B: float,
    R: np.ndarray,
    seed: int | None = None,
    control_fn=None,
):
    """Генерирует истинную траекторию и зашумлённые измерения.
    Args:
        T: Шаг дискретизации (с)
        num_steps: Количество шагов для генерации
        x0: Начальное истинное состояние [x, y, theta]
        W_L: Радиус левого колеса (м)
        W_R: Радиус правого колеса (м)
        B: Расстояние от центра до колеса (м)
        R: Ковариационная матрица шума измерений (3x3)
        seed: Сид для генератора случайных чисел (для воспроизводимости)
        control_fn: Функция управления, которая принимает (i, t) и возвращает (u_L, u_R).
                    Если None, будет использоваться функция по умолчанию.
    Return:
        history_true: (N, 3) [x, y, r] - истинные состояния в начале каждого шага
        history_measurements: (N, 3) - наблюдаемые состояния
        controls: (N, 2) [u_L, u_R] - управляющие воздействия на каждом шаге
        times: (N) - временные метки для каждого шага
    Note:
        - R используется как ковариация измерения; шум генерируется через multivariate_normal.
        - Семантика шага: в историю на шаге i записывается состояние *в начале* шага,
          затем применяется модель движения для перехода к следующему шагу.
    """
    if control_fn is None:

        def control_fn(i: int, t: float):
            # Управление (угловые скорости колес)
            u_L = 10.0 * np.sin(2) + i * 0
            u_R = 10.0 * np.cos(2) + t * 0
            return u_L, u_R

    if T <= 0:
        raise ValueError("T <= 0")
    if num_steps < 0:
        raise ValueError("num_steps < 0")

    x0 = np.asarray(x0, dtype=float).reshape(
        3,
    )
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R должен иметь размерность (3, 3)")

    rng = np.random.default_rng(seed)

    true_state = x0.copy()
    history_true: list[np.ndarray] = []
    history_measurements: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    times: list[float] = []

    for i in range(num_steps):
        t = i * T
        u_L, u_R = control_fn(i, t)

        # Сохраняем истинное состояние (в начале шага)
        history_true.append(true_state.copy())
        controls.append(np.array([u_L, u_R], dtype=float))
        times.append(t)

        # Генерируем зашумленные измерения
        noise = rng.multivariate_normal(mean=np.zeros(3), cov=R)
        measurement = true_state + noise
        history_measurements.append(measurement)

        # Обновление истинного состояния для следующего шага
        s_L = W_L * u_L
        s_R = W_R * u_R
        s_t = (s_R + s_L) / 2.0
        s_r = (s_R - s_L) / (2.0 * B)

        r = true_state[2]
        true_state = np.array(
            [
                true_state[0]
                + T * s_t * np.cos(r)
                - 0.5 * T**2 * s_t * s_r * np.sin(r),
                true_state[1]
                + T * s_t * np.sin(r)
                + 0.5 * T**2 * s_t * s_r * np.cos(r),
                true_state[2] + T * s_r,
            ]
        )

    return (
        np.asarray(history_true),
        np.asarray(history_measurements),
        np.asarray(controls),
        np.asarray(times),
    )


def wrap_to_pi(theta) -> float:
    """
    Нормализует угол(ы) в диапазон [-pi, pi].
    Работает как со скалярами, так и с numpy-массивами.
    """
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


# Параметры симуляции
T = 0.1  # Шаг дискретизации (с)
W_L = 0.05  # Радиус левого колеса (м)
W_R = 0.05  # Радиус правого колеса (м)
B = 0.2  # Расстояние от центра до колеса

# Параметры шума
# Шум процесса (неопределенность модели движения)
# Q = np.diag([0.01, 0.01, np.deg2rad(1.0)])**2
# Шум процесса РАВЕН НУЛЮ, если модель считается идеальной
Q = np.zeros((3, 3))

# Шум измерений (точность сенсоров)
R = np.diag([0.1, 0.1, np.deg2rad(5.0)]) ** 2

# Инициализация
# Начальное истинное состояние [x, y, r]
true_state_0 = np.array([0, 0, 0.0])
# Начальная оценка состояния EKF
estimated_state = np.array([0, 0, 0.0])
# Начальная ковариационная матрица (высокая неопределенность)
P = np.diag([1.0, 1.0, np.deg2rad(50.0)]) ** 2

# Генерация траектории и измерений
num_steps = 300
history_true, history_measurements, controls, times = (
    generate_trajectory_and_measurements(
        T=T,
        num_steps=num_steps,
        x0=true_state_0,
        W_L=W_L,
        W_R=W_R,
        B=B,
        R=R,
        seed=None,
    )
)

# Прогон EKF по сгенерированным данным
history_estimated = []
for i in range(num_steps):
    u_L, u_R = controls[i]
    measurement = history_measurements[i]

    # ЭТАП ПРЕДСКАЗАНИЯ

    # Вычисляем линейные и угловые скорости
    s_L = W_L * u_L
    s_R = W_R * u_R
    s_t = (s_R + s_L) / 2.0
    s_r = (s_R - s_L) / (2.0 * B)

    # Предсказание состояния
    r = estimated_state[2]
    state_pred = np.array([
        estimated_state[0] + T * s_t * np.cos(r) - 0.5 * T**2 * s_t * s_r * np.sin(r),
        estimated_state[1] + T * s_t * np.sin(r) + 0.5 * T**2 * s_t * s_r * np.cos(r),
        estimated_state[2] + T * s_r,
    ])  # (7), (31)
    state_pred[2] = wrap_to_pi(state_pred[2])

    # Вычисление матрицы Якоби
    F = np.array(
        [
            [1, 0, -T * s_t * np.sin(r) - 0.5 * T**2 * s_t * s_r * np.cos(r)],
            [0, 1, T * s_t * np.cos(r) - 0.5 * T**2 * s_t * s_r * np.sin(r)],
            [0, 0, 1],
        ]
    )  # (25)

    # Предсказание ковариации
    P_pred = F @ P @ F.T + Q  # (8), (32)

    # ЭТАП КОРРЕКЦИИ

    # Усиление Калмана
    # H - единичная матрица, поэтому S = P_pred + R
    S = P_pred + R  # (9), (33)
    K = P_pred @ np.linalg.inv(S)  # (10), (34)

    # Обновление состояния с помощью измерения
    innovation = measurement - state_pred
    innovation[2] = wrap_to_pi(innovation[2])

    estimated_state = state_pred + K @ innovation  # (11), (35)
    estimated_state[2] = wrap_to_pi(estimated_state[2])

    # Обновление ковариации
    P = P_pred - K @ S @ K.T  # (12), (36)

    # Сохраняем оценку
    history_estimated.append(estimated_state.copy())

# Визуализация результатов
history_estimated = np.array(history_estimated)

plt.figure(figsize=(12, 8))
plt.plot(history_true[:, 0], history_true[:, 1], "g-", label="Истинная траектория")
plt.plot(
    history_measurements[:, 0],
    history_measurements[:, 1],
    "k.",
    label="Зашумленные измерения",
    markersize=4,
    alpha=0.7,
)
plt.plot(
    history_estimated[:, 0],
    history_estimated[:, 1],
    "b-",
    label="Траектория, оцененная EKF",
)
plt.title("Отслеживание положения мобильного робота с помощью EKF")
plt.xlabel("Координата X (м)")
plt.ylabel("Координата Y (м)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# Визуализация ошибки
error = history_estimated - history_true

fig, (ax_xy, ax_theta) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Ошибки по координатам
ax_xy.plot(error[:, 0], label="Ошибка по X")
ax_xy.plot(error[:, 1], label="Ошибка по Y")
ax_xy.set_title("Ошибка оценки EKF")
ax_xy.set_ylabel("Ошибка (м)")
ax_xy.legend()
ax_xy.grid(True)

# Ошибка по ориентации (в градусах)
ax_theta.plot(np.rad2deg(error[:, 2]), label="Ошибка по ориентации (градусы)")
ax_theta.set_xlabel("Шаг времени")
ax_theta.set_ylabel("Ошибка (градусы)")
ax_theta.legend()
ax_theta.grid(True)

plt.tight_layout()
plt.show()
