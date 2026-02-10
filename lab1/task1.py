import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

# --- Параметры симуляции ---
T = 0.1  # Шаг дискретизации (с)
W_L = 0.05  # Радиус левого колеса (м)
W_R = 0.05  # Радиус правого колеса (м)
B = 0.2    # Расстояние от центра до колеса

# --- Параметры шума ---
# Шум процесса (неопределенность модели движения)
# Q = np.diag([0.01, 0.01, np.deg2rad(1.0)])**2
# Шум процесса РАВЕН НУЛЮ, если модель считается идеальной
Q = np.zeros((3, 3))

# Шум измерений (точность сенсоров)
R = np.diag([0.1, 0.1, np.deg2rad(5.0)])**2

# --- Инициализация ---
# Начальное истинное состояние [x, y, theta]
true_state = np.array([0, 0, 0.0])
# Начальная оценка состояния EKF (можем начать с первого измерения или с нуля)
estimated_state = np.array([0, 0, 0.0])
# Начальная ковариационная матрица (высокая неопределенность)
P = np.diag([1.0, 1.0, np.deg2rad(50.0)])**2

# Списки для хранения истории
history_true = []
history_measurements = []
history_estimated = []

# --- Генерация траектории и измерений ---
# Создадим траекторию в виде восьмерки
num_steps = 300
for i in range(num_steps):
    # Управление (угловые скорости колес) для создания восьмерки
    t = i * T
    u_L = 10.0 * np.sin(2 * t)
    u_R = 10.0 * np.cos(2 * t)

    # Сохраняем истинное состояние
    history_true.append(true_state.copy())

    # Генерируем зашумленные измерения
    w_x = np.random.normal(0, R[0, 0]**0.5)
    w_y = np.random.normal(0, R[1, 1]**0.5)
    w_r = np.random.normal(0, R[2, 2]**0.5)
    measurement = true_state + np.array([w_x, w_y, w_r])
    history_measurements.append(measurement)

    # --- Расширенный фильтр Калмана (EKF) ---

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
        estimated_state[2] + T * s_r
    ])

    # Вычисление матрицы Якоби F
    F = np.array([
        [1, 0, -T * s_t * np.sin(r) - 0.5 * T**2 * s_t * s_r * np.cos(r)],
        [0, 1,  T * s_t * np.cos(r) - 0.5 * T**2 * s_t * s_r * np.sin(r)],
        [0, 0, 1]
    ])

    # Предсказание ковариации
    P_pred = F @ P @ F.T + Q

    # ЭТАП КОРРЕКЦИИ

    # Усиление Калмана
    # H - единичная матрица, поэтому S = P_pred + R
    S = P_pred + R
    K = P_pred @ np.linalg.inv(S)

    # Обновление состояния с помощью измерения
    estimated_state = state_pred + K @ (measurement - state_pred)

    # Обновление ковариации
    # I - K*H = I - K
    P = (np.eye(3) - K) @ P_pred

    # Сохраняем оценку
    history_estimated.append(estimated_state.copy())

    # --- Обновление истинного состояния для следующего шага ---
    true_state = np.array([
        true_state[0] + T * s_t * np.cos(true_state[2]) - 0.5 * T**2 * s_t * s_r * np.sin(true_state[2]),
        true_state[1] + T * s_t * np.sin(true_state[2]) + 0.5 * T**2 * s_t * s_r * np.cos(true_state[2]),
        true_state[2] + T * s_r
    ])


# --- 6. Визуализация результатов ---
history_true = np.array(history_true)
history_measurements = np.array(history_measurements)
history_estimated = np.array(history_estimated)

plt.figure(figsize=(12, 8))
plt.plot(history_true[:, 0], history_true[:, 1], 'g-', label='Истинная траектория')
plt.plot(history_measurements[:, 0], history_measurements[:, 1], 'k.', label='Зашумленные измерения', markersize=4, alpha=0.7)
plt.plot(history_estimated[:, 0], history_estimated[:, 1], 'b-', label='Траектория, оцененная EKF')
plt.title('Отслеживание положения мобильного робота с помощью EKF')
plt.xlabel('Координата X (м)')
plt.ylabel('Координата Y (м)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Визуализация ошибки
error = history_estimated - history_true
plt.figure(figsize=(12, 6))
plt.plot(error[:, 0], label='Ошибка по X')
plt.plot(error[:, 1], label='Ошибка по Y')
plt.plot(np.rad2deg(error[:, 2]), label='Ошибка по ориентации (градусы)')
plt.title('Ошибка оценки EKF')
plt.xlabel('Шаг времени')
plt.ylabel('Ошибка')
plt.legend()
plt.grid(True)
plt.show()