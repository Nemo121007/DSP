import numpy as np
import matplotlib.pyplot as plt

# --- 1. Вспомогательные функции ---

def normalize_angle(angle):
    """Приведение угла к диапазону [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def motion_model(state, odometry):
    """
    Модель движения (формула 29, f(x, q=0)).
    state: [x, y, theta]
    odometry: [delta_r1, delta_trans, delta_r2]
    """
    x, y, theta = state
    delta_r1, delta_trans, delta_r2 = odometry

    x_new = x + delta_trans * np.cos(theta + delta_r1)
    y_new = y + delta_trans * np.sin(theta + delta_r1)
    theta_new = theta + delta_r1 + delta_r2

    return np.array([x_new, y_new, normalize_angle(theta_new)])

def measurement_model(state, landmarks):
    """
    Модель измерений (формула 30, h(x, r=0)).
    Возвращает вектор расстояний до всех ориентиров.
    state: [x, y, theta]
    landmarks: список координат [[mx1, my1], [mx2, my2], ...]
    """
    x, y, _ = state
    predictions = []
    for lm in landmarks:
        dist = np.sqrt((lm[0] - x)**2 + (lm[1] - y)**2)
        predictions.append(dist)
    return np.array(predictions)

# --- 2. Расширенный фильтр Калмана (EKF) ---

class EKF:
    def __init__(self, x0, P0, Q, R):
        self.m = x0  # m_k (оценка состояния)
        self.P = P0  # P_k (ковариация)
        self.Q = Q   # Ковариация шума процесса
        self.R = R   # Ковариация шума измерений (скаляр или матрица)

    def predict(self, odometry):
        """
        Шаг предсказания (формулы 31, 32).
        """
        # 1. Предсказание среднего (31)
        self.m = motion_model(self.m, odometry)

        # 2. Вычисление Якобиана F_x (25)
        x, y, theta = self.m
        delta_r1, delta_trans, delta_r2 = odometry

        # Производные по theta (по x и y равны 0 или 1)
        # f1 = x + dt * cos(th + dr1) -> d/dth = -dt * sin(th + dr1)
        # f2 = y + dt * sin(th + dr1) -> d/dth =  dt * cos(th + dr1)
        # f3 = th + ...

        Fx = np.array([
            [1, 0, -delta_trans * np.sin(theta + delta_r1)],
            [0, 1,  delta_trans * np.cos(theta + delta_r1)],
            [0, 0, 1]
        ])

        # 3. Предсказание ковариации (32)
        self.P = Fx @ self.P @ Fx.T + self.Q

    def update(self, observed_landmarks, measurements):
        """
        Шаг коррекции (формулы 33-36).
        measurements: вектор измеренных расстояний z_k.
        observed_landmarks: координаты наблюдаемых ориентиров.
        """
        if len(measurements) == 0:
            return

        # Вектор предсказанных измерений
        z_pred = measurement_model(self.m, observed_landmarks)

        # Размерность вектора измерений
        n_meas = len(measurements)

        # Якобиан H_x будет размером [n_meas, 3]
        Hx = np.zeros((n_meas, 3))

        x, y, _ = self.m

        for i, lm in enumerate(observed_landmarks):
            mx, my = lm
            dx = mx - x
            dy = my - y
            q = dx**2 + dy**2
            dist = np.sqrt(q)

            # Производные расстояния по x, y, theta
            # h = sqrt((mx-x)^2 + (my-y)^2)
            # dh/dx = -0.5 * 2(mx-x) / h = -dx/h
            # dh/dy = -dy/h
            # dh/dtheta = 0

            Hx[i, 0] = -dx / dist
            Hx[i, 1] = -dy / dist
            Hx[i, 2] = 0

        # Невязка (Innovation)
        y_k = measurements - z_pred

        # Ковариация невязки (33)
        # R_k должна быть матрицей n_meas x n_meas
        R_k = np.eye(n_meas) * self.R

        S_k = Hx @ self.P @ Hx.T + R_k

        # Коэффициент усиления Калмана (34)
        K_k = self.P @ Hx.T @ np.linalg.inv(S_k)

        # Обновление среднего (35)
        self.m = self.m + K_k @ y_k

        # Нормализация угла
        self.m[2] = normalize_angle(self.m[2])

        # Обновление ковариации (36)
        # Форма Джозефа (P = (I - KH)P(I - KH)' + KRK') более стабильна,
        # но в справочных материалах дана простая форма:
        self.P = self.P - K_k @ S_k @ K_k.T

# --- 3. Сигма-точечный фильтр Калмана (UKF) ---

class UKF:
    def __init__(self, x0, P0, Q, R, alpha=1e-3, beta=2, kappa=0):
        self.m = x0
        self.P = P0
        self.Q = Q
        self.R = R

        self.n = len(x0) # Размерность состояния (3)
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

        # Веса (формулы под номером без ссылки, но из текста)
        self.W_m = np.zeros(2 * self.n + 1)
        self.W_c = np.zeros(2 * self.n + 1)

        self.W_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.W_c[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta**2)

        self.W_m[1:] = 1 / (2 * (self.n + self.lambda_))
        self.W_c[1:] = 1 / (2 * (self.n + self.lambda_))

    def sigma_points(self, m, P):
        """
        Формирование сигма-точек (формулы 47-49).
        """
        n = self.n
        sigma_points = np.zeros((n, 2*n + 1))

        # Разложение Холецкого: P = sqrt(P) * sqrt(P).T
        # np.linalg.cholesky возвращает нижнетреугольную L: P = L L.T
        # sqrt(P) в формуле - это L. В формуле sqrt(P)[i] - это i-й столбец.
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Добавляем небольшую регуляризацию, если матрица вырождена
            P = P + 1e-6 * np.eye(n)
            L = np.linalg.cholesky(P)

        # X(0)
        sigma_points[:, 0] = m

        # X(i) и X(i+n)
        gamma = np.sqrt(n + self.lambda_)

        for i in range(n):
            # column i of L
            col = L[:, i]
            sigma_points[:, i+1] = m + gamma * col
            sigma_points[:, i+1+n] = m - gamma * col

        return sigma_points

    def predict(self, odometry):
        """
        Шаг предсказания (формулы 50-52).
        """
        # 1. Генерация точек (47-49)
        X = self.sigma_points(self.m, self.P)

        # 2. Пропуск через модель движения (50)
        X_pred = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_pred[:, i] = motion_model(X[:, i], odometry)

        # 3. Вычисление среднего (51)
        self.m = np.sum(self.W_m * X_pred, axis=1)

        # 4. Вычисление ковариации (52)
        self.P = np.zeros((self.n, self.n))
        for i in range(X_pred.shape[1]):
            diff = X_pred[:, i] - self.m
            # Для углов theta разницу нужно нормализовать!
            diff[2] = normalize_angle(diff[2])

            self.P += self.W_c[i] * np.outer(diff, diff)

        self.P += self.Q

        # Нормализация угла в среднем (опционально, но полезно)
        self.m[2] = normalize_angle(self.m[2])

    def update(self, observed_landmarks, measurements):
        """
        Шаг коррекции (формулы 53-62).
        """
        if len(measurements) == 0:
            return

        # 1. Генерация точек из предсказанного распределения (53-55)
        X = self.sigma_points(self.m, self.P)

        # 2. Пропуск через модель измерений (56)
        # Y должен быть размером [num_meas, 2n+1]
        n_meas = len(measurements)
        Y = np.zeros((n_meas, X.shape[1]))

        for i in range(X.shape[1]):
            Y[:, i] = measurement_model(X[:, i], observed_landmarks)

        # 3. Вычисление параметров измерения (57-59)
        # Среднее (57)
        mu = np.sum(self.W_m * Y, axis=1)

        # Ковариация S (58) и Взаимная ковариация C (59)
        S = np.zeros((n_meas, n_meas))
        C = np.zeros((self.n, n_meas))

        for i in range(X.shape[1]):
            y_diff = Y[:, i] - mu
            x_diff = X[:, i] - self.m
            x_diff[2] = normalize_angle(x_diff[2]) # Нормализация угла в разности

            S += self.W_c[i] * np.outer(y_diff, y_diff)
            C += self.W_c[i] * np.outer(x_diff, y_diff)

        S += np.eye(n_meas) * self.R

        # 4. Обновление состояния (60-62)
        K = C @ np.linalg.inv(S) # (60)

        self.m = self.m + K @ (measurements - mu) # (61)
        self.m[2] = normalize_angle(self.m[2])

        self.P = self.P - K @ S @ K.T # (62)

# --- 4. Генерация данных и Запуск ---

def generate_data(steps=100):
    # Генерация карты ориентиров
    landmarks = np.array([
        [10, 0],
        [10, 10],
        [0, 10],
        [-5, 5]
    ])

    # Истинная траектория (круг или змейка)
    true_states = []
    odometry_data = []
    sensor_data = []

    # Начальное положение
    state = np.array([0, 0, 0])

    for t in range(steps):
        # Сохраняем истинное состояние
        true_states.append(state.copy())

        # Генерация управления (небольшое движение вперед с поворотом)
        # Формат: [delta_r1, delta_trans, delta_r2]
        u = np.array([0.05, 0.5, 0.0])
        if t > steps/2: u = np.array([-0.1, 0.5, 0.0])

        # Истинное движение (без шума для истинного пути, или с малым)
        next_state = motion_model(state, u)

        # Генерация зашумленной одометрии (то, что "измерил" робот)
        # В задании сказано: одометрия задана, мы её используем как вход.
        # Но для симуляции добавим шум к u, чтобы увидеть работу фильтра
        u_noisy = u + np.random.multivariate_normal([0,0,0], np.diag([0.05, 0.1, 0.05]))

        odometry_data.append(u_noisy)

        # Генерация измерений
        current_measurements = []
        current_observed_lm = []
        for lm in landmarks:
            dist = np.sqrt((lm[0] - next_state[0])**2 + (lm[1] - next_state[1])**2)
            # Добавляем шум измерения (дисперсия 0.2)
            dist_noisy = dist + np.random.normal(0, np.sqrt(0.2))
            current_measurements.append(dist_noisy)
            current_observed_lm.append(lm)

        sensor_data.append((current_observed_lm, np.array(current_measurements)))

        state = next_state

    return np.array(true_states), landmarks, odometry_data, sensor_data

def run_filters():
    # Параметры из задачи
    # Дисперсии 0.2
    Q = np.diag([0.2, 0.2, 0.2])
    R = 0.2

    # Начальные условия
    m0 = np.array([0.0, 0.0, 0.0])
    P0 = np.diag([0.1, 0.1, 0.1])

    # Генерация данных
    true_path, landmarks, odometry_data, sensor_data = generate_data(50)

    # Инициализация фильтров
    ekf = EKF(m0.copy(), P0.copy(), Q, R)
    ukf = UKF(m0.copy(), P0.copy(), Q, R, alpha=0.001, beta=2, kappa=0)

    ekf_path = []
    ukf_path = []

    # Основной цикл
    for k in range(len(odometry_data)):
        u_k = odometry_data[k]
        lms, z_k = sensor_data[k]

        # --- EKF ---
        ekf.predict(u_k)
        ekf.update(lms, z_k)
        ekf_path.append(ekf.m.copy())

        # --- UKF ---
        ukf.predict(u_k)
        ukf.update(lms, z_k)
        ukf_path.append(ukf.m.copy())

    ekf_path = np.array(ekf_path)
    ukf_path = np.array(ukf_path)

    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.plot(true_path[:, 0], true_path[:, 1], 'k-', label='Истинный путь', linewidth=2)
    plt.plot(ekf_path[:, 0], ekf_path[:, 1], 'r--', label='EKF')
    plt.plot(ukf_path[:, 0], ukf_path[:, 1], 'g-.', label='UKF')

    # Рисуем ориентиры
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='b', marker='^', s=100, label='Ориентиры')

    plt.legend()
    plt.grid(True)
    plt.title('Сравнение EKF и UKF')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_filters()