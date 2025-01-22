import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete, tf2ss
import matplotlib.pyplot as plt

# ===== Model Motor DC =====
K = 0.5     # Gain motor
tau = 1.0   # Waktu respons
T = 0.05    # Periode sampling
num = [K]
den = [tau, 1]

# Diskritisasi dengan metode Tustin
num_d, den_d, _ = cont2discrete((num, den), T, method='bilinear')
A, B, C, D = tf2ss(num_d, den_d)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C:\n", C)
print("Matrix D:\n", D)

# ===== Parameter MPC =====
N = 20                  # Horizon waktu (prediksi)
Q = 10000                  # Penalti error
R = 0.1                  # Penalti kontrol
delta_u_penalty = 100   # Penalti perubahan kontrol
u_min, u_max = -200, 200   # Batas kontrol PWM
y_refs = [30] * 100 + [100] * 150 + [-75] * 200  # Referensi berubah di iterasi ke-200

# ===== Variabel Simulasi =====
x0 = np.zeros((A.shape[0], 1))  # Status awal
predicted_states = []           # Menyimpan status sistem
applied_inputs = []             # Menyimpan kontrol yang diterapkan
time_steps = []

try:
    for step in range(len(y_refs)):  # Simulasi hingga panjang referensi
        y_ref = y_refs[step]  # Ambil referensi untuk langkah ini

        # ===== Variabel Optimisasi =====
        x = cp.Variable((A.shape[0], N+1))  # Status sistem
        u = cp.Variable((1, N))             # Input kontrol

        # ===== Fungsi Biaya dan Kendala =====
        cost = 0
        constraints = []

        for k in range(N):
            cost += Q * (C @ x[:, k] - y_ref)**2 + R * u[:, k]**2
            if k > 0:
                cost += delta_u_penalty * cp.norm(u[:, k] - u[:, k-1])**2  # Penalti perubahan kontrol
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
            constraints += [u_min <= u[:, k], u[:, k] <= u_max]

        # Status awal
        constraints += [x[:, 0] == x0.flatten()]

        # Problem MPC
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # ===== Ambil Kontrol Optimal =====
        if problem.status != 'optimal':
            print(f"Solver failed at step {step}. Status: {problem.status}")
            break

        u_optimal = u.value[0, 0]

        # ===== Simulasikan Sistem =====
        x0 = A @ x0 + B * u_optimal
        y = C @ x0

        # Simpan Hasil
        predicted_states.append(y.flatten()[0])  # Output sistem
        applied_inputs.append(u_optimal)        # Kontrol diterapkan
        time_steps.append(step * T)

        print(f"Setpoint: {y_ref}, Sensor: {y.flatten()[0]}, Input: {u_optimal}")

except KeyboardInterrupt:
    print("Simulation interrupted.")

# ===== Plot Hasil Simulasi =====
plt.figure(figsize=(12, 6))
plt.plot(time_steps, predicted_states, label="Sensor (System Output)")
plt.plot(time_steps, y_refs[:len(time_steps)], label="Setpoint", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Sensor Value")
plt.title("MPC Simulation: System Response vs Setpoint")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(time_steps, applied_inputs, label="Control Input (PWM)")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (u)")
plt.title("MPC Simulation: Control Input")
plt.legend()
plt.grid()
plt.show()

