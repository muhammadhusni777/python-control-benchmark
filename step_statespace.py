import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete, tf2ss

# ===== Model Motor DC =====
K = 0.5     # Gain motor
tau = 1.0   # Waktu respons
T = 1    # Periode sampling
num = [K]
den = [tau, 1]

# Diskritisasi dengan metode Tustin
num_d, den_d, _ = cont2discrete((num, den), T, method='bilinear')
A, B, C, D = tf2ss(num_d, den_d)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C:\n", C)
print("Matrix D:\n", D)

# ===== Simulasikan Sistem =====
N = 100  # Jumlah langkah simulasi
u = np.ones(N)  # Input step
x = np.zeros((A.shape[0], N))  # Keadaan awal
y = np.zeros(N)  # Output

for k in range(1, N):
    # Update state
    x[:, k] = A @ x[:, k - 1] + B.flatten() * 1
    # Hitung output
    y[k] = C @ x[:, k] + D * u[k]
    print(y[k])

# ===== Plot Hasil =====
t = np.arange(0, N * T, T)

plt.figure(figsize=(8, 5))
plt.step(t, u, where='post', label='Input (u)', linestyle='--')
plt.plot(t, y, label='Output (y)', linewidth=2)
plt.title('Simulasi Step Response Sistem Motor DC')
plt.xlabel('Waktu (detik)')
plt.ylabel('Amplitudo')
plt.grid(True)
plt.legend()
plt.show()

