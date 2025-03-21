import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Constants
hbar, m, L, N = 0.626, 1.0, 10.0, 300
x = np.linspace(-L/2, L/2, N)
dx = L / N
t_max, dt = 5.0, 0.0099
t_vals = np.arange(0, t_max, dt)

# Initial wave packet
x0, p0, sigma = -3.0, 2.0, 1.0
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x)

# Hamiltonian Matrix (Finite Difference)
laplacian = (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / dx**2
H = - (hbar**2 / (2 * m)) * laplacian

# TDSE as ODE system
def tdse_ode(t, psi_vec):
    psi = psi_vec[:N] + 1j * psi_vec[N:]
    dpsi_dt = -1j / hbar * H @ psi
    return np.concatenate([dpsi_dt.real, dpsi_dt.imag])

# Solve ODE
tdse_solution = solve_ivp(tdse_ode, (0, t_max), np.concatenate([psi0.real, psi0.imag]), t_eval=t_vals, method='RK45')
psi_t = tdse_solution.y[:N, :] + 1j * tdse_solution.y[N:, :]

# Extract components
psi_real, psi_imag = np.real(psi_t), np.imag(psi_t)
psi_abs2 = np.abs(psi_t) ** 2
T, X = np.meshgrid(tdse_solution.t, x)

# Set up figure with GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig)

# 3D Plot of Real & Imaginary Parts
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.plot_surface(X, T, psi_real, cmap='coolwarm', alpha=0.6)
ax1.plot_surface(X, T, psi_imag, cmap='plasma', alpha=0.6)
ax1.set_xlabel("Position x")
ax1.set_ylabel("Time t")
ax1.set_zlabel("Re & Im(ψ)")
ax1.set_title("Wave Function Evolution")

# 3D Plot of Probability Density
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax2.plot_surface(X, T, psi_abs2, cmap='viridis')
ax2.set_xlabel("Position x")
ax2.set_ylabel("Time t")
ax2.set_zlabel(r"$|ψ(x,t)|^2$")
ax2.set_title("Probability Density Evolution")

# 2D Plot for Real & Imaginary Parts
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(x.min(), x.max())
ax3.set_ylim(-1.2 * np.max(np.abs(psi_t)), 1.2 * np.max(np.abs(psi_t)))
ax3.set_xlabel("Position x")
ax3.set_ylabel("Wave Function")
ax3.set_title("Real and Imaginary Parts Over Time")
line_real, = ax3.plot([], [], lw=2, label="Re(ψ)")
line_imag, = ax3.plot([], [], lw=2, label="Im(ψ)")
ax3.legend()

def update_wave(frame):
    line_real.set_data(x, psi_real[:, frame])
    line_imag.set_data(x, psi_imag[:, frame])
    return line_real, line_imag

# 2D Animated Probability Density Plot
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(x.min(), x.max())
ax4.set_ylim(0, 1.2 * np.max(psi_abs2))
ax4.set_xlabel("Position x")
ax4.set_ylabel(r"$|ψ(x,t)|^2$")
ax4.set_title("Probability Density Over Time")
line_prob, = ax4.plot([], [], lw=2, color='gold')

def update_prob(frame):
    line_prob.set_data(x, psi_abs2[:, frame])
    return line_prob,

# Create animations
anim1 = animation.FuncAnimation(fig, update_wave, frames=len(tdse_solution.t), interval=50, blit=True)
anim2 = animation.FuncAnimation(fig, update_prob, frames=len(tdse_solution.t), interval=50, blit=True)

plt.tight_layout()
plt.show()
