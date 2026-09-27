# Stroboscopic map for the forced & damped Duffing oscillator

import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters (se reemplazan segun el ejercicio)
# -----------------------------------------------------
delta = 0.15      # amortiguamiento
alpha = 1        # coeficiente del termino lineal
beta = 1          # coeficiente del termino cubico (no linealidad)
gamma = 0.3         # amplitud de forzamiento
om = 1          # frecuencia de forzamiento
T = 2*pi / om     # periodo de forzamiento

# -----------------------------------------------------
# Initial conditions: grid (ensamble de orbitas para
# poblar el atractor mas rapido que con una sola orbita)
# -----------------------------------------------------
x0_values = np.linspace(1, 1.8, 40)
v0_values = np.linspace(-2, 2, 5)

X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()

n_orbits = len(x0_flat)
print(f"Numero de orbitas: {n_orbits}")

y = np.concatenate([x0_flat, v0_flat])

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Trans = 150           # periodos descartados (regimen transitorio)
Nperiods = 800
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Dynamics: forced & damped Duffing oscillator
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = gamma*cos(om*t) - delta*v + alpha*x - beta*x**3
    return np.concatenate([dx, dv])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------
# Stroboscopic storage
# -----------------------------------------------------
n_saved = Nperiods - Trans

x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))
save_index = 0

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
total_steps = Nperiods * steps_per_T

for step in range(total_steps):
    current_time = step * dt
    y = rk4(dyn, current_time, y, dt)

    if (step + 1) % steps_per_T == 0:
        completed_period = (step + 1) // steps_per_T
        if completed_period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# ---------------------------------------------------------
# Stroboscopic map
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.scatter(x_strobe.ravel(), v_strobe.ravel(), s=0.8,
           color='crimson', linewidths=0, rasterized=True, alpha=0.6)

ax.set_xlabel(r"$x$", fontsize=17)
ax.set_ylabel(r"$\dot{x}$", fontsize=17)
ax.tick_params(axis="both", labelsize=13)
ax.set_box_aspect(0.72)
plt.tight_layout()
#plt.savefig('mapa_estroboscopico2.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
