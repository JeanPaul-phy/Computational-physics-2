# Stroboscopic map for the forced & damped Duffing oscillator

import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

# system parameters
q = 2
om = 2/3 
T = 2*pi / om
gamma = 1.1799 

# initial conditions
#x0, v0 = 1, 1
x0_values = np.linspace(-2, 2, 50)
v0_values = np.linspace(-pi, pi, 10)

X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()

n_orbits = len(x0_flat)
print(f"Numero de orbitas: {n_orbits}")

y = np.concatenate([x0_flat, v0_flat])

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Trans = 10          # periodos descartados (regimen transitorio)
Nperiods = 100
steps_per_T = 800
dt = T / steps_per_T

# -----------------------------------------------------
# Dynamics: forced & damped Duffing oscillator
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -(1/q)*v - sin(x) + gamma*cos(om*t)
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
wrap = lambda x: (x + pi) % (2*pi) - pi  # wrap x to [-pi, pi]
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
            x_strobe[save_index] = wrap(y[:n_orbits])
            v_strobe[save_index] = wrap(y[n_orbits:])
            save_index += 1

# ---------------------------------------------------------
# Stroboscopic map
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.scatter(x_strobe.ravel(), v_strobe.ravel(), s=0.5,
           color='blue', linewidths=0, rasterized=True, alpha=0.6)

ax.set_xlabel(r"$x$", fontsize=17)
ax.set_ylabel(r"$\dot{x}$", fontsize=17)
ax.tick_params(axis="both", labelsize=13)
ax.text(0.97, 0.97, rf'$\gamma = {gamma:.4f}$', 
        transform=ax.transAxes,ha='right', fontsize=14, verticalalignment='top')
ax.text(0.03, 0.97, rf'$q = {q:.1f}$', 
        transform=ax.transAxes,ha='left', fontsize=14, verticalalignment='top')
ax.text(0.03, 0.03, rf'$\omega_D = {om:.2f}$', 
        transform=ax.transAxes,ha='left', fontsize=14, verticalalignment='bottom')
ax.set_box_aspect(0.65)
plt.tight_layout()
#plt.savefig('mapa_estroboscopico_B.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
