#clase 9-2

import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

# system parameters
q = 2
om = 2/3 
T = 2*pi / om

# initial conditions
x0 = 1.25
v0 = 0
t = 0

# Gamma values
gamma_min = 0.9
gamma_max = 1.8
dgamma = 0.001
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)

# Numerical method parameters
nTrans = 300
nLyap = 600
steps_per_T = 300
dt = T/steps_per_T

# Initial state & unit tangent vector
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
xi = np.full(n_orbits, 1.0/np.sqrt(2))
eta = np.full(n_orbits, 1.0/np.sqrt(2))
y = np.concatenate((x, v, xi, eta))

# Dynamics
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:2*n_orbits]
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    dx = v
    dxi = eta
    dv = (-(1/q)*v - sin(x) + gamma_values * cos(om*t))
    deta = (-cos(x)*xi -(1/q)*eta)
    return np.concatenate([dx, dv, dxi, deta])

# Fourth-order Runge-Kutta method
def rk4(f,t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# Integration: transient regime
for period in range(nTrans):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    tangent_norm = np.sqrt(xi**2 + eta**2)
    y[2*n_orbits:3*n_orbits] = xi / tangent_norm
    y[3*n_orbits:] = eta / tangent_norm

# Maximum Lyapunov exponent 
sum_log = np.zeros(n_orbits)
for period in range(nLyap):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    tangent_norm = np.sqrt(xi**2 + eta**2)
    sum_log += np.log(tangent_norm)
    y[2*n_orbits:3*n_orbits] = xi / tangent_norm
    y[3*n_orbits:] = eta / tangent_norm

# Maximum Lyapunov exponent per unit time
lambda_max = sum_log / (nLyap * T)
imax = np.argmax(lambda_max)
gamma_at_max = gamma_values[imax]
lambda_at_max = lambda_max[imax]
print(f"Maximum Lyapunov exponent: {lambda_at_max:.6f} at gamma = {gamma_at_max:.6f}")

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(gamma_values, lambda_max, color='blue', linewidth=0.2)
plt.plot(gamma_at_max, lambda_at_max, 'ro', markersize=5)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--') 
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\lambda_{\max}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$\gamma = {gamma_at_max:.6f}$',
        transform=ax.transAxes,ha='left', fontsize=14, verticalalignment='top')
ax.set_xlim(gamma_values[0], gamma_values[-1])
ax.set_box_aspect(0.65)
plt.tight_layout()

# Save figure
plt.savefig('Lyapunov.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
