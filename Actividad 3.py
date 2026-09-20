import numpy as np
from numpy import pi, sin, cos
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#-------------------------------------------------
# System parameters
#-------------------------------------------------
g = 1.0
m = 1.0
l = 1.0
F = 0.01
om = 2/pi
T = 2*pi/om          # período de la fuerza

#-------------------------------------------------
# Initial conditions
#-------------------------------------------------
theta0_list = np.linspace(-3.14, 3.14, 35)
theta_dot0_list = np.linspace(-2.2, 2.2, 15)

TH0, THD0 = np.meshgrid(theta0_list, theta_dot0_list)
theta0_flat = TH0.ravel()
theta_dot0_flat = THD0.ravel()
n_orbitas = len(theta0_flat)
print(f"Número de órbitas: {n_orbitas}")

y0 = np.concatenate([theta0_flat, theta_dot0_flat])

#-------------------------------------------------
# Method parameters
#-------------------------------------------------
N_PERIODOS = 3000
steps_per_T = 200          # dt = T/steps_per_T ≈ 0.001
dt = T / steps_per_T
n = N_PERIODOS * steps_per_T

#-------------------------------------------------
# Dynamics: forced pendulum
#-------------------------------------------------
def dyn(t, y):
    theta = y[:n_orbitas]
    theta_dot = y[n_orbitas:]
    dv = -(g/l) * sin(theta) + (F/(m*l)) * cos(om * t)
    return np.concatenate([theta_dot, dv])

#-------------------------------------------------
# Fourth-order Runge-Kutta method
#-------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

#-------------------------------------------------
# Integration + muestreo estroboscópico (t = kT)
#-------------------------------------------------
strobe = np.empty((N_PERIODOS + 1, 2*n_orbitas))
strobe[0] = y0
yc = y0.copy()
for i in range(n):
    yc = rk4(dyn, i*dt, yc, dt)
    if (i + 1) % steps_per_T == 0:
        strobe[(i + 1)//steps_per_T] = yc

x = (strobe[:, :n_orbitas] + pi) % (2*pi) - pi   # theta en [-pi, pi]
v = strobe[:, n_orbitas:]

#-------------------------------------------------
# Mapa estroboscópico
#-------------------------------------------------
c = np.broadcast_to(theta0_flat, x.shape)        # color según theta0

fig, ax = plt.subplots(figsize=(9, 6), dpi=800)
ax.scatter(x, v, s=0.4, c=c, cmap='turbo',linewidths=0.1, rasterized=True)
ax.set_xlim(-pi, pi)
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel(r"$\theta$", fontsize=22)
ax.set_ylabel(r"$\dot{\theta}$", fontsize=22)
ax.tick_params(axis='both', labelsize=20, direction='in', top=True, right=True)
plt.tight_layout()
ax.set_box_aspect(0.65)
plt.savefig('Dufpy.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
