import numpy as np
from numpy import sin,cos 
import matplotlib
#in case the compiler doesn't have the right version of matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------
# System parameters
# ----------------------------------------------------
g = 9.81
k = 3.0
l = 1.0
m = 1.0
M = 4.0
 
# ----------------------------------------------------
# Method parameters
# ---------------------------------------------------- 
tmax = 20
dt = 0.005
STRIDE = 15

# ----------------------------------------------------
# Initial conditions
# ----------------------------------------------------
th20 = np.radians(30.0)
ome10 = 0
ome20 = 0
x0 = 0.3

# ----------------------------------------------------
# Non lineal dynamics: pendulum on a cart
# ----------------------------------------------------
def dyn(state):
    v, x, w, d = state
    a1 = ((g * cos(d) + l * w ** 2) * sin(d) - (k / m) * x) / ((M / m) + sin(d) ** 2)
    a2 = - (1 / l) * (g * (1.0 + M / m) * sin(d) + cos(d) * (l * w ** 2 * sin(d) - (k / m) * x)) / ((M / m) + sin(d) ** 2)
    return np.array([a1, v, a2, w])

def rk4(f, state, h):
    k1 = h * f(state)
    k2 = h * f(state + k1 / 2)
    k3 = h * f(state + k2 / 2)
    k4 = h * f(state + k3)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([ome10, x0, ome20, th20])

for idx in range(n):
    y[idx + 1] = rk4(dyn, y[idx], dt)

v = y[:, 0]
x = y[:, 1]
w = y[:, 2]
th = y[:, 3]

# ----------------------------------------------------------
# Kinematics
# ----------------------------------------------------------

cart_w, cart_h = 0.5, 0.3
margin = 0.3
offset = np.max(np.abs(x)) + cart_w / 2 + margin   # se adapta a la amplitud real

x_cart = x + offset          # posición del carro YA desplazada para dibujar
y_cart = np.zeros_like(x)

x_bob = x_cart + l * np.sin(th)   # el bob cuelga del carro desplazado
y_bob = -l * np.cos(th)

# ----------------------------------------------------------
# Figure
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x_max = x_cart.max() + l + 0.5
ax.set(xlim=(-0.5, x_max), ylim=(-l - 0.7, 1.5), aspect="equal",
       title="Péndulo en Carro (RK4)")
ax.title.set_fontsize(16)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)

ax.axhline(0, color='gray', linestyle='--', lw=1)

wall_x = 0.0
n_coils = 10
spring_amp = 0.12
end_len = 0.15

ax.plot([wall_x, wall_x], [-0.5, 0.4], color="dimgray", lw=5, zorder=5)
for i in range(5):
    ax.plot([wall_x - 0.12, wall_x], [-0.4 + i * 0.2, -0.4 + i * 0.2 - 0.12],
            color="dimgray", lw=1)

cart_poly, = ax.fill(
    [0, cart_w, cart_w, 0],
    [-cart_h / 2, -cart_h / 2, cart_h / 2, cart_h / 2],
    facecolor="royalblue", edgecolor="black", lw=1.5, zorder=3
)

spring_line, = ax.plot([], [], color="black", lw=1.6, zorder=2, solid_capstyle="round")
rod_line, = ax.plot([], [], "-", lw=2, color="black", zorder=2)
bob_marker, = ax.plot([], [], "o", markersize=16, color="red",
                       markeredgecolor="darkred", zorder=4, label="Masa (m)")
trace, = ax.plot([], [], "-", lw=1, alpha=0.3, color="red")
clock = ax.text(0.05, 0.90, "", transform=ax.transAxes, fontsize=12)


def make_spring(x_start, x_end, y=0.0, n_coils=10, amp=0.12, end_len=0.15):
    total_len = x_end - x_start
    coil_len = max(total_len - 2 * end_len, 0.05)
    n_points = 200
    x_coil = np.linspace(0, coil_len, n_points)
    y_coil = amp * np.sin(2 * np.pi * n_coils * x_coil / coil_len)
    xs = np.concatenate([
        [x_start, x_start + end_len],
        x_start + end_len + x_coil,
        [x_end - end_len, x_end]
    ])
    ys = np.concatenate([[0.0, 0.0], y_coil, [0.0, 0.0]]) + y
    return xs, ys


def animate(frame_idx):
    xc = x_cart[frame_idx]

    verts = [
        [xc - cart_w / 2, -cart_h / 2],
        [xc + cart_w / 2, -cart_h / 2],
        [xc + cart_w / 2, cart_h / 2],
        [xc - cart_w / 2, cart_h / 2],
    ]
    cart_poly.set_xy(verts)

    sx, sy = make_spring(wall_x, xc - cart_w / 2, y=0.0,
                          n_coils=n_coils, amp=spring_amp, end_len=end_len)
    spring_line.set_data(sx, sy)

    rod_line.set_data([xc, x_bob[frame_idx]], [y_cart[frame_idx], y_bob[frame_idx]])
    bob_marker.set_data([x_bob[frame_idx]], [y_bob[frame_idx]])
    trace.set_data(x_bob[:frame_idx + 1], y_bob[:frame_idx + 1])
    clock.set_text(f"t = {t[frame_idx]:.2f} s")
    return cart_poly, spring_line, rod_line, bob_marker, trace, clock


frames_seq = list(range(0, n + 1, STRIDE))
ani = FuncAnimation(fig, animate, frames=frames_seq,
                     interval=STRIDE * dt * 1000, blit=True)

plt.tight_layout()
plt.show()
