import numpy as np
import matplotlib.pyplot as plt

mu = 2.8
E = 1

# Dominio
r_max = 1 / (mu - 1)

r = np.linspace(0, r_max, 400)
dr = np.linspace(-0.8, 0.8, 400)

# Crear una malla de coordenadas
R, DR = np.meshgrid(r, dr)

# Función que determina la región permitida
F = DR**2 - (2 - 2 * R * (mu - 1)) / (1 + mu)

# Graficar la región
plt.figure(figsize=(8, 6))

# Sombrear la región donde F <= 0
plt.contourf(
    R, DR, F,
    levels=[-100, 0],
    colors=['lightblue'],
    alpha=0.8
)

# Dibujar las fronteras de la región
plt.contour(
    R, DR, F,
    levels=[0],
    colors=['blue'],
    linewidths=2
)


plt.xlabel(r'$r$', fontsize=14)
plt.ylabel(r'$\dot{r}$', fontsize=14)
plt.title(r'Región permitida en el espacio de fases ($\mu=2.8$)')

plt.tight_layout()

plt.savefig('region_fase_sombreada.png',dpi=300)

plt.show()
