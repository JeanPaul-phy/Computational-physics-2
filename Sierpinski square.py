import numpy as np
import matplotlib.pyplot as plt

# vertice of the square
V = np.array([[0, 0], [0.5, 0], [1,0], [0, 0.5], [1, 0.5], [0, 1], [0.5, 1], [1, 1]])
vertices = len(V)  # number of vertices
# initial point
P = np.array([0.5, 0.5])

# Iterations and array
N = 1000000
x = np.zeros(N)
y = np.zeros(N)

# Chaos game algorithm
for i in range(N):
    j = np.random.randint(0, 8)  # choose a random vertex
    P = (P + 2*V[j]) / 3  # move halfway towards the chosen vertex
    x[i] = P[0]
    y[i] = P[1]  # store the new point
Dim = np.log(vertices)/np.log(3)  # calculate the dimension of the fractal

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x, y, s=0.1, color='blue')  # plot the points    
ax.text(0.03, 0.99, rf'$\bf{{N}} = {N}$', transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')  # add N value to the plot
ax.text(0.6, 0.99, rf'$\bf{{Dim}} = {Dim:.2f}$', transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', color='red')  # add dimension value to the plot
ax.tick_params(axis='both', labelsize=14)
ax.set_aspect('equal')
plt.tight_layout()

plt.savefig('sierpinski_square.png', dpi=800, bbox_inches='tight')  # save the figure
plt.show()  # remove ticks and labels
