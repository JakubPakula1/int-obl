import math
import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

def endurance(point):
    return (
        -(math.exp(-2 * (point[1] - math.sin(point[0])) ** 2)
        + math.sin(point[2] * point[3])
        + math.cos(point[4] * point[5]))
    )

def f(swarm):
    return np.array([endurance(particle) for particle in swarm])

x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

options = {'c1': 0.3, 'c2': 0.7, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

cost, pos = optimizer.optimize(f, iters=1000)

print(f"Najlepszy koszt: {cost}")
print(f"Najlepsza pozycja: {pos}")

plot_cost_history(optimizer.cost_history)
plt.title("Historia kosztów podczas optymalizacji")
plt.xlabel("Iteracje")
plt.ylabel("Koszt")
# plt.show()

#? Wysokie c1 i niskie c2 powodują, że cząstki poruszają się szybko w kierunku najlepszego rozwiązania, ale mogą nie eksplorować przestrzeni wystarczająco dobrze.
#? Niskie c1 i wysokie c2 powodują, że cząstki eksplorują przestrzeń bardziej, ale mogą nie koncentrować się na najlepszym rozwiązaniu.