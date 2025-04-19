import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


COORDS_1 = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
)

COORDS_2= (
    (20, 52), (43, 50), (20, 84), (70, 65), (29, 90), (87, 83), (73, 23),
    (10, 10), (50, 30), (60, 70), (80, 20), (90, 90), (15, 45), (25, 75),
    (35, 55), (45, 85), (55, 25), (65, 95), (75, 35), (85, 15)
)

COORDS = (
    (20, 52), (43, 50), (20, 84), (70, 65), (29, 90), (87, 83), (73, 23),
    (10, 10), (50, 30), (60, 70), (80, 20), (90, 90), (15, 45), (25, 75),
    (35, 55), (45, 85), (55, 25), (65, 95), (75, 35), (85, 15),
    (5, 5), (95, 95), (30, 60), (60, 30), (40, 40), (70, 70), (20, 20),
    (80, 80), (10, 90), (90, 10), (25, 25), (75, 75), (35, 15), (65, 85),
)
def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
#                     pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
#                     iterations=300)

# colony = AntColony(COORDS, ant_count=300, alpha=1.0, beta=1.0, 
#                     pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
#                     iterations=300)
# colony = AntColony(COORDS, ant_count=600, alpha=1.0, beta=1.0, 
#                     pheromone_evaporation_rate=0.50, pheromone_constant=500.0,
#                     iterations=200)
colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
                    pheromone_evaporation_rate=0.60, pheromone_constant=1000.0,
                    iterations=300)
optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()

# Wnioski:
# 1. Zwiększenie liczby mrówek poprawia jakość rozwiązania, ale wydłuża czas działania.
# 2. Wyższe wartości alpha powodują szybszą konwergencję, ale mogą prowadzić do lokalnych minimów.
# 3. Wyższe wartości beta poprawiają jakość rozwiązania, ale mogą spowolnić eksplorację.
# 4. Szybsze parowanie feromonów poprawia eksplorację, ale może prowadzić do wolniejszej konwergencji.