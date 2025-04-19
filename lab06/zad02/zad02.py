import math
import pygad

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

def fitness_func(pygad_class, solution, solution_idx):
    x = solution[0]
    y = solution[1]
    z = solution[2]
    u = solution[3]
    v = solution[4]
    w = solution[5]
    return endurance(x, y, z, u, v, w)

def run_ga():
    ga_instance = pygad.GA(
        gene_space= [{'low': 0.0, 'high': 1.0}] * 6,
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=6,
        parent_selection_type="sss",
        keep_parents=5,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,

    )
    
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Najlepsze rozwiązanie: {solution}")
    print(f"Najlepsza wartość funkcji przystosowania: {solution_fitness}")
    
    ga_instance.plot_fitness()

if __name__ == "__main__":
    run_ga()