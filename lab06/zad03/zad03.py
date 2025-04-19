import pygad
from LabirynthClass import Labirynth 

def fitness_func(pygad_instance, solution, solution_idx):
    global lab

    lab.reset_position()
    score = 0
    previous_distance = lab.calculate_distance()
    visited = set()

    for move in solution:
        pos_before = lab.get_position()
        # kara za ruch w ścianę
        if not lab.move(move):
            score -= 5  
            break

        pos_after = lab.get_position()
        current_distance = lab.calculate_distance()
        # kara za cofanie się w te same pola
        if pos_after in visited:
            score -= 5  
        else:
            visited.add(pos_after)

        if current_distance < previous_distance:
            score += 50  # nagroda za zbliżenie się
        elif current_distance > previous_distance:
            score -= 20  # kara za oddalenie się
        else:
            score -= 2  # neutralny ruch bez poprawy

        previous_distance = current_distance

        # Meta
        if pos_after == lab.get_finish():
            score = max(score, 1000)
            break

    # Dodatkowa kara jeśli nie osiągnął mety
    if lab.get_position() != lab.get_finish():
        score -= lab.calculate_distance() * 10

    return score


def on_generation(ga_instance):
    print(f"Pokolenie: {ga_instance.generations_completed}")
    print(f"Najlepszy wynik fitness: {ga_instance.best_solution()[1]}")


lab = Labirynth(10,10) 

print("Wygenerowany labirynt:")
lab.display_labirynth()

gene_space = [0, 1, 2, 3]  
chromosome_length = 30  

ga_instance = pygad.GA(
    gene_space=[0, 1, 2, 3],
    num_generations=100,
    num_parents_mating=100,
    fitness_func=fitness_func,
    sol_per_pop=500,
    num_genes=chromosome_length,
    parent_selection_type="sss",
    keep_parents=5,
    crossover_type="two_points",
    mutation_type="random",
    mutation_percent_genes=5,
    stop_criteria=["reach_1000"],
    on_generation=on_generation
)

ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Najlepsze rozwiązanie: {solution}")
print(f"Fitness najlepszego rozwiązania: {solution_fitness}")


lab.reset_position()

print("\nŚcieżka:")
for move in solution:
    if lab.get_position() == lab.get_finish():
        break 
    lab.move(move)  


lab.display_labirynth()

ga_instance.plot_fitness()