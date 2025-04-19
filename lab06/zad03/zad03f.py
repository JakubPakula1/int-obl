from LabirynthClass import Labirynth
import pygad
import time
import matplotlib.pyplot as plt  

def fitness_func(pygad_instance, solution, solution_idx):
    global lab

    lab.reset_position()
    score = 0
    previous_distance = lab.calculate_distance()
    visited = set()

    for move in solution:
        pos_before = lab.get_position()

        if not lab.move(move):
            score -= 5  # kara za ruch w ścianę
            break

        pos_after = lab.get_position()
        current_distance = lab.calculate_distance()

        if pos_after in visited:
            score -= 5  # kara za cofanie się w te same pola
        else:
            visited.add(pos_after)

        if current_distance < previous_distance:
            score += 50  # nagroda za zbliżenie się
        elif current_distance > previous_distance:
            score -= 20  # kara za oddalenie się
        else:
            score -= 2  # neutralny ruch bez poprawy

        previous_distance = current_distance

        if pos_after == lab.get_finish():
            score = max(score, 1000) 
            break

    if lab.get_position() != lab.get_finish():
        score -= lab.calculate_distance() * 10  

    return score

execution_times = []
successful_runs = 0  
fitness_history_all_runs = []  
success_flags = []  

experiment_start_time = time.time()

for i in range(10):
    lab = Labirynth(10, 10)
    print(f"\nUruchomienie {i + 1}:")
    lab.display_labirynth()
    gene_space = [0, 1, 2, 3]
    chromosome_length = 30

    start_time = time.time()

    fitness_history = []

    def on_generation(ga_instance):
        fitness_history.append(ga_instance.best_solution()[1])

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

    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Najlepsze rozwiązanie: {solution}")
    print(f"Fitness najlepszego rozwiązania: {solution_fitness}")
    print(f"Czas wykonania: {execution_time:.2f} sekund")

    success = solution_fitness >= 1000
    success_flags.append(success)
    if success:
        successful_runs += 1

    lab.reset_position()
    print("\nŚcieżka:")
    for move in solution:
        if move == 0:
            lab.move(0)
        elif move == 1:
            lab.move(1)
        elif move == 2:
            lab.move(2)
        elif move == 3:
            lab.move(3)

    lab.display_labirynth()

    fitness_history_all_runs.append(fitness_history)

average_time = sum(execution_times) / len(execution_times)
experiment_end_time = time.time()
total_experiment_time = experiment_end_time - experiment_start_time

print(f"\nŚredni czas wykonania: {average_time:.2f} sekund")
print(f"Liczba udanych przejść labiryntu: {successful_runs} / 10")
print(f"Całkowity czas eksperymentu: {total_experiment_time:.2f} sekund")

plt.figure(figsize=(15, 10))
for i, (fitness_history, success) in enumerate(zip(fitness_history_all_runs, success_flags)):
    plt.subplot(2, 5, i + 1)
    plt.plot(fitness_history, label=f"Run {i + 1}", color='green' if success else 'red')
    plt.title(f"Uruchomienie {i + 1} {'(Sukces)' if success else '(Niepowodzenie)'}")
    plt.xlabel("Pokolenie")
    plt.ylabel("Fitness")
    plt.legend()

plt.tight_layout()
plt.show()