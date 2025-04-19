import pandas as pd
import numpy as np
import pygad
import time

def prepare_data():
    data = {
        'nazwa': ['zegar', 'obraz-pejzaz', 'obraz-portret', 'radio', 'laptop', 'lampka-nocna', 'srebrne sztucce', 'porcelana', 'figura z brazu', 'skorzana torebka', 'odkurzacz'], 
        'cena': [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300], 
        'waga': [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
    }
    df = pd.DataFrame(data)
    df_encoded = pd.get_dummies(df, columns=['nazwa'])
    return df, df_encoded

def fitness_func(pygadclass, solution, solution_idx):
    sum_value = np.sum(solution * df_encoded['cena'])
    sum_weight = np.sum(solution * df_encoded['waga'])
    if sum_weight > 25:
        return -1 
    else:
        return sum_value

def run_experiment(num_experiments, stop_criteria, df_encoded):
    successful_runs = 0
    total_time = 0

    for i in range(num_experiments):
        ga_instance = pygad.GA(
            gene_space=[0, 1],
            num_generations=50,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=30,
            num_genes=len(df_encoded.columns) - 2,
            parent_selection_type="sss",
            keep_parents=5,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            stop_criteria=stop_criteria
        )
        
        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        if solution_fitness == 1630:
            successful_runs += 1
            total_time += (end_time - start_time)

    return successful_runs, total_time

def display_results(successful_runs, total_time, num_experiments):
    success_rate = (successful_runs / num_experiments) * 100
    average_time = total_time / successful_runs if successful_runs > 0 else 0

    print(f"Liczba udanych prób: {successful_runs}/{num_experiments}")
    print(f"Procent udanych prób: {success_rate}%")
    print(f"Średni czas działania dla udanych prób: {average_time:.2f} sekund")

def run_once_and_display(df, df_encoded):
    ga_instance = pygad.GA(
        gene_space=[0, 1],
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=len(df_encoded.columns) - 2,
        parent_selection_type="sss",
        keep_parents=5,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria=["reach_1630"]
    )
    
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Wybrane przedmioty
    selected_items = df[np.array(solution) == 1]
    total_value = np.sum(selected_items['cena'])
    total_weight = np.sum(selected_items['waga'])

    print("Wybrane przedmioty:")
    print(selected_items)
    print(f"Całkowita wartość: {total_value}")
    print(f"Całkowita waga: {total_weight}")

    
    ga_instance.plot_fitness()


if __name__ == "__main__":
    df, df_encoded = prepare_data()
    stop_criteria = ["reach_1630"]
    num_experiments = 10


    successful_runs, total_time = run_experiment(num_experiments, stop_criteria, df_encoded)
    display_results(successful_runs, total_time, num_experiments)


    run_once_and_display(df, df_encoded)