import pandas as pd
import numpy as np
import pygad
import time
# Przykładowe dane
data = {'nazwa': ['zegar', 'obraz-pejzaz', 'obraz-portret', 'radio', 'laptop', 'lampka-nocna', 'srebrne sztucce', 'porcelana', 'figura z brazu', 'skurzana torebka', 'odkurzacz'], 
        'cena': [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300], 
        'waga': [7, 7, 6, 2, 5, 6, 1, 3, 10 ,3 ,15]}
df = pd.DataFrame(data)
# print(df)
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['nazwa'])
# print(df_encoded)

def fitness_func(pygadclass, solution, solution_idx):
    sum_value = np.sum(solution * df_encoded['cena'])
    sum_weight = np.sum(solution * df_encoded['waga'])
    if sum_weight > 25:
        return -1 
    else:
        return sum_value
    
gene_space = [0, 1]
sol_per_pop = 30
num_parents_mating = 5
num_generations = 50
keep_parents = 5
fitness_function = fitness_func
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10
num_genes = len(df_encoded.columns) - 2



# Eksperyment
successful_runs = 0
total_time = 0
num_experiments = 10

for i in range(num_experiments):
    # Tworzenie nowej instancji algorytmu
    ga_instance = pygad.GA(gene_space=gene_space,
                           num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           stop_criteria=["reach_1630"])
    
    # Mierzenie czasu działania
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    # Sprawdzenie, czy znaleziono najlepsze rozwiązanie
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness == 1630:
        successful_runs += 1
        total_time += (end_time - start_time)

# Wyniki
success_rate = (successful_runs / num_experiments) * 100
average_time = total_time / successful_runs if successful_runs > 0 else 0

print(f"Liczba udanych prób: {successful_runs}/{num_experiments}")
print(f"Procent udanych prób: {success_rate}%")
print(f"Średni czas działania dla udanych prób: {average_time:.2f} sekund")

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
# ga_instance = pygad.GA(gene_space=gene_space,
#                         num_generations=num_generations,
#                         num_parents_mating=num_parents_mating,
#                         fitness_func=fitness_function,
#                         sol_per_pop=sol_per_pop,
#                         num_genes=num_genes,
#                         parent_selection_type=parent_selection_type,
#                         keep_parents=keep_parents,
#                         crossover_type=crossover_type,
#                         mutation_type=mutation_type,
#                         mutation_percent_genes=mutation_percent_genes,
#                         stop_criteria=["reach_1630"])
# #uruchomienie algorytmu
# ga_instance.run()

# #podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# #tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
# prediction = np.sum(df["cena"]*solution)
# print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

# selected_items = df[solution == 1]
# print("Wybrane przedmioty:\n", selected_items)

# #TODO wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
# # ga_instance.plot_fitness()