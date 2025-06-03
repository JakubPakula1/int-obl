import gymnasium as gym
import numpy as np
import pygad
import time

# Utwórz środowisko FrozenLake bez poślizgów
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Parametry chromosomu
CHROMOSOME_LENGTH = 100  # Długość sekwencji ruchów
POPULATION_SIZE = 50
NUM_GENERATIONS = 100

def fitness_function(ga_instance, solution, solution_idx):
    """
    Funkcja fitness dla algorytmu genetycznego.
    Każdy chromosom to sekwencja ruchów [0,1,2,3] (lewo, dół, prawo, góra).
    Oceniamy jak daleko agent przeszedł i czy dotarł do celu.
    """
    observation, info = env.reset(seed=0)
    total_reward = 0
    steps_taken = 0
    max_steps = len(solution)
    
    for action in solution:
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps_taken += 1
        
        # Jeśli dotarliśmy do celu, dajemy bonus
        if terminated and reward > 0:
            # Bonus za szybkie dotarcie do celu
            bonus = (max_steps - steps_taken) / max_steps
            total_reward += bonus
            break
        
        # Jeśli wpadliśmy w dziurę, kończymy z karą
        if terminated and reward == 0:
            total_reward -= 0.5
            break
    
    return total_reward

def on_generation(ga_instance):
    """Callback wywoływany po każdej generacji"""
    print(f"Generacja {ga_instance.generations_completed}")
    print(f"Najlepsze rozwiązanie: {ga_instance.best_solution()[1]}")

# Konfiguracja algorytmu genetycznego
ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=10,
    fitness_func=fitness_function,
    sol_per_pop=POPULATION_SIZE,
    num_genes=CHROMOSOME_LENGTH,
    gene_type=int,
    gene_space=[0, 1, 2, 3],  # Możliwe ruchy: lewo, dół, prawo, góra
    mutation_type="random",
    mutation_percent_genes=10,
    on_generation=on_generation
)

# Uruchom algorytm genetyczny
print("Rozpoczynam trening algorytmu genetycznego...")
ga_instance.run()

# Pokaż najlepsze rozwiązanie
best_solution, best_fitness, _ = ga_instance.best_solution()
print(f"\nNajlepsze rozwiązanie znalezione:")
print(f"Sekwencja ruchów: {best_solution}")
print(f"Wartość funkcji fitness: {best_fitness}")

# Pokaż działanie najlepszego rozwiązania
print("\nDemonstracja najlepszego rozwiązania:")
observation, info = env.reset(seed=0)
total_reward = 0

for action in best_solution:
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.5)  # Pauza dla lepszej wizualizacji
    
    if terminated or truncated:
        break

print(f"Końcowa suma nagród: {total_reward}")
env.close() 