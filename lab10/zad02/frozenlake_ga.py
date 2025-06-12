import gymnasium as gym
import numpy as np
import pygad
import time

# Utwórz środowisko FrozenLake 8x8 bez poślizgów
env = gym.make("FrozenLake8x8-v1", is_slippery=False)
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Parametry chromosomu
CHROMOSOME_LENGTH = 30  # Maksymalna długość sekwencji ruchów
POPULATION_SIZE = 200 # Większa populacja dla lepszej eksploracji
NUM_GENERATIONS = 300    # Więcej generacji dla złożonego problemu

def fitness_function(ga_instance, solution, solution_idx):
    """
    Funkcja fitness dla algorytmu genetycznego w FrozenLake8x8.
    
    CHROMOSOM: Lista liczb całkowitych [0,1,2,3] reprezentujących ruchy:
    - 0: LEFT (lewo)
    - 1: DOWN (dół) 
    - 2: RIGHT (prawo)
    - 3: UP (góra)
    
    FUNKCJA FITNESS:
    1. Symuluje grę wykonując ruchy z chromosomu
    2. Ocenia pozycję końcową agenta na mapie 8x8
    3. Przyznaje punkty za:
       - Dotarcie do celu (G) = +100 punktów + bonus za szybkość
       - Odległość od celu = maksymalne punkty za bycie blisko
       - Kara za wpadnięcie w dziurę (H) = -50 punktów
    4. Używa odległości Manhattan do obliczenia bliskości celu
    """
    observation, info = env.reset(seed=42)  # Stały seed dla reprodukowalności
    total_reward = 0
    steps_taken = 0
    max_steps = len(solution)
    
    # Pozycja celu w FrozenLake 8x8 (prawy dolny róg)
    goal_position = (7, 7)  # Wiersz 7, kolumna 7
    
    for action in solution:
        prev_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)
        steps_taken += 1
        
        # Jeśli dotarliśmy do celu
        if terminated and reward > 0:
            # Duża nagroda za sukces
            total_reward += 100
            # Bonus za szybkie dotarcie (mniej kroków = więcej punktów)
            speed_bonus = (max_steps - steps_taken) / max_steps * 50
            total_reward += speed_bonus
            print(f"Sukces! Dotarł do celu w {steps_taken} krokach")
            break
        
        # Jeśli wpadliśmy w dziurę
        if terminated and reward == 0:
            total_reward -= 50  # Kara za wpadnięcie w dziurę
            break
        
        # Nagroda za postęp w kierunku celu (na podstawie pozycji)
        # Konwertuj numer pola na współrzędne (x, y)
        current_pos = (observation // 8, observation % 8)
        
        # Oblicz odległość Manhattan od celu
        distance_to_goal = abs(current_pos[0] - goal_position[0]) + abs(current_pos[1] - goal_position[1])
        
        # Im bliżej celu, tym więcej punktów (maksymalnie 14 punktów za bycie w celu)
        proximity_reward = (14 - distance_to_goal) * 2
        total_reward += proximity_reward
        
        # Dodatkowa nagroda za ruch w dobrym kierunku
        prev_pos = (prev_observation // 8, prev_observation % 8)
        prev_distance = abs(prev_pos[0] - goal_position[0]) + abs(prev_pos[1] - goal_position[1])
        
        if distance_to_goal < prev_distance:
            total_reward += 5  # Bonus za zbliżenie się do celu
        elif distance_to_goal > prev_distance:
            total_reward -= 2  # Mała kara za oddalenie się
    
    return max(0, total_reward)  # Fitness nie może być ujemny

def on_generation(ga_instance):
    """Callback wywoływany po każdej generacji - monitoring postępu"""
    best_fitness = ga_instance.best_solution()[1]
    avg_fitness = np.mean(ga_instance.last_generation_fitness)
    print(f"Generacja {ga_instance.generations_completed}: Najlepszy={best_fitness:.2f}, Średni={avg_fitness:.2f}")

# Konfiguracja algorytmu genetycznego
ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=20,  # Więcej rodziców dla większej różnorodności
    fitness_func=fitness_function,
    sol_per_pop=POPULATION_SIZE,
    num_genes=CHROMOSOME_LENGTH,
    gene_type=int,
    gene_space=[0, 1, 2, 3],  # Możliwe ruchy
    mutation_type="random",
    mutation_percent_genes=15,  # Wyższa mutacja dla eksploracji
    crossover_type="single_point",
    parent_selection_type="tournament",
    on_generation=on_generation,
    random_seed=42  # Dla reprodukowalności wyników
)

# Uruchom algorytm genetyczny
print("Rozpoczynam trening algorytmu genetycznego dla FrozenLake 8x8...")
start_time = time.time()
ga_instance.run()
end_time = time.time()

# Pokaż najlepsze rozwiązanie
best_solution, best_fitness, _ = ga_instance.best_solution()
print(f"\nTrening zakończony w {end_time - start_time:.2f} sekund")
print(f"Najlepsze rozwiązanie:")
print(f"Wartość funkcji fitness: {best_fitness:.2f}")

env.close()
# Wizualna demonstracja najlepszego rozwiązania z render_mode="human"
print("\n" + "="*50)
print("WIZUALNA DEMONSTRACJA NAJLEPSZEGO ROZWIĄZANIA")
print("="*50)
move_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
# Utwórz nowe środowisko z trybem wizualnym
visual_env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode="human")

observation, info = visual_env.reset(seed=42)
total_reward = 0
step_count = 0

print("Naciśnij Enter, aby rozpocząć wizualną demonstrację...")
input()

for action in best_solution:
    print(f"\nKrok {step_count + 1}: {move_names[action]}")
    print(f"Obecna pozycja: {observation}")
    
    observation, reward, terminated, truncated, info = visual_env.step(action)
    total_reward += reward
    step_count += 1
    
    # Pauza między ruchami dla lepszej obserwacji
    time.sleep(1)
    
    if terminated or truncated:
        if reward > 0:
            print(f"\n🎉 SUKCES! Agent dotarł do celu w {step_count} krokach!")
        else:
            print(f"\n❌ PORAŻKA! Agent wpadł w dziurę po {step_count} krokach.")
        break

print(f"\nKońcowa suma nagród: {total_reward}")
print("Naciśnij Enter, aby zamknąć okno gry...")
input()
visual_env.close()

# Pokaż statystyki treningu
# Pokaż statystyki treningu
ga_instance.plot_fitness()