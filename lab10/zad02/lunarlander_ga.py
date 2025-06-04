import gymnasium as gym
import numpy as np
import pygad
import time
import matplotlib.pyplot as plt

# Utwórz środowisko LunarLander bez wizualizacji dla treningu
env = gym.make("LunarLander-v3")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Parametry chromosomu dla LunarLander
CHROMOSOME_LENGTH = 500  # Maksymalna długość epizodu
POPULATION_SIZE = 50     # Populacja chromosomów
NUM_GENERATIONS = 100    # Liczba generacji

def fitness_function(ga_instance, solution, solution_idx):
    """
    Funkcja fitness dla algorytmu genetycznego w LunarLander.
    
    CHROMOSOM: Lista liczb całkowitych [0,1,2,3] reprezentujących akcje:
    - 0: NIC (brak akcji)
    - 1: LEWY silnik
    - 2: GŁÓWNY silnik (do góry)
    - 3: PRAWY silnik
    
    FUNKCJA FITNESS:
    1. Symuluje epizod LunarLander wykonując akcje z chromosomu
    2. Zbiera nagrody środowiska które uwzględniają:
       - Pozycję względem lądowiska
       - Prędkość pionową i poziomą
       - Kąt i prędkość kątową lądownika
       - Kontakt z ziemią (nogi dotykające powierzchni)
       - Zużycie paliwa
    3. Dodatkowe bonusy/kary:
       - Sukces lądowania: +100 punktów
       - Katastrofa: -100 punktów
       - Bonus za długie utrzymanie się w powietrzu
    """
    observation, info = env.reset(seed=42)
    total_reward = 0
    steps_taken = 0
    
    for action in solution:
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps_taken += 1
        
        # Sprawdź czy epizod się zakończył
        if terminated or truncated:
            # Analiza przyczyny zakończenia na podstawie nagrody
            if reward >= 100:  # Sukces lądowania
                total_reward += 100  # Dodatkowy bonus za sukces
                print(f"Udane lądowanie! Kroki: {steps_taken}, Nagroda: {total_reward:.2f}")
            elif reward <= -100:  # Katastrofa
                total_reward -= 100  # Dodatkowa kara za katastrofę
            break
    
    # Bonus za długość epizodu (dłuższe loty = lepsze sterowanie)
    survival_bonus = steps_taken / CHROMOSOME_LENGTH * 10
    total_reward += survival_bonus
    
    return total_reward

def on_generation(ga_instance):
    """Callback wywoływany po każdej generacji"""
    best_fitness = ga_instance.best_solution()[1]
    avg_fitness = np.mean(ga_instance.last_generation_fitness)
    print(f"Generacja {ga_instance.generations_completed}: "
          f"Najlepszy={best_fitness:.2f}, Średni={avg_fitness:.2f}")

# Konfiguracja algorytmu genetycznego
ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=15,
    fitness_func=fitness_function,
    sol_per_pop=POPULATION_SIZE,
    num_genes=CHROMOSOME_LENGTH,
    gene_type=int,
    gene_space=[0, 1, 2, 3],  # Możliwe akcje w LunarLander
    mutation_type="random",
    mutation_percent_genes=20,  # Wyższa mutacja dla eksploracji
    crossover_type="two_points",  # Krzyżowanie dwupunktowe
    parent_selection_type="rank",  # Selekcja rankingowa
    on_generation=on_generation,
    random_seed=42
)

# Uruchom algorytm genetyczny
print("Rozpoczynam trening algorytmu genetycznego dla LunarLander...")
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
print("\n" + "="*60)
print("WIZUALNA DEMONSTRACJA NAJLEPSZEGO ROZWIĄZANIA")
print("="*60)

action_names = {0: "NIC", 1: "LEWY", 2: "GŁÓWNY", 3: "PRAWY"}

# Utwórz nowe środowisko z trybem wizualnym
visual_env = gym.make("LunarLander-v3", render_mode="human")

observation, info = visual_env.reset(seed=42)
total_reward = 0
step_count = 0

print("Naciśnij Enter, aby rozpocząć wizualną demonstrację lądowania...")
input()

for action in best_solution:
    observation, reward, terminated, truncated, info = visual_env.step(action)
    total_reward += reward
    step_count += 1
    
    # Wyświetl informacje co 50 kroków
    if step_count % 50 == 0:
        print(f"\nKrok {step_count}: Akcja={action_names[action]}")
        print(f"Pozycja: ({observation[0]:.2f}, {observation[1]:.2f})")
        print(f"Prędkość: ({observation[2]:.2f}, {observation[3]:.2f})")
        print(f"Kąt: {observation[4]:.2f}, Prędkość kątowa: {observation[5]:.2f}")
        print(f"Nagroda łączna: {total_reward:.2f}")
    
    # Krótka pauza dla lepszej obserwacji
    time.sleep(0.02)
    
    if terminated or truncated:
        if reward >= 100:
            print(f"\n🚀 SUKCES! Udane lądowanie po {step_count} krokach!")
            print(f"Końcowa nagroda za lądowanie: {reward:.2f}")
        elif reward <= -100:
            print(f"\n💥 KATASTROFA! Lądownik rozbił się po {step_count} krokach!")
        else:
            print(f"\n⏱️ Epizod zakończony po {step_count} krokach (timeout).")
        break

print(f"\nKońcowa suma nagród: {total_reward:.2f}")
print("Naciśnij Enter, aby zamknąć okno gry...")
input()
visual_env.close()

# Pokaż wykres postępu fitness
ga_instance.plot_fitness()
plt.title("Postęp algorytmu genetycznego w LunarLander")
plt.xlabel("Generacja")
plt.ylabel("Fitness")
plt.show()