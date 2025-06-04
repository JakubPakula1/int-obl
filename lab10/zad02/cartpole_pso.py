import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, num_actions, num_steps):
        # Pozycja cząstki to sekwencja ruchów
        self.position = np.random.randint(0, 2, num_steps)  # 0 lub 1 (lewo/prawo)
        self.velocity = np.random.uniform(-0.5, 0.5, num_steps)  # Losowa prędkość początkowa
        self.best_position = self.position.copy()
        self.best_score = float('-inf')

class PSO:
    def __init__(self, num_particles, num_steps, env, w=0.7, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.env = env
        self.w = w  # Współczynnik bezwładności
        self.c1 = c1  # Współczynnik uczenia osobistego
        self.c2 = c2  # Współczynnik uczenia społecznego
        
        # Inicjalizacja roju
        self.particles = [Particle(2, num_steps) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = float('-inf')
        
        # Historia dla wykresów
        self.best_scores_history = []
        self.avg_scores_history = []
    
    def evaluate_particle(self, particle):
        """
        Ocena cząstki poprzez symulację w środowisku.
        
        REPREZENTACJA CZĄSTKI:
        - Position: Tablica k-wymiarowa (k=num_steps)
        - Każdy wymiar reprezentuje jedną akcję w sekwencji
        - Wartości 0/1 dla dyskretnych akcji (lewo/prawo)
        
        FUNKCJA FITNESS:
        - Podstawowa nagroda za każdy krok = +1
        - Bonus za długie epizody (>100 kroków)
        - Kara za wczesne zakończenie
        """
        observation, info = self.env.reset(seed=42)  # Stały seed dla reprodukowalności
        total_reward = 0
        steps_taken = 0
        
        for action in particle.position:
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps_taken += 1
            
            if terminated or truncated:
                break
        
        # Dodatkowe nagrody za wydajność
        if steps_taken >= 195:  # Maksymalny wynik w CartPole
            total_reward += 50  # Bonus za perfekcyjny wynik
        elif steps_taken > 100:
            total_reward += (steps_taken - 100) * 0.1  # Bonus za długie utrzymanie
        
        return total_reward
    
    def update_velocity(self, particle):
        """
        Aktualizacja prędkości według wzoru PSO.
        W przestrzeni dyskretnej używamy probabilistycznego podejścia.
        """
        if self.global_best_position is None:
            return
            
        r1, r2 = np.random.rand(2)
        
        # Obliczanie nowej prędkości
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = (self.w * particle.velocity + cognitive + social)
        
        # Ograniczenie prędkości dla stabilności
        particle.velocity = np.clip(particle.velocity, -2, 2)
    
    def update_position(self, particle):
        """
        Aktualizacja pozycji z adaptacją do przestrzeni dyskretnej.
        Używamy funkcji sigmoidalnej do konwersji prędkości na prawdopodobieństwa.
        """
        # Aktualizacja pozycji z prędkością
        continuous_position = particle.position + particle.velocity
        
        # Konwersja do przestrzeni dyskretnej za pomocą sigmoid
        probabilities = 1 / (1 + np.exp(-continuous_position))
        
        # Próbkowanie binarne na podstawie prawdopodobieństw
        particle.position = (np.random.rand(self.num_steps) < probabilities).astype(int)
    
    def optimize(self, num_iterations, verbose=True):
        """Główna pętla optymalizacji PSO z monitoringiem postępu"""
        for iteration in range(num_iterations):
            scores = []
            
            for particle in self.particles:
                # Ocena cząstki
                score = self.evaluate_particle(particle)
                scores.append(score)
                
                # Aktualizacja najlepszej pozycji cząstki
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Aktualizacja globalnej najlepszej pozycji
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                
                # Aktualizacja prędkości i pozycji
                self.update_velocity(particle)
                self.update_position(particle)
            
            # Zapisz statystyki
            self.best_scores_history.append(self.global_best_score)
            self.avg_scores_history.append(np.mean(scores))
            
            # Adaptacyjne zmniejszanie bezwładności
            self.w = max(0.1, self.w * 0.99)
            
            if verbose and (iteration + 1) % 10 == 0:  # Wyświetlaj co 10 iteracji
                print(f"Iteracja {iteration + 1}: Najlepszy={self.global_best_score:.1f}, "
                      f"Średni={np.mean(scores):.1f}, w={self.w:.3f}")

def train_pso_fast():
    """Szybki trening PSO bez wizualizacji"""
    print("=== FAZA TRENINGU PSO ===")
    
    # Środowisko bez renderowania dla szybkiego treningu
    env_train = gym.make("CartPole-v1")
    print(f"Środowisko treningu: {env_train.spec.id}")
    print(f"Przestrzeń obserwacji: {env_train.observation_space}")
    print(f"Przestrzeń akcji: {env_train.action_space}")

    # Parametry PSO
    NUM_PARTICLES = 30
    NUM_STEPS = 200  # Maksymalna długość sekwencji
    NUM_ITERATIONS = 100

    # Utwórz i uruchom PSO
    print(f"\nRozpoczynam szybką optymalizację PSO...")
    print(f"Parametry: {NUM_PARTICLES} cząstek, {NUM_STEPS} kroków, {NUM_ITERATIONS} iteracji")
    
    start_time = time.time()
    pso = PSO(NUM_PARTICLES, NUM_STEPS, env_train)
    pso.optimize(NUM_ITERATIONS, verbose=True)
    end_time = time.time()

    print(f"\n✅ Optymalizacja zakończona w {end_time - start_time:.2f} sekund")
    print(f"🏆 Najlepszy wynik: {pso.global_best_score:.1f}")
    
    env_train.close()
    return pso

def show_training_results(pso):
    """Wyświetl wykresy postępu treningu"""
    print("\n=== ANALIZA WYNIKÓW TRENINGU ===")
    
    plt.figure(figsize=(12, 8))
    
    # Wykres 1: Postęp fitness
    plt.subplot(2, 2, 1)
    plt.plot(pso.best_scores_history, label='Najlepszy wynik', linewidth=2, color='green')
    plt.plot(pso.avg_scores_history, label='Średni wynik', alpha=0.7, color='blue')
    plt.xlabel('Iteracja')
    plt.ylabel('Fitness')
    plt.title('Postęp PSO w CartPole')
    plt.legend()
    plt.grid(True)
    
    # Wykres 2: Ostatnie 20 iteracji (zoom)
    plt.subplot(2, 2, 2)
    last_20_best = pso.best_scores_history[-20:]
    last_20_avg = pso.avg_scores_history[-20:]
    plt.plot(range(len(pso.best_scores_history)-20, len(pso.best_scores_history)), 
             last_20_best, label='Najlepszy (ostatnie 20)', linewidth=2, color='red')
    plt.plot(range(len(pso.avg_scores_history)-20, len(pso.avg_scores_history)), 
             last_20_avg, label='Średni (ostatnie 20)', alpha=0.7, color='orange')
    plt.xlabel('Iteracja')
    plt.ylabel('Fitness')
    plt.title('Ostatnie 20 iteracji')
    plt.legend()
    plt.grid(True)
    
    # Wykres 3: Histogram najlepszych wyników
    plt.subplot(2, 2, 3)
    plt.hist(pso.best_scores_history, bins=20, alpha=0.7, color='purple')
    plt.xlabel('Fitness')
    plt.ylabel('Częstość')
    plt.title('Rozkład najlepszych wyników')
    plt.grid(True)
    
    # Wykres 4: Statystyki
    plt.subplot(2, 2, 4)
    stats_text = f"""
Statystyki treningu:
• Najlepszy wynik: {max(pso.best_scores_history):.1f}
• Średni wynik końcowy: {pso.avg_scores_history[-1]:.1f}
• Poprawa: {pso.best_scores_history[-1] - pso.best_scores_history[0]:.1f}
• Stabilność (std): {np.std(pso.best_scores_history[-10:]):.2f}
"""
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    plt.axis('off')
    plt.title('Statystyki')
    
    plt.tight_layout()
    plt.show()

def demonstrate_solution(pso):
    """Demonstracja najlepszego rozwiązania z wizualizacją"""
    print("\n=== DEMONSTRACJA NAJLEPSZEGO ROZWIĄZANIA ===")
    
    # Środowisko z renderowaniem tylko do demonstracji
    env_demo = gym.make("CartPole-v1", render_mode="human")
    
    print("🎮 Uruchamiam demonstrację najlepszego rozwiązania...")
    print("Naciśnij Enter aby rozpocząć demonstrację...")
    input()
    
    observation, info = env_demo.reset(seed=42)
    total_reward = 0
    step_count = 0
    
    action_names = {0: "⬅️ LEWO", 1: "➡️ PRAWO"}
    
    print(f"Rozpoczynam symulację z {len(pso.global_best_position)} zaplanowanymi ruchami...")
    
    for i, action in enumerate(pso.global_best_position):
        observation, reward, terminated, truncated, info = env_demo.step(action)
        total_reward += reward
        step_count += 1
        
        # Wyświetl informacje co kilka kroków
        if step_count % 20 == 0 or step_count <= 10:
            print(f"Krok {step_count:3d}: {action_names[action]} | "
                  f"Pozycja: {observation[0]:6.3f} | "
                  f"Kąt: {observation[2]:6.3f} | "
                  f"Nagroda: {total_reward:.0f}")
        
        time.sleep(0.05)  # Pauza dla wizualizacji
        
        if terminated or truncated:
            break
    
    print(f"\n🏁 Epizod zakończony!")
    print(f"📊 Statystyki:")
    print(f"   • Czas trwania: {step_count} kroków")
    print(f"   • Końcowa suma nagród: {total_reward}")
    print(f"   • Skuteczność: {(step_count/200)*100:.1f}% (max 200 kroków)")
    
    # Ocena wyników
    if step_count >= 195:
        print("🏆 DOSKONAŁY WYNIK! Perfekcyjne rozwiązanie!")
    elif step_count >= 150:
        print("🥇 BARDZO DOBRY WYNIK!")
    elif step_count >= 100:
        print("🥈 DOBRY WYNIK!")
    else:
        print("🥉 WYNIK DO POPRAWY")
    
    env_demo.close()

def main():
    """Główna funkcja łącząca trening i demonstrację"""
    print("🚀 OPTYMALIZACJA CARTPOLE PRZY UŻYCIU PSO")
    print("=" * 50)
    
    # Faza 1: Szybki trening
    trained_pso = train_pso_fast()
    
    # Faza 2: Analiza wyników
    show_training_results(trained_pso)
    
    # Faza 3: Demonstracja
    demonstrate_solution(trained_pso)
    
    print("\n✅ Program zakończony pomyślnie!")

# Uruchomienie programu
if __name__ == "__main__":
    main()