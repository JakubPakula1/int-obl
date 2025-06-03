import gymnasium as gym
import numpy as np
import time

class Particle:
    def __init__(self, num_actions, num_steps):
        # Pozycja cząstki to sekwencja ruchów
        self.position = np.random.randint(0, 2, num_steps)  # 0 lub 1 (lewo/prawo)
        self.velocity = np.zeros(num_steps)
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
    
    def evaluate_particle(self, particle):
        """Ocena cząstki poprzez symulację w środowisku"""
        observation, info = self.env.reset(seed=0)
        total_reward = 0
        steps_taken = 0
        
        for action in particle.position:
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps_taken += 1
            
            if terminated or truncated:
                break
        
        # Bonus za dłuższe utrzymanie równowagi
        if steps_taken > 100:
            total_reward += (steps_taken - 100) * 0.1
        
        return total_reward
    
    def update_velocity(self, particle):
        """Aktualizacja prędkości cząstki"""
        r1, r2 = np.random.rand(2)
        
        # Obliczanie nowej prędkości
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = (self.w * particle.velocity + cognitive + social)
        
        # Ograniczenie prędkości
        particle.velocity = np.clip(particle.velocity, -1, 1)
    
    def update_position(self, particle):
        """Aktualizacja pozycji cząstki"""
        # Aktualizacja pozycji z prędkością
        particle.position = particle.position + particle.velocity
        
        # Zaokrąglenie do 0 lub 1 (dyskretne akcje)
        particle.position = np.round(particle.position).astype(int)
        particle.position = np.clip(particle.position, 0, 1)
    
    def optimize(self, num_iterations):
        """Główna pętla optymalizacji PSO"""
        for iteration in range(num_iterations):
            for particle in self.particles:
                # Ocena cząstki
                score = self.evaluate_particle(particle)
                
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
            
            print(f"Iteracja {iteration + 1}, Najlepszy wynik: {self.global_best_score}")

# Utwórz środowisko
env = gym.make("CartPole-v1", render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Parametry PSO
NUM_PARTICLES = 20
NUM_STEPS = 200
NUM_ITERATIONS = 50

# Utwórz i uruchom PSO
print("Rozpoczynam optymalizację PSO...")
pso = PSO(NUM_PARTICLES, NUM_STEPS, env)
pso.optimize(NUM_ITERATIONS)

# Pokaż najlepsze rozwiązanie
print("\nDemonstracja najlepszego rozwiązania:")
observation, info = env.reset(seed=0)
total_reward = 0

for action in pso.global_best_position:
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.05)  # Krótka pauza dla lepszej wizualizacji
    
    if terminated or truncated:
        break

print(f"Końcowa suma nagród: {total_reward}")
env.close() 