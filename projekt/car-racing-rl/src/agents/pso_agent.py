import numpy as np
import pickle
import time

class PSOAgent:
    def __init__(self, env, num_particles=30, dimensions=600, w=0.5, c1=1.5, c2=1.5):
        """Agent oparty na Particle Swarm Optimization"""
        self.env = env
        self.num_particles = num_particles
        self.dimensions = dimensions  # 200 akcji × 3 wartości
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
        # Inicjalizacja roju
        self.particles = np.random.uniform(-1, 1, (num_particles, dimensions))
        self.velocities = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))
        
        # Najlepsze pozycje
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(num_particles, -np.inf)
        
        self.global_best = None
        self.global_best_fitness = -np.inf
        
        self.fitness_history = []
        
        print(f"🐝 PSO Agent utworzony:")
        print(f"   🐛 Liczba cząstek: {num_particles}")
        print(f"   📊 Wymiary: {dimensions}")
        print(f"   ⚖️ Parametry: w={w}, c1={c1}, c2={c2}")
    
    def fitness_function(self, particle):
        """Funkcja fitness dla cząstki"""
        try:
            observation, info = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = min(self.dimensions // 3, 1000)
            
            # Przekształć cząstkę na sekwencję akcji
            actions = self._particle_to_actions(particle)
            
            for action in actions[:max_steps]:
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Bonusy
            survival_bonus = steps * 0.1
            exploration_bonus = info.get('tiles_visited', 0) * 2
            
            # Regularyzacja - preferuj mniejsze wartości
            regularization = -0.001 * np.sum(np.abs(particle))
            
            if total_reward < 0:
                total_reward *= 0.5
            
            fitness = total_reward + survival_bonus + exploration_bonus + regularization
            return max(0, fitness)
            
        except Exception as e:
            print(f"Błąd w funkcji fitness PSO: {e}")
            return 0
    
    def _particle_to_actions(self, particle):
        """Przekształca cząstkę na sekwencję akcji"""
        actions = []
        for i in range(0, len(particle), 3):
            if i + 2 < len(particle):
                # Normalizuj wartości do odpowiednich zakresów
                steering = np.clip(np.tanh(particle[i]), -1.0, 1.0)
                gas = np.clip(1.0 / (1 + np.exp(-particle[i + 1])), 0.0, 1.0)  # sigmoid
                brake = np.clip(1.0 / (1 + np.exp(-particle[i + 2])), 0.0, 1.0)  # sigmoid
                actions.append(np.array([steering, gas, brake]))
        return actions
    
    def update_velocities(self, iteration):
        """Aktualizuj prędkości cząstek"""
        for i in range(self.num_particles):
            # Komponenty prędkości
            inertia = self.w * self.velocities[i]
            
            cognitive = (self.c1 * np.random.random(self.dimensions) * 
                        (self.personal_best[i] - self.particles[i]))
            
            if self.global_best is not None:
                social = (self.c2 * np.random.random(self.dimensions) * 
                         (self.global_best - self.particles[i]))
            else:
                social = np.zeros(self.dimensions)
            
            # Nowa prędkość
            self.velocities[i] = inertia + cognitive + social
            
            # Ograniczenie prędkości
            max_velocity = 0.5
            self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
    
    def update_positions(self):
        """Aktualizuj pozycje cząstek"""
        self.particles += self.velocities
        
        # Ograniczenia pozycji
        self.particles = np.clip(self.particles, -3.0, 3.0)
    
    def train(self, max_iterations=50):
        """Trenowanie algorytmu PSO"""
        print(f"🚀 Rozpoczynanie treningu PSO na {max_iterations} iteracji")
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteracja {iteration + 1}/{max_iterations} ---")
            
            # Oceń wszystkie cząstki
            for i in range(self.num_particles):
                fitness = self.fitness_function(self.particles[i])
                
                # Aktualizuj osobiste maksimum
                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                
                # Aktualizuj globalne maksimum
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()
                    print(f"🎯 Nowy rekord! Fitness: {fitness:.2f}")
            
            # Zapisz historię
            self.fitness_history.append(self.global_best_fitness)
            
            print(f"Najlepszy fitness w iteracji: {self.global_best_fitness:.2f}")
            print(f"Średni fitness: {np.mean(self.personal_best_fitness):.2f}")
            
            # Aktualizuj rój
            self.update_velocities(iteration)
            self.update_positions()
            
            # Zapisz checkpoint co 10 iteracji
            if (iteration + 1) % 10 == 0:
                self.save_model(iteration + 1)
        
        end_time = time.time()
        
        print(f"✅ Trening PSO zakończony w {end_time - start_time:.2f}s")
        print(f"🏆 Najlepszy fitness: {self.global_best_fitness:.2f}")
        
        # Finalne zapisanie
        self.save_model(max_iterations, final=True)
    
    def act(self, observation, step=0):
        """Podejmij akcję na podstawie najlepszej cząstki"""
        if self.global_best is None:
            # Fallback
            return np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.0, 0.2)
            ])
        
        # Przekształć najlepszą cząstkę na akcje
        actions = self._particle_to_actions(self.global_best)
        
        # Wybierz akcję cyklicznie
        action_index = step % len(actions)
        return actions[action_index]
    
    def save_model(self, iteration, final=False):
        """Zapisz model PSO"""
        try:
            if final:
                filename = 'models/pso_best_final.pkl'
            else:
                filename = f'checkpoints/pso_iter_{iteration}.pkl'
            
            data = {
                'global_best': self.global_best,
                'global_best_fitness': self.global_best_fitness,
                'particles': self.particles,
                'personal_best': self.personal_best,
                'personal_best_fitness': self.personal_best_fitness,
                'fitness_history': self.fitness_history,
                'parameters': {
                    'num_particles': self.num_particles,
                    'dimensions': self.dimensions,
                    'w': self.w,
                    'c1': self.c1,
                    'c2': self.c2
                }
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"💾 Model PSO zapisany: {filename}")
            
        except Exception as e:
            print(f"❌ Błąd zapisywania PSO: {e}")
    
    @classmethod
    def load_model(cls, filename, env):
        """Wczytaj model PSO"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            params = data['parameters']
            agent = cls(env, params['num_particles'], params['dimensions'], 
                       params['w'], params['c1'], params['c2'])
            
            agent.global_best = data['global_best']
            agent.global_best_fitness = data['global_best_fitness']
            agent.particles = data['particles']
            agent.personal_best = data['personal_best']
            agent.personal_best_fitness = data['personal_best_fitness']
            agent.fitness_history = data['fitness_history']
            
            print(f"✅ Model PSO wczytany z: {filename}")
            return agent
            
        except Exception as e:
            print(f"❌ Błąd wczytywania PSO: {e}")
            return None