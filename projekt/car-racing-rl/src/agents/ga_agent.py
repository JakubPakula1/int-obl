import numpy as np
import pygad
import pickle
import time

class GeneticAgent:
    def __init__(self, env, chromosome_length=200, population_size=50):
        """Agent oparty na algorytmie genetycznym"""
        self.env = env
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.best_genome = None
        self.fitness_history = []
        
        # Definicja przestrzeni genów
        # Każdy gen to 3 wartości: [steering, gas, brake]
        self.gene_space = [
            {'low': -1.0, 'high': 1.0},  # steering
            {'low': 0.0, 'high': 1.0},   # gas
            {'low': 0.0, 'high': 1.0}    # brake
        ]
        
        print(f"🧬 Genetic Agent utworzony:")
        print(f"   📊 Rozmiar populacji: {population_size}")
        print(f"   🧬 Długość chromosomu: {chromosome_length}")
    
    def fitness_function(self, ga_instance, solution, solution_idx):
        """Funkcja fitness dla chromosomu"""
        try:
            observation, info = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = min(self.chromosome_length // 3, 1000)
            
            # Przekształć chromosom na sekwencję akcji
            actions = self._chromosome_to_actions(solution)
            
            for action in actions[:max_steps]:
                observation, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Bonus za długie przetrwanie
            survival_bonus = steps * 0.1
            
            # Bonus za eksplorację (odwiedzone kafelki)
            exploration_bonus = info.get('tiles_visited', 0) * 2
            
            # Kara za negatywne nagrody
            if total_reward < 0:
                total_reward *= 0.5
            
            fitness = total_reward + survival_bonus + exploration_bonus
            
            # Fitness nie może być ujemny
            return max(0, fitness)
            
        except Exception as e:
            print(f"Błąd w funkcji fitness: {e}")
            return 0
    
    def _chromosome_to_actions(self, chromosome):
        """Przekształca chromosom na sekwencję akcji"""
        actions = []
        for i in range(0, len(chromosome), 3):
            if i + 2 < len(chromosome):
                steering = np.clip(chromosome[i], -1.0, 1.0)
                gas = np.clip(chromosome[i + 1], 0.0, 1.0)
                brake = np.clip(chromosome[i + 2], 0.0, 1.0)
                actions.append(np.array([steering, gas, brake]))
        return actions
    
    def on_generation(self, ga_instance):
        """Callback wywoływany po każdej generacji"""
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution()[1]
        self.fitness_history.append(fitness)
        
        print(f"Generacja {generation}: Najlepszy fitness = {fitness:.2f}")
        
        # Zapisz najlepsze rozwiązanie co 10 generacji
        if generation % 10 == 0:
            best_solution = ga_instance.best_solution()[0]
            self.save_best_genome(best_solution, generation)
    
    def train(self, num_generations=100):
        """Trenowanie algorytmu genetycznego"""
        print(f"🚀 Rozpoczynanie treningu GA na {num_generations} generacji")
        
        # Konfiguracja algorytmu genetycznego
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=self.population_size // 4,
            fitness_func=self.fitness_function,
            sol_per_pop=self.population_size,
            num_genes=self.chromosome_length,
            gene_space=self.gene_space * (self.chromosome_length // 3),
            parent_selection_type="tournament",
            keep_parents=5,
            crossover_type="two_points",
            mutation_type="random",
            mutation_percent_genes=15,
            on_generation=self.on_generation,
            random_seed=42
        )
        
        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        
        # Zapisz najlepsze rozwiązanie
        best_solution, best_fitness, _ = ga_instance.best_solution()
        self.best_genome = best_solution
        
        print(f"✅ Trening GA zakończony w {end_time - start_time:.2f}s")
        print(f"🏆 Najlepszy fitness: {best_fitness:.2f}")
        
        # Finalne zapisanie
        self.save_best_genome(best_solution, num_generations, final=True)
        
        return ga_instance
    
    def act(self, observation, step=0):
        """Podejmij akcję na podstawie najlepszego genomu"""
        if self.best_genome is None:
            # Fallback - losowa akcja
            return np.array([
                np.random.uniform(-0.5, 0.5),  # steering
                np.random.uniform(0.3, 0.7),   # gas
                np.random.uniform(0.0, 0.2)    # brake
            ])
        
        # Przekształć genom na akcje
        actions = self._chromosome_to_actions(self.best_genome)
        
        # Wybierz akcję na podstawie kroku
        action_index = step % len(actions)
        return actions[action_index]
    
    def save_best_genome(self, genome, generation, final=False):
        """Zapisz najlepszy genom"""
        try:
            if final:
                filename = 'models/genetic_best_final.pkl'
            else:
                filename = f'checkpoints/genetic_gen_{generation}.pkl'
            
            with open(filename, 'wb') as f:
                pickle.dump({
                    'genome': genome,
                    'generation': generation,
                    'chromosome_length': self.chromosome_length,
                    'fitness_history': self.fitness_history
                }, f)
            
            print(f"💾 Genom zapisany: {filename}")
            
        except Exception as e:
            print(f"❌ Błąd zapisywania genomu: {e}")
    
    @classmethod
    def load_best_genome(cls, filename, env):
        """Wczytaj najlepszy genom"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            agent = cls(env, data['chromosome_length'])
            agent.best_genome = data['genome']
            agent.fitness_history = data.get('fitness_history', [])
            
            print(f"✅ Genom wczytany z: {filename}")
            return agent
            
        except Exception as e:
            print(f"❌ Błąd wczytywania genomu: {e}")
            return None