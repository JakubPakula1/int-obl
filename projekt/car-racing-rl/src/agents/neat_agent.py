import neat
import numpy as np
import pickle
import os
import cv2

class NEATAgent:
    def __init__(self, config_path):
        """
        Inicjalizacja agenta NEAT
        
        Args:
            config_path (str): Ścieżka do pliku konfiguracyjnego NEAT
        """
        self.config_path = config_path
        
        # Sprawdź czy plik konfiguracji istnieje
        if not os.path.exists(config_path):
            print(f"❌ Brak pliku konfiguracji: {config_path}")
            print("Tworzenie domyślnej konfiguracji...")
            self.create_default_config(config_path)
        
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path
        )
        self.population = neat.Population(self.config)
        self.generation = 0
        self.best_genome = None
        self.best_net = None
        
        # Dodaj reporter do śledzenia postępów
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        print(f"🧬 NEAT Agent utworzony:")
        print(f"   Populacja: {self.config.pop_size}")
        print(f"   Wejścia: {self.config.genome_config.num_inputs}")
        print(f"   Wyjścia: {self.config.genome_config.num_outputs}")
        
    def preprocess_observation(self, observation):
        """
        Przetwarzanie konsystentne z DQN/PPO - 84x84 pikseli
        
        Args:
            observation: Obserwacja ze środowiska
            
        Returns:
            numpy.array: Spłaszczona i znormalizowana obserwacja
        """
        # Obsługa różnych formatów wejścia
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Konwersja do skali szarości
        if len(observation.shape) == 3:
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        else:
            gray = observation
            
        # POPRAWKA: Konsystentne z DQN/PPO - 84x84 pikseli
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalizuj do zakresu [-1, 1] (lepsze dla NEAT niż [0, 1])
        normalized = (resized.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        return normalized.flatten()  # 7056 elementów
    def act(self, observation, genome=None):
        """
        POPRAWIONA funkcja wyboru akcji
        
        Args:
            observation: Obserwacja ze środowiska
            genome: Genom NEAT (opcjonalnie)
            
        Returns:
            numpy.array: Akcja [steering, gas, brake]
        """
        if genome is None and self.best_net is None:
            # POPRAWKA: Bardziej sensowne akcje domyślne
            return np.array([
                np.random.uniform(-0.5, 0.5),  # Mniejsze losowe skręty
                np.random.uniform(0.3, 0.7),   # Umiarkowana prędkość
                np.random.uniform(0.0, 0.2)    # Rzadkie hamowanie
            ])
        
        try:
            # Przetwórz obserwację
            inputs = self.preprocess_observation(observation)
            
            # POPRAWKA: Sprawdź długość wejścia
            expected_inputs = self.config.genome_config.num_inputs
            if len(inputs) != expected_inputs:
                print(f"⚠️ Błąd wymiarów: otrzymano {len(inputs)}, oczekiwano {expected_inputs}")
                return np.array([0.0, 0.3, 0.0])  # Bezpieczna akcja
            
            # Użyj sieci neuronowej
            if genome is not None:
                net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            else:
                net = self.best_net
                
            outputs = net.activate(inputs)
            
            # POPRAWKA: Sprawdź czy mamy wystarczająco wyjść
            if len(outputs) < 3:
                print(f"⚠️ Błąd wyjść: otrzymano {len(outputs)}, oczekiwano 3")
                return np.array([0.0, 0.3, 0.0])
            
            # POPRAWKA: Bardziej stabilne przekształcenia
            steering = np.clip(np.tanh(outputs[0]), -1.0, 1.0)  # [-1, 1]
            gas = np.clip(outputs[1], 0.0, 1.0)                 # [0, 1]
            brake = np.clip(outputs[2], 0.0, 1.0)               # [0, 1]
            
            return np.array([steering, gas, brake])
            
        except Exception as e:
            print(f"❌ Błąd w act(): {e}")
            return np.array([0.0, 0.3, 0.0])  # Bezpieczna akcja
    
    def evaluate_genome(self, genome, config, env, num_episodes=1, max_steps=1000):
        """
        POPRAWIONA ocena pojedynczego genomu
        
        Args:
            genome: Genom NEAT do oceny
            config: Konfiguracja NEAT
            env: Środowisko
            num_episodes: Liczba epizodów do oceny
            max_steps: Maksymalna liczba kroków
            
        Returns:
            float: Fitness genomu
        """
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        except Exception as e:
            print(f"❌ Błąd tworzenia sieci: {e}")
            return -1000
        
        total_fitness = 0
        
        for episode in range(num_episodes):
            try:
                # POPRAWKA: Prawidłowy unpacking env.reset()
                observation, info = env.reset()
                episode_reward = 0
                steps = 0
                negative_steps = 0
                stagnant_steps = 0
                prev_tiles = info.get('tiles_visited', 0)
                
                while steps < max_steps:
                    # Wybierz akcję
                    try:
                        inputs = self.preprocess_observation(observation)
                        outputs = net.activate(inputs)
                        
                        # Konwertuj na akcję
                        if len(outputs) >= 3:
                            steering = np.clip(np.tanh(outputs[0]), -1.0, 1.0)
                            gas = np.clip(outputs[1], 0.0, 1.0)
                            brake = np.clip(outputs[2], 0.0, 1.0)
                            action = np.array([steering, gas, brake])
                        else:
                            action = np.array([0.0, 0.3, 0.0])
                            
                    except Exception as e:
                        print(f"❌ Błąd aktywacji: {e}")
                        action = np.array([0.0, 0.3, 0.0])
                    
                    # Wykonaj krok
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    # POPRAWKA: Lepsze kryteria przerwania
                    current_tiles = info.get('tiles_visited', 0)
                    
                    # Sprawdź postęp w eksploracji
                    if current_tiles > prev_tiles:
                        stagnant_steps = 0  # Reset jeśli jest postęp
                        prev_tiles = current_tiles
                    else:
                        stagnant_steps += 1
                    
                    # Przerwij jeśli długo bez postępu
                    if stagnant_steps > 100:  # 100 kroków bez nowych płytek
                        break
                    
                    # Przerwij jeśli zbyt długo negatywne nagrody
                    if reward < -1.0:
                        negative_steps += 1
                        if negative_steps > 50:
                            break
                    else:
                        negative_steps = 0
                    
                    if terminated or truncated:
                        break
                
                # POPRAWKA: Dodatkowe bonusy za długie przetrwanie i eksplorację
                if steps > 200:
                    episode_reward += 20
                if steps > 350:
                    episode_reward += 50
                
                # Bonus za eksplorację
                tiles_bonus = current_tiles * 2  # 2 punkty za każdą odwiedzoną płytkę
                episode_reward += tiles_bonus
                
                total_fitness += episode_reward
                
            except Exception as e:
                print(f"❌ Błąd w epizodzie: {e}")
                total_fitness -= 100  # Kara za błąd
        
        avg_fitness = total_fitness / num_episodes
        return max(avg_fitness, -1000)  # Minimalna fitness
    
    def train(self, env, generations=20):
        """
        POPRAWIONE trenowanie z lepszą obsługą błędów
        """
        print(f"🧬 Rozpoczynam trening NEAT na {generations} generacji")
        
        # Dodaj checkpointer
        os.makedirs('checkpoints', exist_ok=True)
        checkpointer = neat.Checkpointer(
            generation_interval=5, 
            filename_prefix='checkpoints/neat-checkpoint-'
        )
        self.population.add_reporter(checkpointer)
        
        def eval_genomes(genomes, config):
            print(f"Oceniam {len(genomes)} genomów...")
            
            for i, (genome_id, genome) in enumerate(genomes):
                if i % 20 == 0:  # Progress co 20 genomów
                    print(f"  Genom {i+1}/{len(genomes)}")
                
                genome.fitness = self.evaluate_genome(genome, config, env)
                
                # Śledź najlepszy genom
                if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                    self.best_genome = genome
                    self.best_net = neat.nn.FeedForwardNetwork.create(genome, config)
                    print(f"  🌟 Nowy najlepszy: fitness={genome.fitness:.2f}")
        
        # Uruchom ewolucję
        winner = self.population.run(eval_genomes, generations)
        
        # Zapisz najlepszy model
        self.best_genome = winner
        self.best_net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        
        # Automatyczne zapisanie
        os.makedirs('models', exist_ok=True)
        self.save_model('models/neat_best.pkl')
        
        print(f"🏆 Trening zakończony!")
        print(f"📊 Najlepsza fitness: {winner.fitness:.2f}")
        
        return winner
    
    def save_model(self, filename):
        """POPRAWIONE zapisywanie modelu"""
        if self.best_genome is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                # Zapisz genom i konfigurację
                data = {
                    'genome': self.best_genome,
                    'config_path': self.config_path,
                    'fitness': self.best_genome.fitness
                }
                pickle.dump(data, f)
            print(f"💾 Model zapisany: {filename}")
        else:
            print("⚠️ Brak genomu do zapisania!")
    
    @classmethod
    def load_model(cls, filename, config_path):
        """POPRAWIONE ładowanie modelu"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Utwórz agenta z konfiguracji
            agent = cls(config_path)
            
            # Wczytaj genom
            if isinstance(data, dict):
                agent.best_genome = data['genome']
                print(f"📈 Wczytano model z fitness: {data.get('fitness', 'nieznana')}")
            else:
                # Stary format - sam genom
                agent.best_genome = data
            
            # Utwórz sieć
            agent.best_net = neat.nn.FeedForwardNetwork.create(agent.best_genome, agent.config)
            
            print(f"✅ Model wczytany: {filename}")
            return agent
        else:
            print(f"❌ Plik nie istnieje: {filename}")
            return None