import neat
import numpy as np
import pickle
import os

class NEATAgent:
    def __init__(self, config_path):
        """
        Inicjalizacja agenta NEAT
        
        Args:
            config_path (str): Ścieżka do pliku konfiguracyjnego NEAT
        """
        self.config_path = config_path
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
        
    def preprocess_observation(self, observation):
        """
        Przetwarzanie obserwacji do formatu odpowiedniego dla sieci neuronowej
        
        Args:
            observation: Obserwacja ze środowiska (96x96x3 obraz)
            
        Returns:
            numpy.array: Spłaszczona i znormalizowana obserwacja
        """
            # Konwertuj do skali szarości
        if len(observation.shape) == 3:
            gray = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = observation
            
        # Bardziej agresywne zmniejszenie: 96x96 -> 6x4 = 24 piksele
        downsampled = gray[::16, ::24]  # Co 16. piksel w pionie, co 24. w poziomie
        
        # Normalizuj do zakresu [-1, 1]
        normalized = (downsampled / 255.0) * 2.0 - 1.0
        
        return normalized.flatten()
    
    def act(self, observation, genome=None):
        """
        Wybierz akcję na podstawie obserwacji
        
        Args:
            observation: Obserwacja ze środowiska
            genome: Genom NEAT (opcjonalnie)
            
        Returns:
            numpy.array: Akcja [steering, gas, brake]
        """
        if genome is None and self.best_net is None:
            # Zwróć losową akcję jeśli nie ma wytrenowanej sieci
            return np.array([
                np.random.uniform(-1, 1),  # steering
                np.random.uniform(0, 1),   # gas
                np.random.uniform(0, 1)    # brake
            ])
        
        # Przetwórz obserwację
        inputs = self.preprocess_observation(observation)
        
        # Użyj sieci neuronowej
        if genome is not None:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        else:
            net = self.best_net
            
        outputs = net.activate(inputs)
        
        # Przekształć wyjścia na akcje
        steering = np.tanh(outputs[0])  # [-1, 1]
        gas = max(0, outputs[1])        # [0, inf] -> [0, 1]
        brake = max(0, outputs[2])      # [0, inf] -> [0, 1]
        
        # Normalizuj gas i brake
        gas = min(1.0, gas)
        brake = min(1.0, brake)
        
        return np.array([steering, gas, brake])
    
    def evaluate_genome(self, genome, config, env, num_episodes=3):
        """
        Ocena pojedynczego genomu
        
        Args:
            genome: Genom NEAT do oceny
            config: Konfiguracja NEAT
            env: Środowisko
            num_episodes: Liczba epizodów do oceny
            
        Returns:
            float: Fitness genomu
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_fitness = 0
        
        for episode in range(num_episodes):
            observation = env.reset()
            episode_reward = 0
            steps = 0
            negative_steps = 0
            
            while steps < 300:  # Zmniejsz z 1000 do 300 kroków
                # Wybierz akcję
                inputs = self.preprocess_observation(observation)
                outputs = net.activate(inputs)
                
                # Przekształć na akcję
                steering = np.tanh(outputs[0])
                gas = max(0, min(1.0, outputs[1]))
                brake = max(0, min(1.0, outputs[2]))
                action = np.array([steering, gas, brake])
                
                # Wykonaj krok
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Przerwij wcześniej jeśli auto stoi w miejscu
                if reward < -0.1:
                    negative_steps += 1
                    if negative_steps > 50:  # Przerwij po 50 krokach z negatywną nagrodą
                        break
                else:
                    negative_steps = 0
                
                if terminated or truncated:
                    break
            
            total_fitness += episode_reward
    
        return total_fitness / num_episodes
    
    def train(self, env, generations=50):
        """
        Trenowanie z zapisywaniem checkpointów
        """
        # Dodaj checkpointer
        checkpointer = neat.Checkpointer(generation_interval=5, filename_prefix='checkpoints/neat-checkpoint-')
        self.population.add_reporter(checkpointer)
        
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = self.evaluate_genome(genome, config, env)
                
                if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                    self.best_genome = genome
                    self.best_net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        winner = self.population.run(eval_genomes, generations)
        
        self.best_genome = winner
        self.best_net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        
        return winner
    
    def save_model(self, filename):
        """Zapisz najlepszy model"""
        if self.best_genome is not None:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_genome, f)
    
    def load_model(self, filename):
        """Wczytaj model z pliku"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.best_genome = pickle.load(f)
                self.best_net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)


    def continue_training_from_checkpoint(self, env, additional_generations=10):
        """
        Kontynuuj trening z aktualnego stanu populacji
        
        Args:
            env: Środowisko do treningu  
            additional_generations: Liczba dodatkowych generacji
        """
        if self.best_genome is None:
            print("Brak zapisanego genomu do kontynuacji treningu!")
            return None
        
        print(f"Kontynuowanie treningu z fitness: {self.best_genome.fitness}")
        
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = self.evaluate_genome(genome, config, env)
                
                # Śledź najlepszy genom
                if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                    self.best_genome = genome
                    self.best_net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Kontynuuj trening z obecną populacją
        winner = self.population.run(eval_genomes, additional_generations)
        
        # Zaktualizuj najlepszy genom
        self.best_genome = winner
        self.best_net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        
        return winner