import neat
import numpy as np
import pickle
import os
import cv2

class NEATAgent:
    """
    NeuroEvolution of Augmenting Topologies (NEAT) Agent
    
    NEAT to algorytm ewolucyjny, który:
    1. Ewoluuje strukturę sieci neuronowych (topologia + wagi)
    2. Zaczyna od prostych sieci i dodaje złożoność
    3. Używa selekcji naturalnej zamiast gradientów
    4. Nie wymaga backpropagation
    
    - Automatycznie znajduje optymalną architekturę sieci
    - Dobry dla problemów gdzie nie znamy idealnej struktury
    """
    
    def __init__(self, config_path):
        """
        Inicjalizacja agenta NEAT
        
        Args:
            config_path (str): Ścieżka do pliku konfiguracyjnego NEAT
        """
        self.config_path = config_path
        
        # === WALIDACJA KONFIGURACJI ===
        # NEAT absolutnie wymaga pliku konfiguracyjnego
        if not os.path.exists(config_path):
            print(f"❌ Brak pliku konfiguracji: {config_path}")
            print("Tworzenie domyślnej konfiguracji...")
            self.create_default_config(config_path)
        
        # === INICJALIZACJA NEAT ===
        # Cztery główne komponenty NEAT:
        # 1. DefaultGenome - reprezentacja osobnika (sieć + geny)
        # 2. DefaultReproduction - jak tworzyć potomstwo
        # 3. DefaultSpeciesSet - jak grupować podobne osobniki
        # 4. DefaultStagnation - jak radzić sobie z brakiem postępu
        self.config = neat.Config(
            neat.DefaultGenome,        # Struktura genomu
            neat.DefaultReproduction,  # Strategia reprodukcji
            neat.DefaultSpeciesSet,    # Grupowanie w gatunki
            neat.DefaultStagnation,    # Obsługa stagnacji
            config_path
        )
        
        # === POPULACJA ===
        # Populacja to zbiór osobników (genomów) ewoluujących razem
        self.population = neat.Population(self.config)
        
        # === STAN TRENINGU ===
        self.generation = 0           # Obecna generacja
        self.best_genome = None       # Najlepszy genom znaleziony do tej pory
        self.best_net = None          # Sieć neuronowa z najlepszego genomu
        
        # === MONITORING I LOGI ===
        # Reporter pokazuje postęp ewolucji w konsoli
        self.population.add_reporter(neat.StdOutReporter(True))
        
        # Statistyki do analizy
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        # === INFO O KONFIGURACJI ===
        print(f"🧬 NEAT Agent utworzony:")
        print(f"   Populacja: {self.config.pop_size} osobników")
        print(f"   Wejścia: {self.config.genome_config.num_inputs}")
        print(f"   Wyjścia: {self.config.genome_config.num_outputs}")
        
    def preprocess_observation(self, observation):
        """
        Preprocessing obrazu dla NEAT - konsystentny z DQN/PPO
        
        Args:
            observation: Surowy obraz z CarRacing (96x96x3 RGB)
            
        Returns:
            numpy.array: Spłaszczona i znormalizowana obserwacja (7056 elementów)
            
        Proces:
        1. RGB → Grayscale 
        2. 96x96 → 84x84 
        3. [0,255] → [-1,1] 
        """
        # === OBSŁUGA RÓŻNYCH FORMATÓW ===
        # Czasami env.reset() zwraca tuple zamiast array
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # === KONWERSJA DO SKALI SZAROŚCI ===
        if len(observation.shape) == 3:
            # RGB → Grayscale używając OpenCV (najszybsze)
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        else:
            # Już grayscale
            gray = observation
            
        # === RESIZE DO STANDARDOWEGO ROZMIARU ===
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # === NORMALIZACJA DO [-1, 1] ===
        # 1. [0,255] → [0,1]: dziel przez 255
        # 2. [0,1] → [-1,1]: * 2.0 - 1.0
        normalized = (resized.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # === SPŁASZCZENIE 2D → 1D ===
        # NEAT potrzebuje wektora: (84,84) → (7056,)
        return normalized.flatten()  # 84 * 84 = 7056 elementów
    
    def act(self, observation, genome=None):
        """
        Wybór akcji przez sieć neuronową NEAT
        
        Args:
            observation: Obserwacja ze środowiska
            genome: Konkretny genom do użycia (None = użyj najlepszego)
            
        Returns:
            numpy.array: Akcja [steering, gas, brake] dla CarRacing
            
        Proces:
        1. Preprocess obserwacji (7056 wejść dla sieci)
        2. Aktywuj sieć neuronową NEAT
        3. Przekształć wyjścia na akcje CarRacing
        4. Zastosuj ograniczenia i bezpieczeństwo
        """
        # === PRZYPADEK: BRAK WYTRENOWANEJ SIECI ===
        if genome is None and self.best_net is None:
            # Na początku nie mamy jeszcze wytrenowanego modelu
            # Zwróć sensowną losową akcję (nie chaotyczną)
            return np.array([
                np.random.uniform(-0.5, 0.5),  # Łagodne losowe skręty
                np.random.uniform(0.3, 0.7),   # Umiarkowana prędkość (nie pełny gaz)
                np.random.uniform(0.0, 0.2)    # Rzadkie, delikatne hamowanie
            ])
        
        try:
            # === 1. PREPROCESSING OBSERWACJI ===
            inputs = self.preprocess_observation(observation)
            
            # === WALIDACJA WYMIARÓW ===
            expected_inputs = self.config.genome_config.num_inputs
            if len(inputs) != expected_inputs:
                print(f"⚠️ Błąd wymiarów wejścia: otrzymano {len(inputs)}, oczekiwano {expected_inputs}")
                return np.array([0.0, 0.3, 0.0])  # Bezpieczna akcja: prosto + mały gaz
            
            # === 2. AKTYWACJA SIECI NEURONOWEJ ===
            if genome is not None:
                # Użyj konkretnego genomu (podczas oceny populacji)
                net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            else:
                # Użyj najlepszej znalezionej sieci (podczas testowania/gry)
                net = self.best_net
                
            # Przepuść wejścia przez sieć
            outputs = net.activate(inputs)
            
            # === WALIDACJA WYJŚĆ ===
            if len(outputs) < 3:
                print(f"⚠️ Błąd wymiarów wyjścia: otrzymano {len(outputs)}, oczekiwano 3")
                return np.array([0.0, 0.3, 0.0])
            
            # === 3. PRZEKSZTAŁCENIE WYJŚĆ NA AKCJE ===
            # NEAT może produkować dowolne wartości, musimy je ograniczyć
            
            # Steering: [-1, 1] (lewo/prawo)
            # tanh automatycznie daje zakres [-1,1]
            steering = np.clip(np.tanh(outputs[0]), -1.0, 1.0)
            
            # Gas: [0, 1] (brak gazu / pełny gaz)
            # Clip do bezpiecznego zakresu
            gas = np.clip(outputs[1], 0.0, 1.0)
            
            # Brake: [0, 1] (brak hamowania / pełne hamowanie)
            brake = np.clip(outputs[2], 0.0, 1.0)
            
            return np.array([steering, gas, brake])
            
        except Exception as e:
            print(f"❌ Błąd w act(): {e}")
            return np.array([0.0, 0.3, 0.0])  # Akcja awaryjna
    
    def evaluate_genome(self, genome, config, env, num_episodes=1, max_steps=1000):
        """
        Ocena fitness pojedynczego genomu
        
        Args:
            genome: Genom do oceny
            config: Konfiguracja NEAT
            env: Środowisko CarRacing
            num_episodes: Ile razy przetestować genom
            max_steps: Maksymalna długość epizodu
            
        Returns:
            float: Fitness genomu (im wyższa, tym lepszy)
            
        Fitness Design :
        - Nagroda podstawowa: suma reward z środowiska
        - Bonus za długość życia: długo żyjące osobniki = lepsze
        - Bonus za eksplorację: nagroda za odwiedzanie nowych obszarów
        - Kary za stagnację: przerwij jeśli genom się zawiesił
        - Kary za błędy: genomu który crashuje
        
        """
        try:
            # === UTWORZENIE SIECI Z GENOMU ===
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        except Exception as e:
            print(f"❌ Błąd tworzenia sieci: {e}")
            return -1000  # Ciężka kara za nieprawidłowy genom
        
        total_fitness = 0
        
        # === TESTOWANIE NA WIELU EPIZODACH ===
        for episode in range(num_episodes):
            try:
                # === RESET ŚRODOWISKA ===
                observation, info = env.reset()
                episode_reward = 0
                steps = 0
                
                # === ZMIENNE MONITORUJĄCE ===
                negative_steps = 0        # Ile kroków z negatywną nagrodą
                stagnant_steps = 0        # Ile kroków bez postępu w eksploracji
                prev_tiles = info.get('tiles_visited', 0)  # Wcześniej odwiedzone płytki
                
                # === GŁÓWNA PĘTLA EPIZODU ===
                while steps < max_steps:
                    # === WYBÓR AKCJI ===
                    try:
                        inputs = self.preprocess_observation(observation)
                        outputs = net.activate(inputs)
                        
                        # Konwertuj wyjścia sieci na akcję
                        if len(outputs) >= 3:
                            steering = np.clip(np.tanh(outputs[0]), -1.0, 1.0)
                            gas = np.clip(outputs[1], 0.0, 1.0)
                            brake = np.clip(outputs[2], 0.0, 1.0)
                            action = np.array([steering, gas, brake])
                        else:
                            action = np.array([0.0, 0.3, 0.0])  # Akcja awaryjna
                            
                    except Exception as e:
                        print(f"❌ Błąd aktywacji sieci: {e}")
                        action = np.array([0.0, 0.3, 0.0])  # Akcja awaryjna
                    
                    # === KROK W ŚRODOWISKU ===
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    # === MONITORING POSTĘPU EKSPLORACJI ===
                    current_tiles = info.get('tiles_visited', 0)
                    
                    if current_tiles > prev_tiles:
                        # Jest postęp w eksploracji!
                        stagnant_steps = 0  # Reset licznika stagnacji
                        prev_tiles = current_tiles
                    else:
                        # Brak postępu
                        stagnant_steps += 1
                    
                    # === KRYTERIA PRZEDWCZESNEGO ZAKOŃCZENIA ===
                    
                    # 1. Zbyt długa stagnacja eksploracji
                    if stagnant_steps > 100:
                        print(f"   ⏸️ Przerwano: 100 kroków bez nowych płytek")
                        break
                    
                    # 2. Zbyt długo negatywne nagrody (prawdopodobnie wyleciał)
                    if reward < -1.0:
                        negative_steps += 1
                        if negative_steps > 50:
                            print(f"   ⏸️ Przerwano: 50 kroków negatywnych nagród")
                            break
                    else:
                        negative_steps = 0  # Reset przy pozytywnej nagrodzie
                    
                    # 3. Naturalne zakończenie
                    if terminated or truncated:
                        break
                
                # === BONUSY ZA DŁUGOŚĆ ŻYCIA ===
                # Osobniki które długo przetrwają = lepsze
                if steps > 200:
                    episode_reward += 20   # Bonus za przetrwanie >200 kroków
                if steps > 350:
                    episode_reward += 50   # Większy bonus za >350 kroków
                
                # === BONUS ZA EKSPLORACJĘ ===
                # Nagroda za odwiedzanie nowych obszarów toru
                tiles_bonus = current_tiles * 2  # 2 punkty za każdą nową płytkę
                episode_reward += tiles_bonus
                
                total_fitness += episode_reward
                
            except Exception as e:
                print(f"❌ Błąd w epizodzie {episode}: {e}")
                total_fitness -= 100  # Kara za crashowanie
        
        # === ŚREDNIA FITNESS ===
        avg_fitness = total_fitness / num_episodes
        
        # === MINIMALNA GRANICA ===
        # Zapobiega ekstremalnie niskim wartościom które mogą szkodzić ewolucji
        return max(avg_fitness, -1000)
    
    def train(self, env, generations=20):
        """
        Główna pętla treningu ewolucyjnego NEAT
        
        Args:
            env: Środowisko CarRacing
            generations: Liczba generacji do ewolucji
            
        Returns:
            genome: Najlepszy genom po treningu
            
        Proces ewolucji NEAT:
        1. Oceń wszystkie genomy w populacji (evaluate_genome)
        2. Wybierz najlepsze (selekcja)
        3. Stwórz potomstwo przez krzyżowanie i mutację
        4. Zastąp najsłabsze potomstwem
        5. Powtórz dla następnej generacji
        
        NEAT automatycznie:
        - Dodaje nowe neurony (complexification)
        - Dodaje nowe połączenia
        - Mutuje wagi
        - Krzyżuje topologie sieci
        """
        print(f"🧬 Rozpoczynam trening NEAT na {generations} generacji")
        print(f"📊 Populacja: {self.config.pop_size} osobników")
        
        # === CHECKPOINTING ===
        # Automatyczne zapisywanie co 5 generacji (na wypadek crash)
        os.makedirs('checkpoints', exist_ok=True)
        checkpointer = neat.Checkpointer(
            generation_interval=5,                          # Co ile generacji zapisywać
            filename_prefix='checkpoints/neat-checkpoint-'  # Prefix nazwy pliku
        )
        self.population.add_reporter(checkpointer)
        
        def eval_genomes(genomes, config):
            """
            Funkcja oceny całej populacji (wywoływana przez NEAT)
            
            Args:
                genomes: Lista (genome_id, genome) par do oceny
                config: Konfiguracja NEAT
                
            Ta funkcja jest wywoływana przez NEAT dla każdej generacji.
            Musi ustawić genome.fitness dla każdego genomu.
            """
            print(f"🔬 Oceniam {len(genomes)} genomów w generacji {self.generation}...")
            
            for i, (genome_id, genome) in enumerate(genomes):
                # === PROGRESS MONITORING ===
                if i % 20 == 0:  # Pokaż postęp co 20 genomów
                    print(f"   Genom {i+1}/{len(genomes)}")
                
                # === OCENA GENOMU ===
                # To jest główny punkt gdzie genom "gra" w grę
                genome.fitness = self.evaluate_genome(genome, config, env)
                
                # === ŚLEDZENIE NAJLEPSZEGO ===
                if self.best_genome is None or genome.fitness > self.best_genome.fitness:
                    self.best_genome = genome  # Zapisz najlepszy genom
                    self.best_net = neat.nn.FeedForwardNetwork.create(genome, config)
                    print(f"   🌟 Nowy rekord! Fitness: {genome.fitness:.2f}")
            
            # Zwiększ licznik generacji
            self.generation += 1
        
        # === URUCHOMIENIE EWOLUCJI ===
        # To jest główna pętla NEAT - uruchomi eval_genomes dla każdej generacji
        winner = self.population.run(eval_genomes, generations)
        
        # === FINALIZACJA ===
        # Winner to najlepszy genom po wszystkich generacjach
        self.best_genome = winner
        self.best_net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        
        # === AUTOMATYCZNE ZAPISANIE ===
        os.makedirs('models', exist_ok=True)
        self.save_model('models/neat_best.pkl')
        
        print(f"🏆 Trening NEAT zakończony!")
        print(f"📊 Najlepsza fitness: {winner.fitness:.2f}")
        print(f"🧬 Struktura sieci: {len(winner.connections)} połączeń, {len(winner.nodes)} neuronów")
        
        return winner
    
    def save_model(self, filename):
        """
        Zapisywanie najlepszego genomu NEAT
        
        Args:
            filename: Ścieżka do zapisu (.pkl)
            
        Zapisuje:
        - Genom (struktura sieci + wagi)
        - Ścieżkę do konfiguracji (potrzebna do odtworzenia)
        - Fitness (do statystyk)
        
        Format pickle pozwala na dokładne odtworzenie genomu
        """
        if self.best_genome is not None:
            # === TWORZENIE KATALOGU ===
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # === PRZYGOTOWANIE DANYCH ===
            with open(filename, 'wb') as f:
                data = {
                    'genome': self.best_genome,           # Główny genom
                    'config_path': self.config_path,      # Ścieżka konfiguracji
                    'fitness': self.best_genome.fitness   # Fitness dla informacji
                }
                pickle.dump(data, f)
                
            print(f"💾 Model NEAT zapisany: {filename}")
            print(f"📊 Fitness: {self.best_genome.fitness:.2f}")
        else:
            print("⚠️ Brak wytrenowanego genomu do zapisania!")
    
    @classmethod
    def load_model(cls, filename, config_path):
        """
        Ładowanie zapisanego modelu NEAT
        
        Args:
            filename: Ścieżka do pliku .pkl
            config_path: Ścieżka do konfiguracji NEAT
            
        Returns:
            NEATAgent: Agent z wczytanym najlepszym genomem
            
        Proces:
        1. Wczytaj dane z pickle
        2. Stwórz nowego agenta z konfiguracji
        3. Ustaw najlepszy genom
        4. Stwórz sieć neuronową z genomu
        """
        if os.path.exists(filename):
            # === WCZYTYWANIE DANYCH ===
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            print(f"📂 Ładowanie modelu NEAT: {filename}")
            
            # === TWORZENIE AGENTA ===
            agent = cls(config_path)
            
            # === OBSŁUGA RÓŻNYCH FORMATÓW ===
            if isinstance(data, dict):
                # Nowy format - słownik z metadanymi
                agent.best_genome = data['genome']
                fitness = data.get('fitness', 'nieznana')
                print(f"📈 Wczytano model z fitness: {fitness}")
            else:
                # Stary format - bezpośrednio genom
                agent.best_genome = data
                print(f"📈 Wczytano model (stary format)")
            
            # === TWORZENIE SIECI ===
            agent.best_net = neat.nn.FeedForwardNetwork.create(
                agent.best_genome, 
                agent.config
            )
            
            print(f"✅ Model NEAT wczytany pomyślnie!")
            print(f"🧬 Neurony: {len(agent.best_genome.nodes)}")
            print(f"🔗 Połączenia: {len(agent.best_genome.connections)}")
            
            return agent
        else:
            print(f"❌ Plik modelu nie istnieje: {filename}")
            return None
    
    def create_default_config(self, config_path):
        """
        Tworzy domyślną konfigurację NEAT dla CarRacing
        
        Args:
            config_path: Gdzie zapisać konfigurację
            
        Parametry dostrojone dla CarRacing:
        - 7056 wejść (84x84 pikseli)
        - 3 wyjścia (steering, gas, brake)
        - Populacja 150 (kompromis jakość/szybkość)
        - Funkcje aktywacji: tanh, sigmoid, relu
        """
        config_content = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 800.0
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.2
conn_delete_prob        = 0.2

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 7056
num_outputs             = 3

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        
        # === ZAPISYWANIE KONFIGURACJI ===
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"📝 Domyślna konfiguracja NEAT utworzona: {config_path}")

# === PRZYKŁAD UŻYCIA ===
if __name__ == "__main__":
    """
    Przykład treningu NEAT:
    
    from environments.car_racing_env import CarRacingEnv  
    from agents.neat_agent import NEATAgent
    
    env = CarRacingEnv(render_mode=None, continuous=True)
    agent = NEATAgent('configs/neat_config.txt')
    
    # Trening
    winner = agent.train(env, generations=50)
    
    # Test najlepszego
    observation = env.reset()
    action = agent.act(observation)
    """
    print("💡 To jest moduł NEAT Agent. Importuj klasę do głównego skryptu.")