import numpy as np
import random
import os
from collections import deque
from tensorflow.keras.models import load_model

class DQNAgent:
    
    def __init__(self, state_size, action_size, model, checkpoint_dir='checkpoints/dqn'):
        """
        Inicjalizacja agenta DQN
        
        Args:
            state_size: Rozmiar stanu (84, 84, 1) dla preprocessowanych obrazów
            action_size: Liczba możliwych akcji (5 dla CarRacing)
            model: Skompilowana sieć neuronowa (CNN)
            checkpoint_dir: Katalog do zapisywania modeli
        """
        # === ARCHITEKTURA SIECI ===
        self.state_size = state_size      # (84, 84, 1) - preprocessowany obraz
        self.action_size = action_size    # 5 akcji: nic, lewo, prawo, gaz, hamuj
        self.model = model                # Główna sieć Q-Network
        self.target_model = None          # Sieć docelowa (stabilizuje trening)
        
        # === EXPERIENCE REPLAY ===
        # Bufor cykliczny przechowujący ostatnie 50k doświadczeń (s,a,r,s',done)
        self.memory = deque(maxlen=50000)
        
        # === HYPERPARAMETRY Q-LEARNING ===
        self.gamma = 0.95                 # Współczynnik dyskontowania (jak ważne są nagrody w przyszłości)
        self.epsilon = 1.0                # Początkowa eksploracja (100% losowych akcji)
        self.epsilon_min = 0.1            # Minimalna eksploracja (10% losowych akcji)
        self.epsilon_decay = 0.9995       # Tempo zmniejszania eksploracji (~0.05% co krok)
        self.batch_size = 32              # Rozmiar mini-batcha do treningu
        
        # === KONTROLA TRENINGU ===
        self.checkpoint_dir = checkpoint_dir
        self.steps = 0                    # Licznik kroków treningu
        self.episodes = 0                 # Licznik ukończonych epizodów
        self.update_target_every = 10     # Aktualizuj target network co 10 treningów
        self.train_every = 4              # Trenuj co 4 kroki (stabilność)
        
        # === INICJALIZACJA ===
        # Utwórz katalog na checkpointy jeśli nie istnieje
        import os
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Stwórz target model jako kopię głównego modelu
        self.update_target_model()
    
    def update_target_model(self):
        """
        Aktualizuje target network
        
        Target network to kopia głównego modelu, która jest aktualizowana rzadziej.
        Zapobiega to niestabilności treningu, która występuje gdy sieć uczy się
        ze swoich własnych predykcji.
        
        Mechanizm:
        1. Główna sieć: Q(s,a) - uczona na każdym kroku
        2. Target sieć: Q_target(s',a') - aktualizowana co kilka kroków
        3. Loss = [r + γ * max(Q_target(s',a')) - Q(s,a)]²
        """
        if self.target_model is None:
            # Pierwsza inicjalizacja - sklonuj architekturę
            from tensorflow.keras.models import clone_model
            self.target_model = clone_model(self.model)
            self.target_model.compile(loss='mse', optimizer='adam')
        
        # Skopiuj wagi z głównego modelu do target model
        self.target_model.set_weights(self.model.get_weights())
        print(f"🎯 Target model zaktualizowany (krok {self.steps})")
    
    def act(self, state):
        """
        Wybiera akcję na podstawie stanu (Epsilon-Greedy Strategy)
        
        Args:
            state: Preprocessowany stan środowiska (84, 84, 1)
            
        Returns:
            action: Indeks akcji (0-4)
            
        
        Na początku ε=1.0 (100% eksploracji), kończy na ε=0.1 (10% eksploracji)
        """
        if np.random.rand() <= self.epsilon:
            # EKSPLORACJA: wybierz losową akcję
            action = np.random.randint(0, self.action_size)
            return action
        
        # EKSPLOATACJA: wybierz najlepszą akcję według Q-sieci
        # Dodaj batch dimension (1, 84, 84, 1) - TensorFlow wymaga batch
        state_batch = np.expand_dims(state, axis=0) 
        
        # Oblicz Q-values dla wszystkich akcji
        q_values = self.model.predict(state_batch, verbose=0)  # Shape: (1, 5)
        
        # Zwróć akcję z najwyższą Q-wartością
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        """
        Główna funkcja treningu - zapisuje doświadczenie i uczy sieć
        
        Args:
            state: Stan przed akcją
            action: Wykonana akcja (0-4)
            reward: Otrzymana nagroda
            next_state: Stan po akcji
            done: Czy epizod się skończył
            
        Proces:
        1. Zapisz przejście (s,a,r,s',done) w experience replay buffer
        2. Co 4 kroki: trenuj sieć na losowym batch'u z bufora
        3. Co 10 treningów: aktualizuj target network
        4. Co 5 epizodów: zapisz checkpoint modelu
        """
        # === 1. ZAPISZ DOŚWIADCZENIE ===
        # Experience tuple: (stan, akcja, nagroda, następny_stan, czy_koniec)
        self.memory.append((state, action, reward, next_state, done))
        self.steps += 1
        
        # === 2. TRENING CO 4 KROKI ===
        # Trenuj tylko gdy mamy wystarczająco doświadczeń i co train_every kroków
        if len(self.memory) > self.batch_size and self.steps % self.train_every == 0:
            # Trenuj na losowym batch'u z experience replay
            self.replay(self.batch_size)
            
            # === 3. AKTUALIZACJA TARGET NETWORK ===
            # Co update_target_every treningów skopiuj wagi do target model
            if self.steps % self.update_target_every == 0:
                self.update_target_model()
        
        # === 4. MONITORING I LOGI ===
        # Wyświetl progress co 100 kroków
        if self.steps % 100 == 0:
            print(f"    🔄 Krok: {self.steps}, Epsilon: {self.epsilon:.4f}, "
                  f"Pamięć: {len(self.memory)}/{self.memory.maxlen}")

        # === 5. CHECKPOINT PO EPIZODZIE ===
        # Jeśli epizod się kończy, zwiększ licznik i zapisz model co 5 epizodów
        if done:
            self.episodes += 1
            if self.episodes % 5 == 0:
                self.save_model()
                print(f"💾 Checkpoint: Model zapisany po {self.episodes} epizodach")
    
    def replay(self, batch_size):
        """
        Experience Replay - uczy sieć na losowym batch'u przeszłych doświadczeń
        
        Args:
            batch_size: Rozmiar mini-batcha (32)
            
        Algorytm Q-Learning:
        1. Pobierz losowy batch doświadczeń z bufora
        2. Oblicz Q(s,a) dla obecnych stanów (główna sieć)
        3. Oblicz Q(s',a') dla następnych stanów (target sieć)
        4. Oblicz target: Q_target = r + γ * max(Q_target(s',a'))
        5. Minimalizuj błąd: Loss = [Q_target - Q(s,a)]²
        6. Zmniejsz epsilon (mniej eksploracji z czasem)
        """
        # === 1. LOSOWY BATCH Z EXPERIENCE REPLAY ===
        minibatch = random.sample(self.memory, batch_size)
        
        # === 2. PRZYGOTOWANIE DANYCH WSADOWYCH ===
        # Zamiast 32 pojedynczych predykcji, robimy 2 batch'e (ZNACZNIE SZYBSZE!)
        states = np.array([transition[0] for transition in minibatch])        # (32, 84, 84, 1)
        actions = np.array([transition[1] for transition in minibatch])       # (32,)
        rewards = np.array([transition[2] for transition in minibatch])       # (32,)
        next_states = np.array([transition[3] for transition in minibatch])   # (32, 84, 84, 1)
        dones = np.array([transition[4] for transition in minibatch])         # (32,)
        
        # === 3. OBLICZ Q-VALUES ===
        # TYLKO 2 wywołania predict zamiast 64! (batch processing)
        current_q_values = self.model.predict(states, verbose=0)         # Q(s,a) - (32, 5)
        next_q_values = self.target_model.predict(next_states, verbose=0) # Q_target(s',a') - (32, 5)
        
        # === 4. BELLMAN EQUATION ===
        # Skopiuj obecne Q-values (będziemy modyfikować tylko wybraną akcję)
        target_q_values = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                # Stan terminalny: Q_target = r (brak przyszłych nagród)
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Stan nieterminalny: Q_target = r + γ * max(Q_target(s',a'))
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # === 5. TRENING SIECI ===
        # Minimalizuj błąd między Q(s,a) a Q_target
        # Loss = MSE([target_q_values - current_q_values])
        self.model.fit(states, target_q_values, epochs=1, verbose=0)
        
        # === 6. ZMNIEJSZ EKSPLORACJĘ ===
        # Z czasem zmniejszaj epsilon: agent przechodzi z eksploracji do eksploatacji
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # Po ~1400 krokach epsilon spadnie z 1.0 do ~0.1

    def save_model(self, custom_name=None):
        """
        Zapisuje model wraz z parametrami treningu (checkpointing)
        
        Args:
            custom_name: Opcjonalna nazwa pliku (domyślnie: dqn_model_ep{episodes})
            
        Zapisuje:
        1. Model .keras - architektura sieci + wagi
        2. Parametry .json - stan treningu (episodes, steps, epsilon, itp.)
        
        Dzięki temu można wznowić trening od dowolnego punktu!
        """
        try:
            # === ŚCIEŻKI PLIKÓW ===
            if custom_name:
                filepath = os.path.join(self.checkpoint_dir, f"{custom_name}.keras")
                params_filepath = os.path.join(self.checkpoint_dir, f"{custom_name}_params.json")
            else:
                filepath = os.path.join(self.checkpoint_dir, f"dqn_model_ep{self.episodes}.keras")
                params_filepath = os.path.join(self.checkpoint_dir, f"dqn_model_ep{self.episodes}_params.json")
            
            # === ZAPISZ MODEL ===
            # Keras format - zawiera architekturę + wagi + optimizer state
            self.model.save(filepath)
            
            # === ZAPISZ PARAMETRY TRENINGU ===
            import json
            params = {
                'episodes': self.episodes,      # Ile epizodów przeszedł trening
                'steps': self.steps,            # Ile kroków treningu wykonano
                'epsilon': self.epsilon,        # Obecny poziom eksploracji
                'memory_size': len(self.memory) # Ile doświadczeń w buforze
            }
            with open(params_filepath, 'w') as f:
                json.dump(params, f, indent=2)
                
            print(f"💾 Model zapisany: {filepath}")
            print(f"📊 Parametry: eps={self.episodes}, steps={self.steps}, ε={self.epsilon:.4f}")
            
        except Exception as e:
            print(f"❌ BŁĄD podczas zapisywania: {e}")
    
    @classmethod
    def load(cls, model_path, state_size, action_size):
        """
        Ładuje zapisany model i przywraca stan treningu (resumable training)
        
        Args:
            model_path: Ścieżka do pliku .keras
            state_size: Rozmiar stanu (84, 84, 1)
            action_size: Liczba akcji (5)
            
        Returns:
            DQNAgent: Agent z wczytanym modelem i przywróconymi parametrami
            
        Proces:
        1. Wczytaj model .keras (architektura + wagi)
        2. Stwórz agenta z tym modelem
        3. Przywróć parametry treningu z .json (epsilon, episodes, itp.)
        4. Zrekonstruuj target model
        """
        try:
            # === 1. WCZYTAJ MODEL ===
            from tensorflow.keras.models import load_model
            import json
            
            print(f"📂 Ładowanie modelu z: {model_path}")
            model = load_model(model_path)
            
            # === 2. STWÓRZ AGENTA ===
            agent = cls(state_size, action_size, model)
            
            # === 3. PRZYWRÓĆ PARAMETRY TRENINGU ===
            params_path = model_path.replace('.keras', '_params.json')
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                    
                    # Przywróć stan treningu
                    agent.episodes = params.get('episodes', 0)
                    agent.steps = params.get('steps', 0)
                    agent.epsilon = params.get('epsilon', 1.0)
                    
                    print(f"✅ Model wczytany pomyślnie!")
                    print(f"📊 Przywrócone: epizody={agent.episodes}, "
                          f"kroki={agent.steps}, ε={agent.epsilon:.4f}")
                          
                except Exception as e:
                    print(f"⚠️ Nie udało się wczytać parametrów: {e}")
                    print("🔄 Używam domyślnych parametrów treningu")
            else:
                print(f"⚠️ Brak pliku parametrów: {params_path}")
                print("🔄 Używam domyślnych parametrów")
            
            return agent
            
        except Exception as e:
            print(f"❌ BŁĄD podczas ładowania modelu: {e}")
            raise