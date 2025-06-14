import numpy as np
import random
import os
from collections import deque
from tensorflow.keras.models import load_model

class DQNAgent:
    def __init__(self, state_size, action_size, model, checkpoint_dir='checkpoints/dqn'):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.target_model = None  # Zostanie utworzony przy pierwszym update_target_model
        self.memory = deque(maxlen=50000)  # Użycie deque z maxlen
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.batch_size = 32
        self.checkpoint_dir = checkpoint_dir
        self.steps = 0
        self.episodes = 0
        self.update_target_every = 10  # Aktualizacja target model co 10 kroków treningu
        self.train_every = 4  # Trenowanie co 4 kroki
        
        # Utworzenie folderu na checkpointy
        import os
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Utworzenie target model jako kopii głównego modelu
        self.update_target_model()
    
    def update_target_model(self):
        """Aktualizuje target model na podstawie głównego modelu"""
        if self.target_model is None:
            # Użyj clone_model z Keras do stworzenia kopii architektury modelu
            from tensorflow.keras.models import clone_model
            self.target_model = clone_model(self.model)
            self.target_model.compile(loss='mse', optimizer='adam')
        
        # Kopiowanie wag z modelu głównego do target modelu
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Dodaj batch dimension dla predict()
        state_batch = np.expand_dims(state, axis=0)  # (84, 84, 1) -> (1, 84, 84, 1)
        q_values = self.model.predict(state_batch, verbose=0)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))
        self.steps += 1
        
        # Trening tylko co kilka kroków
        if len(self.memory) > self.batch_size and self.steps % self.train_every == 0:
            self.replay(self.batch_size)
            
            # Aktualizacja target network
            if self.steps % self.update_target_every == 0:
                self.update_target_model()
            # Dodaj logi debugowania
        if self.steps % 100 == 0:
            print(f"    Krok treningu: {self.steps}, Epsilon: {self.epsilon:.4f}, Pamięć: {len(self.memory)}")

        # Jeśli epizod się kończy, zwiększamy licznik i zapisujemy model
        if done:
            self.episodes += 1
            if self.episodes % 5 == 0:
                self.save_model()
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        # Przygotowanie danych wsadowych - wydajniejsze
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # TYLKO 2 wywołania predict zamiast 64!
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        target_q_values = current_q_values.copy()
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Trenowanie modelu
        self.model.fit(states, target_q_values, epochs=1, verbose=0)
        
        # Zmniejszanie epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, custom_name=None):
        """Zapisuje model wraz z parametrami treningu"""
        try:
            if custom_name:
                filepath = os.path.join(self.checkpoint_dir, f"{custom_name}.keras")
                params_filepath = os.path.join(self.checkpoint_dir, f"{custom_name}_params.json")
            else:
                filepath = os.path.join(self.checkpoint_dir, f"dqn_model_ep{self.episodes}.keras")
                params_filepath = os.path.join(self.checkpoint_dir, f"dqn_model_ep{self.episodes}_params.json")
            
            # Zapisz model
            self.model.save(filepath)
            
            # Zapisz parametry treningu
            import json
            params = {
                'episodes': self.episodes,
                'steps': self.steps,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory)
            }
            with open(params_filepath, 'w') as f:
                json.dump(params, f)
                
            print(f"Model i parametry zapisane w: {filepath}")
        except Exception as e:
            print(f"BŁĄD podczas zapisywania modelu: {e}")
    
    @classmethod
    def load(cls, model_path, state_size, action_size):
        """Ładuje zapisany wcześniej model i przywraca stan treningu"""
        from tensorflow.keras.models import load_model
        import json
        
        model = load_model(model_path)
        agent = cls(state_size, action_size, model)
        
        # Próbuj wczytać parametry treningu
        params_path = model_path.replace('.keras', '_params.json')
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                agent.episodes = params.get('episodes', 0)
                agent.steps = params.get('steps', 0)
                agent.epsilon = params.get('epsilon', 1.0)
                
                print(f"Model wczytany z: {model_path}")
                print(f"Przywrócone parametry: epizody={agent.episodes}, epsilon={agent.epsilon:.4f}")
            except Exception as e:
                print(f"Nie udało się wczytać parametrów: {e}")
                print("Używam domyślnych parametrów")
        else:
            print("Brak pliku z parametrami - używam domyślnych")
        
        return agent