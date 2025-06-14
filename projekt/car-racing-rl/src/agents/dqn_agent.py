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
        self.memory = deque(maxlen=10000)  # Użycie deque z maxlen
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
        q_values = self.model.predict(state, verbose=0)  # Dodanie verbose=0
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
        
        # Zbieranie danych do wsadowego treningu
        states = np.zeros((batch_size,) + self.state_size)
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            target = reward
            if not done:
                # Użycie target model do obliczenia przyszłych Q-wartości
                target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            targets[i] = self.model.predict(state, verbose=0)
            targets[i, action] = target
        
        # Jednorazowe trenowanie na całym batch
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        # Zmniejszanie epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, custom_name=None):
        """Zapisuje model w katalogu checkpointów"""
        try:
            if custom_name:
                filepath = os.path.join(self.checkpoint_dir, f"{custom_name}.keras")
            else:
                filepath = os.path.join(self.checkpoint_dir, f"dqn_model_ep{self.episodes}.keras")
            self.model.save(filepath)
            print(f"Model zapisany w: {filepath}")
        except Exception as e:
            print(f"BŁĄD podczas zapisywania modelu: {e}")
    
    @classmethod
    def load(cls, model_path, state_size, action_size):
        """Ładuje zapisany wcześniej model i tworzy nowego agenta"""
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        agent = cls(state_size, action_size, model)
        print(f"Model wczytany z: {model_path}")
        return agent