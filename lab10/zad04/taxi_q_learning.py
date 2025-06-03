import gymnasium as gym
import numpy as np
import time
from IPython.display import clear_output

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.lr = learning_rate  # Współczynnik uczenia
        self.gamma = discount_factor  # Współczynnik dyskontowania
        self.epsilon = epsilon  # Prawdopodobieństwo eksploracji
        self.epsilon_decay = epsilon_decay  # Współczynnik zaniku epsilon
        self.epsilon_min = epsilon_min  # Minimalna wartość epsilon
        
        # Inicjalizacja tablicy Q
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    def choose_action(self, state):
        """Wybierz akcję używając strategii epsilon-zachłannej"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Eksploracja
        else:
            return np.argmax(self.q_table[state])  # Eksploatacja
    
    def learn(self, state, action, reward, next_state, done):
        """Aktualizuj wartości Q"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Wzór Q-learning
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max * (not done))
        self.q_table[state, action] = new_value
        
        # Aktualizuj epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, num_episodes):
        """Trenowanie agenta"""
        rewards_history = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            
            rewards_history.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Epizod {episode + 1}, Średnia nagroda (ostatnie 100): {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
        
        return rewards_history
    
    def play_episode(self, render=True):
        """Odtwórz epizod używając wyuczonej polityki"""
        state, info = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                self.env.render()
                time.sleep(0.5)
            
            action = np.argmax(self.q_table[state])
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        return total_reward

# Utwórz środowisko
env = gym.make("Taxi-v3", render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Utwórz i wytrenuj agenta
agent = QLearningAgent(env)
print("\nRozpoczynam trening Q-learning...")
rewards_history = agent.train(num_episodes=1000)

# Wyświetl wykres nagród
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.title("Historia nagród podczas treningu")
plt.xlabel("Epizod")
plt.ylabel("Suma nagród")
plt.show()

# Odtwórz kilka epizodów
print("\nOdtwarzanie epizodów z wyuczoną polityką:")
for episode in range(5):
    reward = agent.play_episode()
    print(f"Epizod {episode+1}: Suma nagród = {reward}")

env.close() 