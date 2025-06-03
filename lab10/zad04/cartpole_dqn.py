import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import time

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicjalizacja sieci
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self, num_episodes):
        rewards_history = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.memory) >= self.batch_size:
                    self._update_network()
            
            # Aktualizuj epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Aktualizuj sieć docelową co 10 epizodów
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            rewards_history.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Epizod {episode + 1}, Średnia nagroda (ostatnie 10): {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
        
        return rewards_history
    
    def _update_network(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Pobierz próbkę z pamięci
        batch = self.memory.sample(self.batch_size)
        state_batch = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        action_batch = torch.LongTensor([[x[1]] for x in batch]).to(self.device)
        reward_batch = torch.FloatTensor([[x[2]] for x in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        done_batch = torch.FloatTensor([[x[4]] for x in batch]).to(self.device)
        
        # Oblicz wartości Q
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Oblicz stratę i zaktualizuj sieć
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def play_episode(self, render=True):
        state, info = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                self.env.render()
                time.sleep(0.05)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.policy_net(state_tensor).argmax().item()
            
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        return total_reward

# Utwórz środowisko
env = gym.make("CartPole-v1", render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Utwórz i wytrenuj agenta
agent = DQNAgent(env)
print("\nRozpoczynam trening DQN...")
rewards_history = agent.train(num_episodes=200)

# Wyświetl wykres nagród
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