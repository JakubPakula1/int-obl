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

# ============================================================================
# DEFINICJE KLAS
# ============================================================================

class DQN(nn.Module):
    """Sieć neuronowa Deep Q-Network"""
    
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
    """Bufor doświadczeń do przechowywania przejść"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Agent DQN do uczenia CartPole"""
    
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicjalizacja sieci neuronowych
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optymalizator i pamięć
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
    
    def choose_action(self, state):
        """Wybór akcji z epsilon-greedy"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self, num_episodes):
        """Główna pętla treningu z zapisywaniem najlepszego modelu"""
        rewards_history = []
        best_avg_reward = 0
        best_episode = 0
        best_model_state = None

        for episode in range(num_episodes):
            # Reset środowiska
            state, info = self.env.reset()
            total_reward = 0
            done = False
            
            # Główna pętla epizodu
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Zapisz doświadczenie w buforze
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Ucz sieć jeśli mamy wystarczająco danych
                if len(self.memory) >= self.batch_size:
                    self._update_network()
            
            # Aktualizuj parametry
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Aktualizuj sieć docelową co 10 epizodów
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            rewards_history.append(total_reward)
            
            # Sprawdź postęp co 10 epizodów
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Epizod {episode + 1}, Średnia nagroda (ostatnie 10): {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
                
                # Zapisz najlepszy model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_episode = episode + 1
                    best_model_state = self.policy_net.state_dict().copy()
                    print(f"🏆 NOWY REKORD! Najlepsza średnia: {best_avg_reward:.2f} (epizod {best_episode})")
                    torch.save(best_model_state, 'best_cartpole_model.pth')
        
        # Załaduj najlepszy model na koniec treningu
        if best_model_state is not None:
            self.policy_net.load_state_dict(best_model_state)
            print(f"\n📊 PODSUMOWANIE TRENINGU:")
            print(f"Najlepsza średnia w całym treningu: {best_avg_reward:.2f} (epizod {best_episode})")
            print(f"Ostatnia średnia (ostatnie 10 epizodów): {np.mean(rewards_history[-10:]):.2f}")
            print(f"✅ Załadowano najlepszy model do pamięci")
        
        return rewards_history, best_avg_reward
    
    def _update_network(self):
        """Aktualizacja sieci neuronowej"""
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
    
    def play_episode(self, render=True, show_details=False):
        """Zagraj jeden epizod z aktualną polityką"""
        state, info = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            if render:
                self.env.render()
                time.sleep(0.05)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            if show_details and steps % 10 == 0:
                print(f"Krok {steps}:")
                print(f"  Stan: [pozycja={state[0]:.3f}, prędkość={state[1]:.3f}, kąt={state[2]:.3f}, prędkość_kątowa={state[3]:.3f}]")
                print(f"  Q-wartości: {q_values.cpu().numpy()[0]}")
                print(f"  Wybrana akcja: {'LEWO' if action == 0 else 'PRAWO'}")
            
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        return total_reward, steps

# ============================================================================
# GŁÓWNY PROGRAM
# ============================================================================

def main():
    # Inicjalizacja środowiska
    env = gym.make("CartPole-v1")
    print(f"Środowisko: {env.spec.id}")
    print(f"Przestrzeń obserwacji: {env.observation_space}")
    print(f"Przestrzeń akcji: {env.action_space}")

    # Trening agenta
    agent = DQNAgent(env)
    print("\nRozpoczynam trening DQN...")
    start_time = time.time()
    rewards_history, best_avg_reward = agent.train(num_episodes=1000)
    end_time = time.time()

    print(f"\nTrening zakończony w {end_time - start_time:.2f} sekund")
    print(f"Najlepsza średnia nagroda w całym treningu: {best_avg_reward:.2f}")

    env.close()

    # Wykresy postępu
    plot_training_progress(rewards_history)
    
    # Demonstracja najlepszego rozwiązania
    demonstrate_best_solution(agent, best_avg_reward)

def plot_training_progress(rewards_history):
    """Wyświetl wykresy postępu treningu"""
    plt.figure(figsize=(12, 5))
    
    # Historia nagród
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Historia nagród podczas treningu")
    plt.xlabel("Epizod")
    plt.ylabel("Suma nagród")
    plt.grid(True)

    # Średnia krocząca
    window_size = 10
    moving_avg = [np.mean(rewards_history[max(0, i-window_size):i+1]) for i in range(len(rewards_history))]
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg)
    plt.title(f"Średnia krocząca (okno {window_size})")
    plt.xlabel("Epizod")
    plt.ylabel("Średnia suma nagród")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def demonstrate_best_solution(agent, best_avg_reward):
    """Demonstracja najlepszego rozwiązania"""
    print("\n" + "="*60)
    print("WIZUALNA DEMONSTRACJA NAJLEPSZEGO ROZWIĄZANIA DQN")
    print("="*60)

    # Test bez wizualizacji
    test_performance_without_rendering(agent, best_avg_reward)
    
    # Wizualna demonstracja
    visual_demonstration(agent)

def test_performance_without_rendering(agent, best_avg_reward):
    """Test wydajności bez wizualizacji"""
    print("Testowanie modelu bez wizualizacji...")
    test_env = gym.make("CartPole-v1")
    agent.env = test_env
    agent.epsilon = 0  # Wyłącz eksplorację

    pre_test_rewards = []
    for i in range(10):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        pre_test_rewards.append(total_reward)

    test_env.close()

    avg_test_reward = np.mean(pre_test_rewards)
    print(f"Średnia nagroda w teście: {avg_test_reward:.1f}")
    print(f"Najlepszy wynik: {max(pre_test_rewards)}")
    print(f"Najgorszy wynik: {min(pre_test_rewards)}")

    # Fallback do zapisanego modelu jeśli potrzeba
    if avg_test_reward < 400 and best_avg_reward >= 400:
        try:
            agent.policy_net.load_state_dict(torch.load('best_cartpole_model.pth'))
            print("🔄 Załadowano zapisany najlepszy model")
        except:
            print("⚠️ Nie można załadować zapisanego modelu")

def visual_demonstration(agent):
    """Wizualna demonstracja z renderowaniem"""
    visual_env = gym.make("CartPole-v1", render_mode="human")
    agent.env = visual_env
    agent.epsilon = 0  # Wyłącz eksplorację

    print("\nNaciśnij Enter, aby rozpocząć wizualną demonstrację...")
    input()

    test_rewards = []
    test_steps = []

    for episode in range(5):
        print(f"\n=== EPIZOD {episode + 1} ===")
        
        state, _ = visual_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        print(f"Stan początkowy: [pos={state[0]:.3f}, vel={state[1]:.3f}, angle={state[2]:.3f}, ang_vel={state[3]:.3f}]")
        
        while not done:
            visual_env.render()
            time.sleep(0.03)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            # Szczegóły dla pierwszego epizodu
            if episode == 0 and steps % 50 == 0:
                print(f"Krok {steps}: Q-wartości=[{q_values.cpu().numpy()[0][0]:.3f}, {q_values.cpu().numpy()[0][1]:.3f}], "
                      f"Akcja={'LEWO' if action == 0 else 'PRAWO'}")
            
            state, reward, terminated, truncated, _ = visual_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
            
            if steps > 1000:  # Limit kroków
                print("Epizod przerwany po 1000 krokach")
                break
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        
        # Ocena wyniku
        if total_reward >= 500:
            print(f"🎉 SUKCES! Epizod trwał {steps} kroków")
        elif total_reward >= 200:
            print(f"👍 DOBRY WYNIK! {steps} kroków")
        else:
            print(f"⚠️ Słup upadł po {steps} krokach")
        
        print(f"Suma nagród: {total_reward}")
        
        if episode < 4:
            print("\nNaciśnij Enter dla następnego epizodu...")
            input()

    visual_env.close()
    
    # Podsumowanie demonstracji
    print_demonstration_summary(test_rewards, test_steps)

def print_demonstration_summary(test_rewards, test_steps):
    """Wydrukuj podsumowanie demonstracji"""
    print(f"\n=== PODSUMOWANIE DEMONSTRACJI ===")
    
    # Wyniki każdego epizodu
    for i, (reward, steps) in enumerate(zip(test_rewards, test_steps)):
        status = "🎉 SUKCES" if reward >= 500 else "👍 DOBRY" if reward >= 200 else "⚠️ SŁABY"
        print(f"  Epizod {i+1}: {steps:3d} kroków, {reward:3.0f} nagród {status}")

    # Statystyki ogólne
    print(f"\nStatystyki:")
    print(f"Średnia liczba kroków: {np.mean(test_steps):.1f}")
    print(f"Średnia suma nagród: {np.mean(test_rewards):.1f}")
    print(f"Najdłuższy epizod: {max(test_steps)} kroków")
    print(f"Najkrótszy epizod: {min(test_steps)} kroków")

    # Analiza sukcesu
    successful_episodes = sum(1 for r in test_rewards if r >= 500)
    good_episodes = sum(1 for r in test_rewards if r >= 200)

    print(f"\nWyniki:")
    print(f"Epizody sukcesu (≥500 kroków): {successful_episodes}/5 ({successful_episodes/5*100:.0f}%)")
    print(f"Epizody dobre (≥200 kroków): {good_episodes}/5 ({good_episodes/5*100:.0f}%)")

    # Ostateczna ocena
    if successful_episodes >= 4:
        print("🏆 DOSKONAŁY WYNIK! Model w pełni opanował zadanie!")
    elif successful_episodes >= 2:
        print("🎯 ŚWIETNY WYNIK! Model bardzo dobrze balansuje!")
    elif good_episodes >= 3:
        print("👍 DOBRY WYNIK! Model pokazuje solidne umiejętności!")
    else:
        print("🔄 Model potrzebuje więcej treningu...")

if __name__ == "__main__":
    main()