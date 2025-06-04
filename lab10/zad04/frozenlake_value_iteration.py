import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        Inicjalizacja Value Iteration
        
        Args:
            env: środowisko Gym
            gamma: współczynnik dyskontowania
            theta: próg konwergencji
        """
        self.env = env
        # Uzyskanie dostępu do podstawowego środowiska (bez wrapperów)
        self.unwrapped_env = env.unwrapped
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        
        # Inicjalizacja tabeli wartości
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)
        
    def value_iteration(self):
        """
        Implementacja algorytmu Value Iteration
        """
        iteration = 0
        convergence_history = []
        
        while True:
            delta = 0
            old_V = self.V.copy()
            
            # Aktualizacja wartości dla każdego stanu
            for s in range(self.n_states):
                v = self.V[s]
                action_values = []
                
                # Obliczenie wartości dla każdej akcji
                for a in range(self.n_actions):
                    action_value = 0
                    
                    # Suma po wszystkich możliwych przejściach
                    # Używamy unwrapped_env aby uzyskać dostęp do P
                    for prob, next_state, reward, done in self.unwrapped_env.P[s][a]:
                        action_value += prob * (reward + self.gamma * self.V[next_state])
                    
                    action_values.append(action_value)
                
                # Wybór najlepszej wartości (maksimum)
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            convergence_history.append(delta)
            iteration += 1
            
            print(f"Iteracja {iteration}: delta = {delta:.6f}")
            
            # Sprawdzenie konwergencji
            if delta < self.theta:
                break
        
        print(f"Algorytm skonwergował po {iteration} iteracjach")
        return convergence_history
    
    def extract_policy(self):
        """
        Wyodrębnienie optymalnej polityki z funkcji wartości
        """
        for s in range(self.n_states):
            action_values = []
            
            for a in range(self.n_actions):
                action_value = 0
                for prob, next_state, reward, done in self.unwrapped_env.P[s][a]:
                    action_value += prob * (reward + self.gamma * self.V[next_state])
                action_values.append(action_value)
            
            # Wybór najlepszej akcji
            self.policy[s] = np.argmax(action_values)
    
    def evaluate_policy(self, episodes=1000, render=False):
        """
        Ocena wyuczonej polityki
        """
        total_rewards = []
        success_rate = 0
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            max_steps = 100
            
            while not done and steps < max_steps:
                action = self.policy[state]
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                if render and episode < 5:
                    self.env.render()
                    time.sleep(0.5)
            
            total_rewards.append(total_reward)
            if total_reward > 0:
                success_rate += 1
        
        success_rate = success_rate / episodes * 100
        avg_reward = np.mean(total_rewards)
        
        return avg_reward, success_rate, total_rewards
    
    def visualize_value_function(self):
        """
        Wizualizacja funkcji wartości dla FrozenLake 4x4
        """
        if self.n_states != 16:  # Tylko dla FrozenLake 4x4
            print("Wizualizacja dostępna tylko dla FrozenLake 4x4")
            return
        
        # Przekształcenie do macierzy 4x4
        value_grid = self.V.reshape(4, 4)
        policy_grid = self.policy.reshape(4, 4)
        
        # Mapowanie akcji na strzałki
        action_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Wizualizacja funkcji wartości
        im1 = ax1.imshow(value_grid, cmap='viridis', interpolation='nearest')
        ax1.set_title('Funkcja Wartości V(s)')
        
        # Dodanie wartości do każdej komórki
        for i in range(4):
            for j in range(4):
                ax1.text(j, i, f'{value_grid[i, j]:.3f}', 
                        ha='center', va='center', color='white', fontsize=8)
        
        ax1.set_xticks(range(4))
        ax1.set_yticks(range(4))
        plt.colorbar(im1, ax=ax1)
        
        # Wizualizacja polityki
        ax2.imshow(np.zeros((4, 4)), cmap='gray', alpha=0.3)
        ax2.set_title('Optymalna Polityka π(s)')
        
        for i in range(4):
            for j in range(4):
                action = policy_grid[i, j]
                arrow = action_map[action]
                ax2.text(j, i, arrow, ha='center', va='center', 
                        fontsize=20, color='blue')
        
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self, convergence_history):
        """
        Wykres konwergencji algorytmu
        """
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_history)
        plt.xlabel('Iteracja')
        plt.ylabel('Delta (różnica wartości)')
        plt.title('Konwergencja Value Iteration')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

def main():
    # Utworzenie środowiska FrozenLake
    # Usunięcie render_mode='human' z głównego środowiska
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", 
                   is_slippery=True)
    
    print("=== VALUE ITERATION ALGORITHM ===")
    print(f"Środowisko: FrozenLake 4x4")
    print(f"Liczba stanów: {env.observation_space.n}")
    print(f"Liczba akcji: {env.action_space.n}")
    print(f"Slippery: True")
    print()
    
    # Inicjalizacja i uruchomienie Value Iteration
    vi = ValueIteration(env, gamma=0.9, theta=1e-6)
    
    print("Rozpoczynam Value Iteration...")
    convergence_history = vi.value_iteration()
    
    print("\nWyodrębnianie optymalnej polityki...")
    vi.extract_policy()
    
    # Ocena polityki
    print("\nOcena wyuczonej polityki...")
    avg_reward, success_rate, rewards = vi.evaluate_policy(episodes=1000)
    
    print(f"Średnia nagroda: {avg_reward:.4f}")
    print(f"Wskaźnik sukcesu: {success_rate:.2f}%")
    
    # Wizualizacje
    print("\nGenerowanie wizualizacji...")
    vi.visualize_value_function()
    vi.plot_convergence(convergence_history)
    
    # Demonstracja najlepszego rozwiązania
    print("\n=== DEMONSTRACJA NAJLEPSZEGO ROZWIĄZANIA ===")
    print("Uruchamiam 5 epizodów z wizualizacją...")
    
    # Osobne środowisko do wizualizacji
    env_render = gym.make('FrozenLake-v1', desc=None, map_name="4x4", 
                         is_slippery=True, render_mode='human')
    
    for episode in range(5):
        print(f"\nEpizod {episode + 1}:")
        state, _ = env_render.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"Stan początkowy: {state}")
        
        while not done and steps < 100:
            action = vi.policy[state]
            action_names = ['Lewo', 'Dół', 'Prawo', 'Góra']
            print(f"Krok {steps + 1}: Stan {state} -> Akcja: {action_names[action]}")
            
            state, reward, terminated, truncated, _ = env_render.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if done:
                if reward > 0:
                    print(f"SUKCES! Osiągnięto cel w {steps} krokach!")
                else:
                    print(f"Porażka w kroku {steps}")
                print(f"Całkowita nagroda: {total_reward}")
        
        time.sleep(1)
    
    env.close()
    env_render.close()

if __name__ == "__main__":
    main()