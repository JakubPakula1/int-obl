import gymnasium as gym
import numpy as np
import time
from IPython.display import clear_output

def value_iteration(env, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Implementacja algorytmu Value Iteration.
    
    Args:
        env: Środowisko Gym
        gamma: Współczynnik dyskontowania
        theta: Próg zbieżności
        max_iterations: Maksymalna liczba iteracji
    
    Returns:
        policy: Optymalna polityka
        V: Wartości stanów
    """
    # Inicjalizacja tablicy wartości
    V = np.zeros(env.observation_space.n)
    
    for i in range(max_iterations):
        delta = 0
        
        # Dla każdego stanu
        for state in range(env.observation_space.n):
            v = V[state]
            
            # Oblicz wartość dla każdej akcji
            action_values = []
            for action in range(env.action_space.n):
                # Symuluj akcję
                env.reset()
                env.unwrapped.s = state
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Oblicz wartość akcji
                action_value = reward + gamma * V[next_state]
                action_values.append(action_value)
            
            # Aktualizuj wartość stanu
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))
        
        # Sprawdź zbieżność
        if delta < theta:
            print(f"Zbieżność osiągnięta po {i+1} iteracjach")
            break
    
    # Wyznacz optymalną politykę
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        action_values = []
        for action in range(env.action_space.n):
            env.reset()
            env.unwrapped.s = state
            next_state, reward, terminated, truncated, _ = env.step(action)
            action_value = reward + gamma * V[next_state]
            action_values.append(action_value)
        policy[state] = np.argmax(action_values)
    
    return policy, V

def play_episode(env, policy, render=True):
    """Odtwórz epizod używając danej polityki"""
    observation, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if render:
            env.render()
            time.sleep(0.5)
        
        action = policy[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    return total_reward

# Utwórz środowisko
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Uruchom Value Iteration
print("\nRozpoczynam Value Iteration...")
policy, V = value_iteration(env)

# Wyświetl wartości stanów
print("\nWartości stanów:")
print(V.reshape(4, 4))  # Przekształć na macierz 4x4

# Wyświetl politykę
print("\nOptymalna polityka:")
policy_display = np.array(['←', '↓', '→', '↑'])[policy]
print(policy_display.reshape(4, 4))  # Przekształć na macierz 4x4

# Odtwórz kilka epizodów
print("\nOdtwarzanie epizodów z optymalną polityką:")
for episode in range(5):
    reward = play_episode(env, policy)
    print(f"Epizod {episode+1}: Suma nagród = {reward}")

env.close() 