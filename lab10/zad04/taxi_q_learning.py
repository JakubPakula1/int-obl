import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicjalizacja tablicy Q
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        
        # Statystyki uczenia
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'epsilons': [],
            'steps': []
        }
    
    def choose_action(self, state):
        """Strategia epsilon-zachłanna"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Eksploracja
        else:
            return np.argmax(self.q_table[state])  # Eksploatacja
    
    def learn(self, state, action, reward, next_state, done):
        """Aktualizacja Q-learning"""
        # Obecna wartość Q
        old_value = self.q_table[state, action]
        
        # Najlepsza wartość Q dla następnego stanu
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Aktualizacja Q(s,a)
        self.q_table[state, action] = old_value + self.lr * (target - old_value)
    
    def train(self, num_episodes, verbose=True):
        """Trenowanie agenta"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Aktualizuj epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Zapisz statystyki
            self.training_stats['episodes'].append(episode)
            self.training_stats['rewards'].append(total_reward)
            self.training_stats['epsilons'].append(self.epsilon)
            self.training_stats['steps'].append(steps)
            
            if verbose and (episode + 1) % 200 == 0:
                avg_reward = np.mean(self.training_stats['rewards'][-100:])
                avg_steps = np.mean(self.training_stats['steps'][-100:])
                print(f"Epizod {episode + 1:4d} | "
                      f"Śr. nagroda: {avg_reward:6.1f} | "
                      f"Śr. kroki: {avg_steps:5.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return self.training_stats
    
    def evaluate(self, num_episodes=100, render=False):
        """Ewaluuj wyuczoną politykę"""
        old_epsilon = self.epsilon
        self.epsilon = 0  # Wyłącz eksplorację
        
        total_rewards = []
        total_steps = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                if render and episode < 3:  # Pokaż pierwsze 3 epizody
                    self.env.render()
                    time.sleep(0.3)
                
                action = self.choose_action(state)  # Zawsze zachłanny
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                if steps > 200:  # Zabezpieczenie przed nieskończonymi pętlami
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
        
        self.epsilon = old_epsilon  # Przywróć epsilon
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': np.mean([r > 0 for r in total_rewards])
        }

def plot_training_progress(stats):
    """Wykresy postępu treningu"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Nagrody
    axes[0,0].plot(stats['episodes'], stats['rewards'], alpha=0.6)
    # Średnia krocząca
    window = 100
    if len(stats['rewards']) >= window:
        moving_avg = np.convolve(stats['rewards'], np.ones(window)/window, mode='valid')
        axes[0,0].plot(stats['episodes'][window-1:], moving_avg, 'r-', linewidth=2)
    axes[0,0].set_title('Nagrody podczas treningu')
    axes[0,0].set_xlabel('Epizod')
    axes[0,0].set_ylabel('Suma nagród')
    axes[0,0].grid(True)
    
    # Liczba kroków
    axes[0,1].plot(stats['episodes'], stats['steps'], alpha=0.6)
    if len(stats['steps']) >= window:
        moving_avg = np.convolve(stats['steps'], np.ones(window)/window, mode='valid')
        axes[0,1].plot(stats['episodes'][window-1:], moving_avg, 'r-', linewidth=2)
    axes[0,1].set_title('Liczba kroków')
    axes[0,1].set_xlabel('Epizod')
    axes[0,1].set_ylabel('Kroki')
    axes[0,1].grid(True)
    
    # Epsilon
    axes[1,0].plot(stats['episodes'], stats['epsilons'])
    axes[1,0].set_title('Zanik epsilon (eksploracja)')
    axes[1,0].set_xlabel('Epizod')
    axes[1,0].set_ylabel('Epsilon')
    axes[1,0].grid(True)
    
    # Histogram nagród
    axes[1,1].hist(stats['rewards'][-1000:], bins=50, alpha=0.7)
    axes[1,1].set_title('Rozkład nagród (ostatnie 1000 epizodów)')
    axes[1,1].set_xlabel('Suma nagród')
    axes[1,1].set_ylabel('Częstość')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Główny test
print("🚕 Q-LEARNING - TAXI")
print("="*50)

# Stwórz środowisko
env = gym.make("Taxi-v3")
print(f"Środowisko: {env.spec.id}")
print(f"Stany: {env.observation_space.n}")
print(f"Akcje: {env.action_space.n}")
print("\nAkcje: 0=Południe, 1=Północ, 2=Wschód, 3=Zachód, 4=Podnieś, 5=Opuść")

# Trenowanie z różnymi parametrami
configs = [
    {"lr": 0.1, "gamma": 0.99, "eps_decay": 0.995, "name": "Standardowe"},
    {"lr": 0.5, "gamma": 0.99, "eps_decay": 0.99, "name": "Szybkie uczenie"},
    {"lr": 0.05, "gamma": 0.95, "eps_decay": 0.999, "name": "Konserwatywne"}
]

best_agent = None
best_score = -float('inf')

for config in configs:
    print(f"\n--- Test konfiguracji: {config['name']} ---")
    
    agent = QLearningAgent(env, 
                          learning_rate=config['lr'],
                          discount_factor=config['gamma'],
                          epsilon_decay=config['eps_decay'])
    
    # Trenowanie
    stats = agent.train(num_episodes=2000, verbose=False)
    
    # Ewaluacja
    eval_results = agent.evaluate(num_episodes=100)
    
    print(f"Średnia nagroda: {eval_results['avg_reward']:.1f} ± {eval_results['std_reward']:.1f}")
    print(f"Średnie kroki: {eval_results['avg_steps']:.1f} ± {eval_results['std_steps']:.1f}")
    print(f"Wskaźnik sukcesu: {eval_results['success_rate']:.1%}")
    
    if eval_results['avg_reward'] > best_score:
        best_score = eval_results['avg_reward']
        best_agent = agent

# Wykresy dla najlepszego agenta
print(f"\n🏆 Najlepszy wynik: {best_score:.1f}")
plot_training_progress(best_agent.training_stats)

# Demo najlepszego agenta
print("\n🎮 Demo najlepszego agenta:")
env_render = gym.make("Taxi-v3", render_mode="human")
best_agent.env = env_render
best_agent.evaluate(num_episodes=3, render=True)
env_render.close()

env.close()