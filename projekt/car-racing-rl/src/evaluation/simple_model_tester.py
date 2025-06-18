import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

def test_all_agents_directly(episodes=5):
    """Test wszystkich agentów bezpośrednio przez import"""
    results = {}
    
    print("🎯 BEZPOŚREDNIE TESTOWANIE AGENTÓW")
    print("=" * 50)
    
    # 1. Test DQN
    print("\n🧪 Testowanie DQN...")
    try:
        from environments.car_racing_env import CarRacingEnv
        from agents.dqn_agent import DQNAgent
        from training.train_dqn import preprocess_state
        
        env = CarRacingEnv(render_mode=None, continuous=False)
        agent = DQNAgent.load('models/dqn_model_ep210.keras', (84, 84, 1), 5)
        agent.epsilon = 0.0  # Tylko eksploatacja
        
        dqn_rewards = []
        for episode in range(episodes):
            observation, info = env.reset()
            state = preprocess_state(observation)
            episode_reward = 0
            
            for step in range(1000):
                action = agent.act(state)
                observation, reward, terminated, truncated, info = env.step(action)
                state = preprocess_state(observation)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            dqn_rewards.append(episode_reward)
            if episode % 10 == 0:
                print(f"  Epizod {episode+1}: {episode_reward:.2f}")
        
        results['dqn'] = {
            'rewards': dqn_rewards,
            'mean': np.mean(dqn_rewards),
            'std': np.std(dqn_rewards),
            'completion_rate': sum(1 for r in dqn_rewards if r > 600) / episodes * 100
        }
        env.close()
        print(f"✅ DQN: średnia = {results['dqn']['mean']:.2f}")
        
    except Exception as e:
        print(f"❌ DQN błąd: {e}")
        results['dqn'] = None
    
    # 2. Test NEAT
    print("\n🧪 Testowanie NEAT...")
    try:
        from environments.car_racing_env import CarRacingEnv
        from agents.neat_agent import NEATAgent
        
        env = CarRacingEnv(render_mode=None, continuous=True)
        agent = NEATAgent.load_model('models/neat_best.pkl', 'configs/neat_config.txt')
        
        neat_rewards = []
        for episode in range(episodes):
            observation, info = env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = agent.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            neat_rewards.append(episode_reward)
            if episode % 10 == 0:
                print(f"  Epizod {episode+1}: {episode_reward:.2f}")
        
        results['neat'] = {
            'rewards': neat_rewards,
            'mean': np.mean(neat_rewards),
            'std': np.std(neat_rewards),
            'completion_rate': sum(1 for r in neat_rewards if r > 600) / episodes * 100
        }
        env.close()
        print(f"✅ NEAT: średnia = {results['neat']['mean']:.2f}")
        
    except Exception as e:
        print(f"❌ NEAT błąd: {e}")
        results['neat'] = None
    
    # 3. Test PPO (Stable Baselines3)
    print("\n🧪 Testowanie PPO (Stable Baselines3)...")
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
        
        # Sprawdź czy model istnieje
        model_path = "models/ppo_carracing1"
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Model PPO nie znaleziony: {model_path}.zip")
            results['ppo'] = None
        else:
            # Wczytaj model
            model = PPO.load(model_path)
            print(f"✅ Model PPO wczytany: {model_path}")
            
            # Stwórz środowisko (bez renderowania dla testów)
            env = gym.make("CarRacing-v3", render_mode=None)
            env = LapCompletionFixWrapper(env)
            
            ppo_rewards = []
            for episode in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                steps = 0
                
                for step in range(1000):
                    action, _ = model.predict(obs, deterministic=True)  # deterministic=True dla testów
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                ppo_rewards.append(episode_reward)
                if episode % 10 == 0:
                    print(f"  Epizod {episode+1}: {episode_reward:.2f} ({steps} kroków)")
            
            results['ppo'] = {
                'rewards': ppo_rewards,
                'mean': np.mean(ppo_rewards),
                'std': np.std(ppo_rewards),
                'completion_rate': sum(1 for r in ppo_rewards if r > 600) / episodes * 100
            }
            env.close()
            print(f"✅ PPO: średnia = {results['ppo']['mean']:.2f}")
            
    except Exception as e:
        print(f"❌ PPO błąd: {e}")
        import traceback
        traceback.print_exc()
        results['ppo'] = None
    
    # 4. Test Random
    print("\n🧪 Testowanie Random...")
    try:
        from environments.car_racing_env import CarRacingEnv
        from agents.random_agent import RandomAgent
        
        env = CarRacingEnv(render_mode=None, continuous=True)
        agent = RandomAgent()
        
        random_rewards = []
        for episode in range(episodes):
            observation, info = env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = agent.act()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            random_rewards.append(episode_reward)
            if episode % 10 == 0:
                print(f"  Epizod {episode+1}: {episode_reward:.2f}")
        
        results['random'] = {
            'rewards': random_rewards,
            'mean': np.mean(random_rewards),
            'std': np.std(random_rewards),
            'completion_rate': sum(1 for r in random_rewards if r > 600) / episodes * 100
        }
        env.close()
        print(f"✅ Random: średnia = {results['random']['mean']:.2f}")
        
    except Exception as e:
        print(f"❌ Random błąd: {e}")
        results['random'] = None
    
    return results

# Dodaj funkcję specjalnie dla PPO (zgodną z evaluate_ppo.py)
def test_ppo_stable_baselines(episodes=50, render_mode=None):
    """Test PPO używając Stable Baselines3 - dokładnie jak w evaluate_ppo.py"""
    print("🧪 Testowanie PPO z Stable Baselines3...")
    
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
        
        # Sprawdź czy model istnieje
        model_path = "models/ppo_carracing1"
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Model nie znaleziony: {model_path}.zip")
            print("Dostępne modele:")
            if os.path.exists("models/"):
                for file in os.listdir("models/"):
                    if file.endswith(('.zip', '.pkl')):
                        print(f"  - {file}")
            return None
        
        # Wczytaj model
        model = PPO.load(model_path)
        print(f"✅ Model wczytany: {model_path}")
        
        # Stwórz środowisko
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = LapCompletionFixWrapper(env)
        
        total_rewards = []
        
        print(f"🚀 Rozpoczynanie ewaluacji PPO na {episodes} epizodów")
        
        for ep in range(episodes):
            print(f"\n=== EPIZOD {ep + 1}/{episodes} ===")
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)  # deterministic=True dla testów
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                
                # Zabezpieczenie przed nieskończoną pętlą
                if steps > 1000:
                    print("⏰ Timeout - przerwano epizod")
                    break
            
            total_rewards.append(total_reward)
            
            # Ocena wyniku (jak w evaluate_ppo.py)
            if terminated and total_reward > 600:
                result = "🏆 TOR UKOŃCZONY!"
            elif terminated and total_reward > 300:
                result = "🚗 Dobra jazda!"
            elif total_reward > 0:
                result = "✅ Pozytywny wynik"
            else:
                result = "❌ Słaby wynik"
            
            print(f"Epizod {ep + 1}: {steps} kroków, {total_reward:.2f} pkt - {result}")
            
            # Informacje o postępie na torze
            tiles_visited = info.get('tiles_visited', 0)
            total_tiles = info.get('total_tiles', 0)
            if total_tiles > 0:
                progress = (tiles_visited / total_tiles) * 100
                print(f"Postęp na torze: {tiles_visited}/{total_tiles} płytek ({progress:.1f}%)")
        
        # Podsumowanie (jak w evaluate_ppo.py)
        print(f"\n{'='*50}")
        print(f"=== PODSUMOWANIE EWALUACJI PPO ===")
        print(f"{'='*50}")
        print(f"Średnia nagroda: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Najlepszy wynik: {max(total_rewards):.2f}")
        print(f"Najgorszy wynik: {min(total_rewards):.2f}")
        
        # Analiza sukcesu
        successful_runs = sum(1 for r in total_rewards if r > 600)
        good_runs = sum(1 for r in total_rewards if r > 300)
        positive_runs = sum(1 for r in total_rewards if r > 0)
        
        print(f"\nAnaliza wyników:")
        print(f"Ukończone tory (>600 pkt): {successful_runs}/{episodes} ({successful_runs/episodes*100:.1f}%)")
        print(f"Dobre wyniki (>300 pkt): {good_runs}/{episodes} ({good_runs/episodes*100:.1f}%)")
        print(f"Pozytywne wyniki (>0 pkt): {positive_runs}/{episodes} ({positive_runs/episodes*100:.1f}%)")
        
        env.close()
        print("\n✅ Ewaluacja PPO zakończona!")
        
        return {
            'rewards': total_rewards,
            'mean': np.mean(total_rewards),
            'std': np.std(total_rewards),
            'completion_rate': successful_runs/episodes*100
        }
        
    except Exception as e:
        print(f"❌ Błąd wczytywania modelu PPO: {e}")
        import traceback
        traceback.print_exc()
        return None

# Reszta funkcji pozostaje bez zmian...
def create_comparison_plots(results):
    """Stwórz wykresy porównawcze"""
    # Filtruj tylko dostępne wyniki
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ Brak wyników do wizualizacji!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Porównanie Agentów RL w CarRacing-v3', fontsize=16)
    
    agents = list(valid_results.keys())
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    # 1. Średnie nagrody
    means = [valid_results[agent]['mean'] for agent in agents]
    stds = [valid_results[agent]['std'] for agent in agents]
    
    bars = axes[0,0].bar(agents, means, yerr=stds, capsize=5, color=colors[:len(agents)])
    axes[0,0].set_title('Średnie nagrody ± odchylenie standardowe')
    axes[0,0].set_ylabel('Nagroda')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Próg ukończenia')
    axes[0,0].legend()
    
    # Wartości na słupkach
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 10,
                      f'{mean:.1f}', ha='center', va='bottom')
    
    # 2. Wskaźnik ukończenia
    completion_rates = [valid_results[agent]['completion_rate'] for agent in agents]
    bars2 = axes[0,1].bar(agents, completion_rates, color=colors[:len(agents)])
    axes[0,1].set_title('Wskaźnik ukończenia toru (%)')
    axes[0,1].set_ylabel('Procent ukończonych torów')
    axes[0,1].set_ylim(0, 100)
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, rate in zip(bars2, completion_rates):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{rate:.1f}%', ha='center', va='bottom')
    
    # 3. Box plot
    rewards_data = [valid_results[agent]['rewards'] for agent in agents]
    bp = axes[1,0].boxplot(rewards_data, labels=agents, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors[:len(agents)]):
        patch.set_facecolor(color)
    
    axes[1,0].set_title('Rozkład nagród')
    axes[1,0].set_ylabel('Nagroda')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=600, color='red', linestyle='--', alpha=0.7)
    
    # 4. Histogramy wszystkich wyników
    for i, agent in enumerate(agents):
        axes[1,1].hist(valid_results[agent]['rewards'], bins=15, alpha=0.7, 
                      label=agent, color=colors[i])
    
    axes[1,1].set_title('Histogramy wszystkich wyników')
    axes[1,1].set_xlabel('Nagroda')
    axes[1,1].set_ylabel('Częstość')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvline(x=600, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Zapisz
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/direct_test_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Wykres zapisany: {filename}")
    plt.show()

def print_summary(results):
    """Wydrukuj podsumowanie"""
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    print("\n" + "="*60)
    print("📊 PODSUMOWANIE WYNIKÓW")
    print("="*60)
    
    for agent, data in valid_results.items():
        print(f"\n🤖 {agent.upper()}:")
        print(f"   Średnia nagroda: {data['mean']:.2f} ± {data['std']:.2f}")
        print(f"   Najlepszy wynik: {max(data['rewards']):.2f}")
        print(f"   Najgorszy wynik: {min(data['rewards']):.2f}")
        print(f"   Ukończone tory: {data['completion_rate']:.1f}%")
        
        # Ocena jakościowa
        if data['completion_rate'] > 80:
            rating = "🏆 DOSKONAŁY"
        elif data['completion_rate'] > 50:
            rating = "🥇 BARDZO DOBRY"
        elif data['completion_rate'] > 20:
            rating = "🥈 DOBRY"
        elif data['mean'] > 300:
            rating = "🥉 PRZYZWOITY"
        else:
            rating = "❌ SŁABY"
        
        print(f"   Ocena: {rating}")
    
    # Ranking
    if valid_results:
        print(f"\n🏆 RANKING (wg średniej nagrody):")
        sorted_agents = sorted(valid_results.items(), 
                             key=lambda x: x[1]['mean'], 
                             reverse=True)
        
        for i, (agent, data) in enumerate(sorted_agents, 1):
            print(f"   {i}. {agent.upper()} - {data['mean']:.2f} pkt ({data['completion_rate']:.1f}% ukończeń)")

def save_results(results):
    """Zapisz wyniki"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/direct_test_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Wyniki zapisane: {filename}")

def main():
    """Główna funkcja"""
    print("🚀 ROZPOCZYNANIE BEZPOŚREDNICH TESTÓW")
    
    # Utwórz katalog wyników
    os.makedirs('results', exist_ok=True)
    
    # Testuj agentów
    results = test_all_agents_directly(episodes=5)  # 30 epizodów każdy
    
    # Analizuj wyniki
    print_summary(results)
    create_comparison_plots(results)
    save_results(results)
    
    print("\n✅ Analiza zakończona!")

if __name__ == "__main__":
    main()