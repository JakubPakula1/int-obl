import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse

class SingleModelAnalyzer:
    def __init__(self, model_type, model_path=None):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.results = None
        self.episode_data = []
        
    def test_model(self, episodes=100, save_details=True):
        """Test wybranego modelu z detalami każdego epizodu"""
        print(f"🧪 Szczegółowe testowanie modelu: {self.model_type.upper()}")
        if self.model_path:
            print(f"📁 Ścieżka modelu: {self.model_path}")
        print("=" * 60)
        
        if self.model_type == 'dqn':
            return self._test_dqn(episodes, save_details)
        elif self.model_type == 'neat':
            return self._test_neat(episodes, save_details)
        elif self.model_type == 'ppo':
            return self._test_ppo(episodes, save_details)
        elif self.model_type == 'random':
            return self._test_random(episodes, save_details)
        else:
            print(f"❌ Nieznany typ modelu: {self.model_type}")
            return None

    def _test_dqn(self, episodes, save_details):
        """Test szczegółowy DQN"""
        try:
            from environments.car_racing_env import CarRacingEnv
            from agents.dqn_agent import DQNAgent
            from training.train_dqn import preprocess_state
            from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
            
            env = CarRacingEnv(render_mode=None, continuous=False)
            env = LapCompletionFixWrapper(env)
            
            # Wybierz model
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
                print(f"🔍 Testowanie podanego modelu: {model_path}")
            else:
                # Automatyczne znalezienie najnowszego modelu
                import glob
                models = glob.glob('models/dqn_model_ep*.keras')
                if not models:
                    models = glob.glob('checkpoints/dqn/*.keras')
                if not models:
                    print("❌ Nie znaleziono modeli DQN")
                    return None
                
                def extract_episode_number(filename):
                    import re
                    match = re.search(r'ep(\d+)', filename)
                    return int(match.group(1)) if match else 0
                
                model_path = max(models, key=extract_episode_number)
                print(f"🔍 Testowanie najnowszego modelu: {model_path}")
            
            agent = DQNAgent.load(model_path, (84, 84, 1), 5)
            agent.epsilon = 0.0  # Tylko eksploatacja
            
            episode_data = []
            
            for episode in range(episodes):
                observation, info = env.reset()
                state = preprocess_state(observation)
                episode_reward = 0
                steps = 0
                actions_taken = []
                rewards_per_step = []
                tiles_visited = 0
                
                for step in range(1000):
                    action = agent.act(state)
                    observation, reward, terminated, truncated, info = env.step(action)
                    state = preprocess_state(observation)
                    
                    episode_reward += reward
                    steps += 1
                    actions_taken.append(action)
                    rewards_per_step.append(reward)
                    tiles_visited = info.get('tiles_visited', tiles_visited)
                    
                    if terminated or truncated:
                        break
                
                # Dane epizodu
                episode_info = {
                    'episode': episode + 1,
                    'total_reward': episode_reward,
                    'steps': steps,
                    'completed': terminated and episode_reward > 600,
                    'tiles_visited': tiles_visited,
                    'actions': actions_taken if save_details else [],
                    'step_rewards': rewards_per_step if save_details else [],
                    'avg_reward_per_step': episode_reward / steps if steps > 0 else 0,
                    'action_distribution': self._analyze_dqn_actions(actions_taken)
                }
                
                episode_data.append(episode_info)
                
                if episode % 10 == 0:
                    print(f"  Epizod {episode+1}: {episode_reward:.2f} pkt, {steps} kroków, {'✅' if episode_info['completed'] else '❌'}")
            
            env.close()
            
            self.episode_data = episode_data
            self.results = {
                'model_type': 'DQN',
                'model_path': model_path,
                'total_episodes': episodes,
                'rewards': [ep['total_reward'] for ep in episode_data],
                'completion_rate': sum(1 for ep in episode_data if ep['completed']) / episodes * 100,
                'avg_steps': np.mean([ep['steps'] for ep in episode_data]),
                'avg_tiles': np.mean([ep['tiles_visited'] for ep in episode_data])
            }
            
            print(f"✅ DQN test zakończony - średnia: {np.mean(self.results['rewards']):.2f}")
            return self.results
            
        except Exception as e:
            print(f"❌ Błąd testowania DQN: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _test_ppo(self, episodes, save_details):
        """Test szczegółowy PPO"""
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO
            from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
            
            # Wybierz model
            if self.model_path and os.path.exists(self.model_path + ".zip"):
                model_path = self.model_path
                print(f"🔍 Testowanie podanego modelu: {model_path}")
            elif self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path.replace('.zip', '')
                print(f"🔍 Testowanie podanego modelu: {model_path}")
            else:
                # Automatyczne znalezienie modelu
                possible_paths = [
                    "models/ppo_carracing1",
                    "models/ppo_carracing",
                    "models/ppo_model"
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path + ".zip"):
                        model_path = path
                        break
                
                if not model_path:
                    print("❌ Model PPO nie znaleziony")
                    return None
                print(f"🔍 Testowanie domyślnego modelu: {model_path}")
            
            model = PPO.load(model_path)
            
            env = gym.make("CarRacing-v3", render_mode=None)
            env = LapCompletionFixWrapper(env)
            
            episode_data = []
            
            for episode in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                steps = 0
                actions_taken = []
                rewards_per_step = []
                tiles_visited = 0
                
                for step in range(1000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    actions_taken.append(action.copy())
                    rewards_per_step.append(reward)
                    tiles_visited = info.get('tiles_visited', tiles_visited)
                    
                    if terminated or truncated:
                        break
                
                episode_info = {
                    'episode': episode + 1,
                    'total_reward': episode_reward,
                    'steps': steps,
                    'completed': terminated and episode_reward > 600,
                    'tiles_visited': tiles_visited,
                    'actions': actions_taken if save_details else [],
                    'step_rewards': rewards_per_step if save_details else [],
                    'avg_reward_per_step': episode_reward / steps if steps > 0 else 0,
                    'action_stats': self._analyze_continuous_actions(actions_taken)
                }
                
                episode_data.append(episode_info)
                
                if episode % 10 == 0:
                    print(f"  Epizod {episode+1}: {episode_reward:.2f} pkt, {steps} kroków, {'✅' if episode_info['completed'] else '❌'}")
            
            env.close()
            
            self.episode_data = episode_data
            self.results = {
                'model_type': 'PPO',
                'model_path': model_path,
                'total_episodes': episodes,
                'rewards': [ep['total_reward'] for ep in episode_data],
                'completion_rate': sum(1 for ep in episode_data if ep['completed']) / episodes * 100,
                'avg_steps': np.mean([ep['steps'] for ep in episode_data]),
                'avg_tiles': np.mean([ep['tiles_visited'] for ep in episode_data])
            }
            
            print(f"✅ PPO test zakończony - średnia: {np.mean(self.results['rewards']):.2f}")
            return self.results
            
        except Exception as e:
            print(f"❌ Błąd testowania PPO: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _test_neat(self, episodes, save_details):
        """Test szczegółowy NEAT"""
        try:
            from environments.car_racing_env import CarRacingEnv
            from agents.neat_agent import NEATAgent
            from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
            
            env = CarRacingEnv(render_mode=None, continuous=True)
            env = LapCompletionFixWrapper(env)
            
            # Wybierz model
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
                print(f"🔍 Testowanie podanego modelu: {model_path}")
            else:
                # Automatyczne znalezienie modelu
                possible_paths = [
                    'models/neat_best.pkl',
                    'models/neat_winner.pkl',
                    'checkpoints/neat_best.pkl'
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if not model_path:
                    print("❌ Model NEAT nie znaleziony")
                    return None
                print(f"🔍 Testowanie domyślnego modelu: {model_path}")
            
            # Znajdź plik konfiguracji
            config_path = 'configs/neat_config.txt'
            if not os.path.exists(config_path):
                print(f"❌ Plik konfiguracji nie znaleziony: {config_path}")
                return None
            
            agent = NEATAgent.load_model(model_path, config_path)
            if agent is None:
                print("❌ Nie można wczytać modelu NEAT")
                return None
            
            print(f"🔍 Testowanie modelu NEAT - fitness: {agent.best_genome.fitness:.2f}")
            
            episode_data = []
            
            for episode in range(episodes):
                observation, info = env.reset()
                episode_reward = 0
                steps = 0
                actions_taken = []
                rewards_per_step = []
                tiles_visited = 0
                
                for step in range(1000):
                    action = agent.act(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    actions_taken.append(action.copy())
                    rewards_per_step.append(reward)
                    tiles_visited = info.get('tiles_visited', tiles_visited)
                    
                    if terminated or truncated:
                        break
                
                episode_info = {
                    'episode': episode + 1,
                    'total_reward': episode_reward,
                    'steps': steps,
                    'completed': terminated and episode_reward > 600,
                    'tiles_visited': tiles_visited,
                    'actions': actions_taken if save_details else [],
                    'step_rewards': rewards_per_step if save_details else [],
                    'avg_reward_per_step': episode_reward / steps if steps > 0 else 0,
                    'action_stats': self._analyze_continuous_actions(actions_taken)
                }
                
                episode_data.append(episode_info)
                
                if episode % 10 == 0:
                    print(f"  Epizod {episode+1}: {episode_reward:.2f} pkt, {steps} kroków, {'✅' if episode_info['completed'] else '❌'}")
            
            env.close()
            
            self.episode_data = episode_data
            self.results = {
                'model_type': 'NEAT',
                'model_path': model_path,
                'total_episodes': episodes,
                'rewards': [ep['total_reward'] for ep in episode_data],
                'completion_rate': sum(1 for ep in episode_data if ep['completed']) / episodes * 100,
                'avg_steps': np.mean([ep['steps'] for ep in episode_data]),
                'avg_tiles': np.mean([ep['tiles_visited'] for ep in episode_data])
            }
            
            print(f"✅ NEAT test zakończony - średnia: {np.mean(self.results['rewards']):.2f}")
            return self.results
            
        except Exception as e:
            print(f"❌ Błąd testowania NEAT: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _test_random(self, episodes, save_details):
        """Test szczegółowy Random"""
        try:
            from environments.car_racing_env import CarRacingEnv
            from agents.random_agent import RandomAgent
            from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
            
            env = CarRacingEnv(render_mode=None, continuous=True)
            env = LapCompletionFixWrapper(env)
            agent = RandomAgent()
            
            print(f"🔍 Testowanie agenta losowego (baseline)")
            
            episode_data = []
            
            for episode in range(episodes):
                observation, info = env.reset()
                episode_reward = 0
                steps = 0
                actions_taken = []
                rewards_per_step = []
                tiles_visited = 0
                
                for step in range(1000):
                    action = agent.act()
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    actions_taken.append(action.copy())
                    rewards_per_step.append(reward)
                    tiles_visited = info.get('tiles_visited', tiles_visited)
                    
                    if terminated or truncated:
                        break
                
                episode_info = {
                    'episode': episode + 1,
                    'total_reward': episode_reward,
                    'steps': steps,
                    'completed': terminated and episode_reward > 600,
                    'tiles_visited': tiles_visited,
                    'actions': actions_taken if save_details else [],
                    'step_rewards': rewards_per_step if save_details else [],
                    'avg_reward_per_step': episode_reward / steps if steps > 0 else 0,
                    'action_stats': self._analyze_continuous_actions(actions_taken)
                }
                
                episode_data.append(episode_info)
                
                if episode % 10 == 0:
                    print(f"  Epizod {episode+1}: {episode_reward:.2f} pkt, {steps} kroków, {'✅' if episode_info['completed'] else '❌'}")
            
            env.close()
            
            self.episode_data = episode_data
            self.results = {
                'model_type': 'Random',
                'model_path': 'baseline',
                'total_episodes': episodes,
                'rewards': [ep['total_reward'] for ep in episode_data],
                'completion_rate': sum(1 for ep in episode_data if ep['completed']) / episodes * 100,
                'avg_steps': np.mean([ep['steps'] for ep in episode_data]),
                'avg_tiles': np.mean([ep['tiles_visited'] for ep in episode_data])
            }
            
            print(f"✅ Random test zakończony - średnia: {np.mean(self.results['rewards']):.2f}")
            return self.results
            
        except Exception as e:
            print(f"❌ Błąd testowania Random: {e}")
            return None

    def _analyze_dqn_actions(self, actions):
        """Analiza akcji dyskretnych DQN"""
        if not actions:
            return {}
        
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total_actions = len(actions)
        action_percentages = {k: (v/total_actions)*100 for k, v in action_counts.items()}
        
        return {
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'most_used_action': max(action_counts, key=action_counts.get),
            'action_diversity': len(action_counts)
        }

    def _analyze_continuous_actions(self, actions):
        """Analiza akcji ciągłych"""
        if not actions:
            return {}
        
        actions_array = np.array(actions)
        
        return {
            'steering_mean': np.mean(actions_array[:, 0]),
            'steering_std': np.std(actions_array[:, 0]),
            'gas_mean': np.mean(actions_array[:, 1]),
            'gas_std': np.std(actions_array[:, 1]),
            'brake_mean': np.mean(actions_array[:, 2]),
            'brake_std': np.std(actions_array[:, 2]),
            'extreme_steering': np.sum(np.abs(actions_array[:, 0]) > 0.8),
            'high_gas': np.sum(actions_array[:, 1] > 0.8),
            'high_brake': np.sum(actions_array[:, 2] > 0.5)
        }

    def create_comprehensive_plots(self):
        """Twórz szczegółowe wykresy dla modelu"""
        if not self.results or not self.episode_data:
            print("❌ Brak danych do wizualizacji!")
            return
        
        # Ustaw styl wykresów
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Główny figura z subplotami
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Szczegółowa Analiza Modelu {self.results["model_type"]}', fontsize=20, fontweight='bold')
        
        # 1. Nagrody w czasie (górny lewy)
        ax1 = plt.subplot(3, 4, 1)
        episodes = [ep['episode'] for ep in self.episode_data]
        rewards = [ep['total_reward'] for ep in self.episode_data]
        
        ax1.plot(episodes, rewards, 'o-', linewidth=2, markersize=4, alpha=0.7)
        ax1.axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Próg ukończenia')
        
        # Dodaj trend line
        z = np.polyfit(episodes, rewards, 1)
        p = np.poly1d(z)
        ax1.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax1.set_title('Nagrody w kolejnych epizodach')
        ax1.set_xlabel('Epizod')
        ax1.set_ylabel('Nagroda')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram nagród (górny drugi)
        ax2 = plt.subplot(3, 4, 2)
        ax2.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Średnia: {np.mean(rewards):.1f}')
        ax2.axvline(x=600, color='orange', linestyle='--', linewidth=2, label='Próg ukończenia')
        ax2.set_title('Rozkład nagród')
        ax2.set_xlabel('Nagroda')
        ax2.set_ylabel('Częstość')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot dla różnych zakresów nagród (górny trzeci)
        ax3 = plt.subplot(3, 4, 3)
        
        # Podziel nagrody na kategorie
        excellent = [r for r in rewards if r > 800]
        good = [r for r in rewards if 400 <= r <= 800]
        poor = [r for r in rewards if 0 <= r < 400]
        bad = [r for r in rewards if r < 0]
        
        data_to_plot = []
        labels = []
        
        if excellent:
            data_to_plot.append(excellent)
            labels.append(f'Doskonałe\n(>800)\n{len(excellent)} ep.')
        if good:
            data_to_plot.append(good)
            labels.append(f'Dobre\n(400-800)\n{len(good)} ep.')
        if poor:
            data_to_plot.append(poor)
            labels.append(f'Słabe\n(0-400)\n{len(poor)} ep.')
        if bad:
            data_to_plot.append(bad)
            labels.append(f'Bardzo słabe\n(<0)\n{len(bad)} ep.')
        
        if data_to_plot:
            bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['gold', 'lightgreen', 'lightcoral', 'lightgray']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        
        ax3.set_title('Nagrody według kategorii')
        ax3.set_ylabel('Nagroda')
        ax3.grid(True, alpha=0.3)
        
        # 4. Kroki vs Nagrody (górny prawy)
        ax4 = plt.subplot(3, 4, 4)
        steps = [ep['steps'] for ep in self.episode_data]
        colors_scatter = ['green' if ep['completed'] else 'red' for ep in self.episode_data]
        
        scatter = ax4.scatter(steps, rewards, c=colors_scatter, alpha=0.6, s=50)
        ax4.set_title('Kroki vs Nagrody')
        ax4.set_xlabel('Liczba kroków')
        ax4.set_ylabel('Nagroda')
        ax4.grid(True, alpha=0.3)
        
        # Legenda dla scatter
        import matplotlib.patches as mpatches
        completed_patch = mpatches.Patch(color='green', label='Ukończone')
        failed_patch = mpatches.Patch(color='red', label='Nieukończone')
        ax4.legend(handles=[completed_patch, failed_patch])
        
        # 5. Statystyki sukcesu w czasie (środkowy lewy)
        ax5 = plt.subplot(3, 4, 5)
        
        # Oblicz wskaźnik sukcesu w przesuwającym się oknie
        window_size = 10
        success_rates = []
        window_centers = []
        
        for i in range(window_size, len(self.episode_data) + 1):
            window_episodes = self.episode_data[i-window_size:i]
            success_rate = sum(1 for ep in window_episodes if ep['completed']) / window_size * 100
            success_rates.append(success_rate)
            window_centers.append(i - window_size/2)
        
        if success_rates:
            ax5.plot(window_centers, success_rates, 'o-', linewidth=2, markersize=4)
            ax5.set_title(f'Wskaźnik sukcesu (okno {window_size} epizodów)')
            ax5.set_xlabel('Epizod (środek okna)')
            ax5.set_ylabel('Wskaźnik sukcesu (%)')
            ax5.set_ylim(0, 100)
            ax5.grid(True, alpha=0.3)
        
        # 6. Analiza akcji (środkowy drugi)
        ax6 = plt.subplot(3, 4, 6)
        
        if self.model_type == 'dqn' and self.episode_data and 'action_distribution' in self.episode_data[0]:
            # Analiza akcji dyskretnych dla DQN
            all_action_counts = {}
            for ep in self.episode_data:
                action_dist = ep.get('action_distribution', {})
                action_counts = action_dist.get('action_counts', {})
                for action, count in action_counts.items():
                    all_action_counts[action] = all_action_counts.get(action, 0) + count
            
            if all_action_counts:
                actions = list(all_action_counts.keys())
                counts = list(all_action_counts.values())
                
                bars = ax6.bar(actions, counts, alpha=0.7)
                ax6.set_title('Rozkład użycia akcji (DQN)')
                ax6.set_xlabel('Akcja')
                ax6.set_ylabel('Częstość użycia')
                ax6.grid(True, alpha=0.3)
                
                # Dodaj wartości na słupkach
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                            f'{count}', ha='center', va='bottom')
        
        elif self.episode_data and 'action_stats' in self.episode_data[0]:
            # Analiza akcji ciągłych
            steering_means = [ep['action_stats']['steering_mean'] for ep in self.episode_data if 'action_stats' in ep]
            gas_means = [ep['action_stats']['gas_mean'] for ep in self.episode_data if 'action_stats' in ep]
            brake_means = [ep['action_stats']['brake_mean'] for ep in self.episode_data if 'action_stats' in ep]
            
            if steering_means:
                x = np.arange(len(['Steering', 'Gas', 'Brake']))
                means = [np.mean(steering_means), np.mean(gas_means), np.mean(brake_means)]
                stds = [np.std(steering_means), np.std(gas_means), np.std(brake_means)]
                
                bars = ax6.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                              color=['lightblue', 'lightgreen', 'lightcoral'])
                ax6.set_title('Średnie wartości akcji')
                ax6.set_xticks(x)
                ax6.set_xticklabels(['Steering', 'Gas', 'Brake'])
                ax6.set_ylabel('Średnia wartość')
                ax6.grid(True, alpha=0.3)
                
                # Dodaj wartości na słupkach
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + max(means)*0.05,
                            f'{mean:.3f}', ha='center', va='bottom')
        
        # 7. Odwiedzone kafelki vs Nagrody (środkowy trzeci)
        ax7 = plt.subplot(3, 4, 7)
        tiles_visited = [ep['tiles_visited'] for ep in self.episode_data]
        
        scatter = ax7.scatter(tiles_visited, rewards, c=colors_scatter, alpha=0.6, s=50)
        ax7.set_title('Kafelki vs Nagrody')
        ax7.set_xlabel('Odwiedzone kafelki')
        ax7.set_ylabel('Nagroda')
        ax7.grid(True, alpha=0.3)
        
        # 8. Wydajność w czasie (środkowy prawy)
        ax8 = plt.subplot(3, 4, 8)
        rewards_per_step = [ep['avg_reward_per_step'] for ep in self.episode_data]
        
        ax8.plot(episodes, rewards_per_step, 'o-', linewidth=2, markersize=4, alpha=0.7)
        ax8.set_title('Wydajność (nagroda/krok)')
        ax8.set_xlabel('Epizod')
        ax8.set_ylabel('Nagroda na krok')
        ax8.grid(True, alpha=0.3)
        
        # 9. Statystyki tekstowe (środkowy lewy dolny)
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        
        stats_text = f"""
STATYSTYKI MODELU {self.results['model_type']}

📊 Podstawowe:
• Średnia nagroda: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}
• Mediana: {np.median(rewards):.2f}
• Najlepszy wynik: {max(rewards):.2f}
• Najgorszy wynik: {min(rewards):.2f}

🏆 Ukończenia:
• Wskaźnik sukcesu: {self.results['completion_rate']:.1f}%
• Ukończone tory: {sum(1 for ep in self.episode_data if ep['completed'])}/{len(self.episode_data)}

📏 Kroki:
• Średnie kroki: {np.mean(steps):.1f}
• Średnie kafelki: {np.mean(tiles_visited):.1f}

🎯 Wydajność:
• Śr. nagroda/krok: {np.mean(rewards_per_step):.3f}
• Konsystentność: {(1 - np.std(rewards)/np.mean(np.abs(rewards)))*100 if np.mean(np.abs(rewards)) > 0 else 0:.1f}%
        """
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 10. Korelacje (środkowy prawy dolny)
        ax10 = plt.subplot(3, 4, 10)
        
        # Macierz korelacji
        data_for_corr = {
            'Nagroda': rewards,
            'Kroki': steps,
            'Kafelki': tiles_visited,
            'Nagroda/Krok': rewards_per_step
        }
        
        import pandas as pd
        df_corr = pd.DataFrame(data_for_corr)
        corr_matrix = df_corr.corr()
        
        im = ax10.imshow(corr_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
        ax10.set_xticks(range(len(corr_matrix.columns)))
        ax10.set_yticks(range(len(corr_matrix.columns)))
        ax10.set_xticklabels(corr_matrix.columns, rotation=45)
        ax10.set_yticklabels(corr_matrix.columns)
        ax10.set_title('Macierz korelacji')
        
        # Dodaj wartości korelacji na wykres
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax10.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax10, shrink=0.8)
        
        # 11. Trend analizy (dolny lewy)
        ax11 = plt.subplot(3, 4, 11)
        
        # Analiza trendu w różnych okresach
        periods = [(0, len(episodes)//3), (len(episodes)//3, 2*len(episodes)//3), (2*len(episodes)//3, len(episodes))]
        period_names = ['Początek', 'Środek', 'Koniec']
        period_means = []
        period_stds = []
        
        for start, end in periods:
            period_rewards = rewards[start:end]
            if period_rewards:
                period_means.append(np.mean(period_rewards))
                period_stds.append(np.std(period_rewards))
            else:
                period_means.append(0)
                period_stds.append(0)
        
        bars = ax11.bar(period_names, period_means, yerr=period_stds, capsize=5, alpha=0.7)
        ax11.set_title('Wyniki w różnych okresach')
        ax11.set_ylabel('Średnia nagroda')
        ax11.grid(True, alpha=0.3)
        
        # Dodaj wartości na słupkach
        for bar, mean in zip(bars, period_means):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height + max(period_means)*0.05,
                     f'{mean:.1f}', ha='center', va='bottom')
        
        # 12. Porównanie z próg (dolny prawy)
        ax12 = plt.subplot(3, 4, 12)
        
        # Analiza względem progu ukończenia
        above_threshold = [r for r in rewards if r >= 600]
        below_threshold = [r for r in rewards if r < 600]
        
        categories = []
        values = []
        
        if above_threshold:
            categories.append(f'≥600 pkt\n({len(above_threshold)} ep.)')
            values.append(len(above_threshold))
        
        if below_threshold:
            categories.append(f'<600 pkt\n({len(below_threshold)} ep.)')
            values.append(len(below_threshold))
        
        if categories:
            colors_pie = ['lightgreen', 'lightcoral']
            wedges, texts, autotexts = ax12.pie(values, labels=categories, autopct='%1.1f%%',
                                               colors=colors_pie[:len(values)], startangle=90)
            ax12.set_title('Rozkład względem progu')
        
        plt.tight_layout()
        
        # Zapisz wykres
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/{self.model_type}_detailed_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Szczegółowy wykres zapisany: {filename}")
        
        plt.show()

    def save_detailed_results(self):
        """Zapisz szczegółowe wyniki"""
        if not self.results:
            print("❌ Brak wyników do zapisania!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Zapisz podstawowe wyniki
        results_filename = f'results/{self.model_type}_detailed_results_{timestamp}.json'
        with open(results_filename, 'w') as f:
            json.dump({
                'results': self.results,
                'episode_data': self.episode_data[:10]  # Tylko pierwsze 10 epizodów z detalami
            }, f, indent=2)
        
        print(f"💾 Szczegółowe wyniki zapisane: {results_filename}")
        
        # Zapisz podsumowanie CSV
        csv_filename = f'results/{self.model_type}_summary_{timestamp}.csv'
        import pandas as pd
        
        summary_data = {
            'episode': [ep['episode'] for ep in self.episode_data],
            'total_reward': [ep['total_reward'] for ep in self.episode_data],
            'steps': [ep['steps'] for ep in self.episode_data],
            'completed': [ep['completed'] for ep in self.episode_data],
            'tiles_visited': [ep['tiles_visited'] for ep in self.episode_data],
            'avg_reward_per_step': [ep['avg_reward_per_step'] for ep in self.episode_data]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_filename, index=False)
        print(f"💾 Podsumowanie CSV zapisane: {csv_filename}")

def main():
    """Główna funkcja z argumentami wiersza poleceń"""
    parser = argparse.ArgumentParser(description='Szczegółowa analiza pojedynczego modelu RL')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['dqn', 'neat', 'ppo', 'random'],
                       help='Typ modelu do analizy')
    parser.add_argument('--path', type=str, default=None,
                       help='Bezpośrednia ścieżka do modelu (opcjonalne)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Liczba epizodów testowych (domyślnie: 50)')
    parser.add_argument('--save_details', action='store_true',
                       help='Zapisz szczegóły każdego kroku (zwiększa rozmiar plików)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Nie generuj wykresów (tylko testy)')
    
    args = parser.parse_args()
    
    print(f"🚀 SZCZEGÓŁOWA ANALIZA MODELU: {args.model.upper()}")
    if args.path:
        print(f"📁 Ścieżka modelu: {args.path}")
    print("=" * 60)
    
    # Utwórz katalog wyników
    os.makedirs('results', exist_ok=True)
    
    # Stwórz analizator
    analyzer = SingleModelAnalyzer(args.model, args.path)
    
    # Przeprowadź testy
    results = analyzer.test_model(episodes=args.episodes, save_details=args.save_details)
    
    if results:
        print(f"\n📊 PODSUMOWANIE {args.model.upper()}:")
        print(f"   Średnia nagroda: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
        print(f"   Wskaźnik ukończenia: {results['completion_rate']:.1f}%")
        print(f"   Średnie kroki: {results['avg_steps']:.1f}")
        print(f"   Średnie kafelki: {results['avg_tiles']:.1f}")
        
        # Zapisz wyniki
        analyzer.save_detailed_results()
        
        # Generuj wykresy (jeśli nie wyłączone)
        if not args.no_plots:
            analyzer.create_comprehensive_plots()
        
        print(f"\n✅ Analiza {args.model.upper()} zakończona!")
    else:
        print(f"\n❌ Nie udało się przeprowadzić analizy {args.model.upper()}")

if __name__ == "__main__":
    main()
