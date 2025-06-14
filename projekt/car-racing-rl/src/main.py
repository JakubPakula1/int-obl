import argparse
import numpy as np
from environments.car_racing_env import CarRacingEnv
from agents.neat_agent import NEATAgent
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from training.train_neat import train_neat, continue_from_checkpoint as continue_neat
from training.train_dqn import train_dqn, continue_from_checkpoint as continue_dqn
from evaluation.evaluate import evaluate
import os
import glob

def main():
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description='Car Racing RL')
    parser.add_argument('--agent', type=str, default='neat', choices=['neat', 'dqn', 'random', 'ppo'],
                        help='Rodzaj agenta do trenowania/testowania')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Tryb pracy: trenowanie lub testowanie')
    parser.add_argument('--model', type=str, default=None,
                        help='Ścieżka do modelu do wczytania (dla trybu test)')
    parser.add_argument('--continue_training', action='store_true',
                        help='Kontynuuj trening z ostatniego checkpointu')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Liczba epizodów treningu/testowania')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maksymalna liczba kroków w epizodzie')
    parser.add_argument('--render', action='store_true',
                        help='Wyświetlanie środowiska w trakcie treningu')
    
    args = parser.parse_args()
    
    # Tworzenie środowiska
    render_mode = "human" if args.render else None
    
    
    # Wybór agenta i trybu
    if args.mode == 'train':
        if args.agent == 'neat':
            env = CarRacingEnv(render_mode=render_mode)
            train_neat_agent(env, args)
        elif args.agent == 'dqn':
            env = CarRacingEnv(render_mode=render_mode, continuous=False)
            train_dqn_agent(env, args)
        elif args.agent == 'ppo':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)  # PPO używa ciągłych akcji
            train_ppo_agent(env, args)
        elif args.agent == 'random':
            print("Agent losowy nie wymaga treningu")
    elif args.mode == 'test':
        if args.agent == 'neat':
            env = CarRacingEnv(render_mode=render_mode)
            test_agent(env, args)
        elif args.agent == 'dqn':
            env = CarRacingEnv(render_mode=render_mode, continuous=False)
            test_dqn_agent(env, args)
        elif args.agent == 'ppo':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            test_ppo_agent(env, args)
        elif args.agent == 'random':
            print("Agent losowy nie wymaga treningu")
        
    
    env.close()

def train_neat_agent(env, args):
    if args.continue_training:
        checkpoints = glob.glob('checkpoints/neat-checkpoint-*')
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            print(f"Kontynuowanie z checkpointu: {latest_checkpoint}")
            neat_agent, winner = continue_neat(latest_checkpoint, env, args.episodes)
        else:
            print("Nie znaleziono checkpointów, rozpoczynanie nowego treningu...")
            neat_agent, winner = train_neat(env, generations=args.episodes)
    else:
        print("Rozpoczynanie nowego treningu NEAT...")
        neat_agent, winner = train_neat(env, generations=args.episodes)

def train_dqn_agent(env, args):
    from models.neural_networks import DQNNetwork
    
    if args.continue_training:
        checkpoints = glob.glob('checkpoints/dqn/dqn_model_ep*.keras')
        if checkpoints:
            # Sortowanie numeryczne zamiast alfabetycznego
            def extract_episode_number(filename):
                import re
                match = re.search(r'ep(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            latest_checkpoint = max(checkpoints, key=extract_episode_number)
            print(f"Kontynuowanie z checkpointu: {latest_checkpoint}")
            agent = DQNAgent.load(latest_checkpoint, (84,84,1), 5)
        else:
            print("Nie znaleziono checkpointów, rozpoczynanie nowego treningu...")
            model = DQNNetwork(input_shape=(84, 84, 1), action_space=5).model
            agent = DQNAgent((84, 84, 1), 5, model)
    else:
        print("Rozpoczynanie nowego treningu DQN...")
        model = DQNNetwork(input_shape=(84, 84, 1), action_space=5).model
        agent = DQNAgent((84, 84, 1), 5, model)
    
    # Trenowanie agenta DQN
    train_dqn(env, agent, episodes=args.episodes)

def test_agent(env, args):
    
    if args.agent == 'neat':
        env_visual = CarRacingEnv(render_mode="human")
        print(f"Action space: {env_visual.action_space}")
        print(f"Action space type: {type(env_visual.action_space)}")
        if args.model:
            # Wczytaj konkretny model
            import pickle
            with open(args.model, 'rb') as f:
                agent = pickle.load(f)
            neat_agent = NEATAgent(agent)
        else:
            # Wczytaj najnowszy model
            models = glob.glob('models/neat*.pkl')
            if models:
                latest_model = sorted(models)[-1]
                print(f"Wczytywanie modelu: {latest_model}")
                import pickle
                with open(latest_model, 'rb') as f:
                    agent = pickle.load(f)
                neat_agent = NEATAgent(agent)
            else:
                print("Nie znaleziono modelu NEAT")
                return
        agent = neat_agent
            
    elif args.agent == 'dqn':
        env_visual = CarRacingEnv(render_mode="human", continuous=False)
        print(f"Action space: {env_visual.action_space}")
        print(f"Action space type: {type(env_visual.action_space)}")
        if args.model:
            # Wczytaj konkretny model
            agent = DQNAgent.load(args.model, (84, 84, 1), 5)
        else:
            # Wczytaj najnowszy model
            models = glob.glob('checkpoints/dqn/*.keras')
            if models:
                latest_model = sorted(models)[-1]
                print(f"Wczytywanie modelu: {latest_model}")
                agent = DQNAgent.load(latest_model, (84, 84, 1), 5)
            else:
                print("Nie znaleziono modelu DQN")
                return
    
                
    elif args.agent == 'random':
        agent = RandomAgent(5)  # 5 akcji
    
    else:
        print(f"Testowanie agenta {args.agent} nie jest zaimplementowane")
        return
    
    # Testowanie agenta
    print(f"Testowanie agenta {args.agent}...")
    observation = env_visual.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(args.episodes * 1000):  # Maksymalna liczba kroków
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env_visual.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Epizod zakończony po {steps} krokach z nagrodą: {total_reward:.2f}")
            observation = env_visual.reset()
            total_reward = 0
            steps = 0
            
    env_visual.close()


def test_dqn_agent(env, args):
    """Dedykowana funkcja testowania DQN z preprocessingiem"""
    from training.train_dqn import preprocess_state
    import time
    from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
    env = LapCompletionFixWrapper(env)
    # Wczytaj model
    if args.model:
        agent = DQNAgent.load(args.model, (84, 84, 1), 5)
    else:
        models = glob.glob('checkpoints/dqn/*.keras')
        if models:
            def extract_episode_number(filename):
                import re
                match = re.search(r'ep(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            latest_model = max(models, key=extract_episode_number)
            print(f"Wczytywanie modelu: {latest_model}")
            agent = DQNAgent.load(latest_model, (84, 84, 1), 5)
        else:
            print("Nie znaleziono modelu DQN")
            return
    
    # Epsilon = 0 dla czystego testowania (tylko eksploatacja)
    agent.epsilon = 0.0
    print(f"Model: epsilon={agent.epsilon}, epizody treningowe={agent.episodes}")
    
    # Testowanie
    total_rewards = []
    total_steps = []
    completed_laps = 0
    
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation,info = env.reset()
        observation = preprocess_state(observation)
        
        episode_reward = 0
        steps = 0
        start_time = time.time()
        
        for step in range(1000):
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = preprocess_state(next_observation)
            
            observation = next_observation
            episode_reward += reward
            steps += 1
            
            # Wyświetl progress co 100 kroków
            if steps % 100 == 0:
                print(f"  Krok {steps}, Nagroda: {episode_reward:.2f}")
                print(info)
            
            if terminated or truncated:
                break
        
        episode_time = time.time() - start_time
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        # Ocena rezultatu epizodu
        if terminated and episode_reward > 600:
            completed_laps += 1
            result_text = "🏆 TOR UKOŃCZONY!"
        elif terminated and episode_reward > 300:
            result_text = "🚗 Dobra jazda - prawie ukończył!"
        elif terminated and episode_reward > 0:
            result_text = "✅ Pozytywny wynik"
        elif terminated:
            result_text = "❌ Wypadł z toru"
        else:
            result_text = "⏱️ Timeout - przekroczono limit czasu"
        
        print(f"Epizod {episode+1} ukończony:")
        print(f"  Kroki: {steps}/1000")
        print(f"  Nagroda: {episode_reward:.2f}")
        print(f"  Czas: {episode_time:.2f}s")
        print(f"  Wynik: {result_text}")
        
        # Dodatkowe informacje o postępie
        if episode_reward > 800:
            progress = "95-100%"
        elif episode_reward > 600:
            progress = "80-95%"
        elif episode_reward > 400:
            progress = "60-80%"
        elif episode_reward > 200:
            progress = "30-60%"
        elif episode_reward > 0:
            progress = "0-30%"
        else:
            progress = "Problemy z jazdą"
        
        print(f"  Szacowany postęp na torze: ~{progress}")
    
    # Podsumowanie końcowe
    print(f"\n{'='*50}")
    print(f"=== PODSUMOWANIE TESTÓW DQN ===")
    print(f"{'='*50}")
    print(f"Średnia nagroda: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Najlepszy wynik: {max(total_rewards):.2f}")
    print(f"Najgorszy wynik: {min(total_rewards):.2f}")
    print(f"Średnia liczba kroków: {np.mean(total_steps):.1f}")
    print(f"Najdłuższy epizod: {max(total_steps)} kroków")
    print(f"Najkrótszy epizod: {min(total_steps)} kroków")
    
    # Analiza sukcesu
    print(f"\nAnaliza wyników:")
    successful_runs = sum(1 for r in total_rewards if r > 600)
    good_runs = sum(1 for r in total_rewards if r > 300)
    positive_runs = sum(1 for r in total_rewards if r > 0)
    
    print(f"Ukończone tory (>600 pkt): {successful_runs}/{args.episodes} ({successful_runs/args.episodes*100:.1f}%)")
    print(f"Dobre wyniki (>300 pkt): {good_runs}/{args.episodes} ({good_runs/args.episodes*100:.1f}%)")
    print(f"Pozytywne wyniki (>0 pkt): {positive_runs}/{args.episodes} ({positive_runs/args.episodes*100:.1f}%)")
    
    # Ocena ogólna modelu
    print(f"\nOcena modelu:")
    avg_reward = np.mean(total_rewards)
    if successful_runs >= args.episodes * 0.8:
        print("🏆 DOSKONAŁY MODEL! Konsekwentnie ukańcza tory.")
    elif successful_runs >= args.episodes * 0.6:
        print("🌟 BARDZO DOBRY MODEL! Często ukańcza tory.")
    elif successful_runs >= args.episodes * 0.4:
        print("👍 DOBRY MODEL! Czasami ukańcza tory.")
    elif good_runs >= args.episodes * 0.6:
        print("⚡ PRZYZWOITY MODEL! Dobrze jeździ, ale rzadko ukańcza.")
    elif positive_runs >= args.episodes * 0.6:
        print("🔄 SŁABY MODEL! Wymaga więcej treningu.")
    else:
        print("❌ BARDZO SŁABY MODEL! Potrzebuje znacznie więcej treningu.")
    
    # Rekomendacje
    print(f"\nRekomendacje:")
    if successful_runs < args.episodes * 0.5:
        print("• Kontynuuj trening modelu")
        print("• Rozważ dostrojenie hiperparametrów")
        print("• Sprawdź funkcję nagrody")
    else:
        print("• Model jest dobrze wytrenowany!")
        print("• Możesz użyć go do dalszych eksperymentów")

def train_ppo_agent(env, args):
    """Trenowanie agenta PPO"""
    from agents.ppo_agent import PPOAgent
    from training.train_ppo import train_ppo
    from training.train_ppo_v2 import train_ppo_enhanced
    
    if args.continue_training:
        # Implementacja kontynuacji treningu PPO
        import glob
        actor_models = glob.glob('checkpoints/ppo/*_actor.keras')
        if actor_models:
            def extract_episode_number(filename):
                import re
                match = re.search(r'ep(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            latest_actor = max(actor_models, key=extract_episode_number)
            latest_critic = latest_actor.replace('_actor.keras', '_critic.keras')
            
            print(f"Kontynuowanie z modeli: {latest_actor}, {latest_critic}")
            agent = PPOAgent.load(latest_actor, latest_critic, (84, 84, 1), 3)
        else:
            print("Nie znaleziono modeli PPO, rozpoczynanie nowego treningu...")
            agent = PPOAgent((84, 84, 1), 3)
    else:
        print("Rozpoczynanie nowego treningu PPO...")
        agent = PPOAgent((84, 84, 1), 3)
    
    # train_ppo(env, agent, episodes=args.episodes)
    train_ppo_enhanced(env, agent, episodes=args.episodes)

def test_ppo_agent(env, args):
    """Testowanie agenta PPO"""
    from agents.ppo_agent import PPOAgent
    from training.train_ppo import preprocess_state
    import time
    
    # Wczytaj model
    if args.model:
        actor_path = args.model
        critic_path = args.model.replace('_actor.keras', '_critic.keras')
        agent = PPOAgent.load(actor_path, critic_path, (84, 84, 1), 3)
    else:
        import glob
        actor_models = glob.glob('checkpoints/ppo/*_actor.keras')
        if actor_models:
            def extract_episode_number(filename):
                import re
                match = re.search(r'ep(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            latest_actor = max(actor_models, key=extract_episode_number)
            latest_critic = latest_actor.replace('_actor.keras', '_critic.keras')
            
            print(f"Wczytywanie modeli: {latest_actor}, {latest_critic}")
            agent = PPOAgent.load(latest_actor, latest_critic, (84, 84, 1), 3)
        else:
            print("Nie znaleziono modeli PPO")
            return
    
    print(f"Model PPO: epizody treningowe={agent.episodes}")
    
    # Testowanie
    total_rewards = []
    completed_laps = 0
    
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation, info = env.reset()
        observation = preprocess_state(observation)
        
        episode_reward = 0
        steps = 0
        start_time = time.time()
        
        for step in range(1000):
            action = agent.act(observation)  # Używa deterministycznej polityki w testach
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = preprocess_state(next_observation)
            
            observation = next_observation
            episode_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"  Krok {steps}, Nagroda: {episode_reward:.2f}")
            
            if terminated or truncated:
                break
        
        episode_time = time.time() - start_time
        total_rewards.append(episode_reward)
        
        # Ocena rezultatu
        if terminated and episode_reward > 600:
            completed_laps += 1
            result_text = "🏆 TOR UKOŃCZONY!"
        elif terminated and episode_reward > 300:
            result_text = "🚗 Dobra jazda - prawie ukończył!"
        else:
            result_text = "❌ Nie ukończył toru"
        
        print(f"Epizod {episode+1}: {steps} kroków, {episode_reward:.2f} pkt, {result_text}")
    
    # Podsumowanie
    print(f"\n=== PODSUMOWANIE PPO ===")
    print(f"Średnia nagroda: {np.mean(total_rewards):.2f}")
    print(f"Ukończone tory: {completed_laps}/{args.episodes}")
    print(f"Wskaźnik sukcesu: {completed_laps/args.episodes*100:.1f}%")

  # Dla innych agentów
if __name__ == "__main__":
    main()