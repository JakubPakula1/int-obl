import argparse
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
    env = CarRacingEnv(render_mode=render_mode)
    
    # Wybór agenta i trybu
    if args.mode == 'train':
        if args.agent == 'neat':
            train_neat_agent(env, args)
        elif args.agent == 'dqn':
            train_dqn_agent(env, args)
        elif args.agent == 'ppo':
            # Implementacja treningu PPO
            print("Trenowanie PPO nie jest jeszcze zaimplementowane")
        elif args.agent == 'random':
            print("Agent losowy nie wymaga treningu")
    elif args.mode == 'test':
        test_agent(env, args)
    
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
    # Ustawienie środowiska na tryb wizualizacji
    env_visual = CarRacingEnv(render_mode="human")
    
    if args.agent == 'neat':
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
        if args.model:
            # Wczytaj konkretny model
            agent = DQNAgent.load(args.model, (96, 96, 3), 5)
        else:
            # Wczytaj najnowszy model
            models = glob.glob('checkpoints/dqn/*.h5')
            if models:
                latest_model = sorted(models)[-1]
                print(f"Wczytywanie modelu: {latest_model}")
                agent = DQNAgent.load(latest_model, (96, 96, 3), 5)
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

if __name__ == "__main__":
    main()