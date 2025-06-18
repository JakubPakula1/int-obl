import argparse
import numpy as np
from environments.car_racing_env import CarRacingEnv
from agents.neat_agent import NEATAgent
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from training.train_neat import train_neat, continue_from_checkpoint as continue_neat
from training.train_dqn import train_dqn, continue_from_checkpoint as continue_dqn
from evaluation.evaluate import evaluate
from training.train_ppo import train_ppo
import os
import time
import glob

def main():
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description='Car Racing RL')
    parser.add_argument('--agent', type=str, default='neat', 
                       choices=['neat', 'dqn', 'random', 'ppo', 'fuzzy', 'genetic', 'pso'],
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
        elif args.agent == 'genetic':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            train_genetic_agent(env, args)
        elif args.agent == 'pso':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            train_pso_agent(env, args)
        elif args.agent == 'random':
            print("Agent losowy nie wymaga treningu")
    elif args.mode == 'test':
        if args.agent == 'neat':
            env = CarRacingEnv(render_mode=render_mode)
            test_neat_agent(env, args)
        elif args.agent == 'dqn':
            env = CarRacingEnv(render_mode=render_mode, continuous=False)
            test_dqn_agent(env, args)
        elif args.agent == 'ppo':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            test_ppo_agent(env, args)
        elif args.agent == 'fuzzy':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            test_fuzzy_agent(env, args)
        elif args.agent == 'genetic':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            test_genetic_agent(env, args)
        elif args.agent == 'pso':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            test_pso_agent(env, args)
        elif args.agent == 'random':
            env = CarRacingEnv(render_mode=render_mode, continuous=True)
            test_random_agent(env, args)
        
    
    env.close()

def train_neat_agent(env, args):
    if args.continue_training:
        # TYMCZASOWA POPRAWKA: Użyj konkretnego checkpointu
        latest_checkpoint = 'checkpoints/neat-checkpoint-10'  # ← ZMIEŃ NA NAJNOWSZY
        
        if os.path.exists(latest_checkpoint):
            print(f"Kontynuowanie z checkpointu: {latest_checkpoint}")
            neat_agent, winner = continue_neat(latest_checkpoint, env, args.episodes)
        else:
            print("Szukam najnowszego checkpointu...")
            checkpoints = glob.glob('checkpoints/neat-checkpoint-*')
            if checkpoints:
                def extract_number(filename):
                    import re
                    match = re.search(r'checkpoint-(\d+)', filename)
                    return int(match.group(1)) if match else 0
                
                latest_checkpoint = max(checkpoints, key=extract_number)
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

def test_neat_agent(env, args):
    """Testowanie agenta NEAT - POPRAWIONA WERSJA"""
    from agents.neat_agent import NEATAgent
    
    if args.model:
        print(f"Wczytywanie modelu: {args.model}")
        # POPRAWKA: Użyj metody klasowej load_model
        neat_agent = NEATAgent.load_model(args.model, 'configs/neat_config.txt')
        if neat_agent is None:
            print("❌ Nie udało się wczytać modelu!")
            return
    else:
        # Znajdź najnowszy model
        import glob
        models = glob.glob('models/neat_*.pkl')
        if models:
            latest_model = max(models, key=os.path.getctime)
            print(f"Wczytywanie najnowszego modelu: {latest_model}")
            neat_agent = NEATAgent.load_model(latest_model, 'configs/neat_config.txt')
        else:
            print("❌ Nie znaleziono modeli NEAT!")
            return
    
    print(f"🧬 Model NEAT wczytany - fitness: {neat_agent.best_genome.fitness:.2f}")
    
    # Testowanie
    total_rewards = []
    
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation, info = env.reset()  # POPRAWKA: Prawidłowy unpacking
        
        episode_reward = 0
        steps = 0
        start_time = time.time()
        
        for step in range(1000):
            action = neat_agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_time = time.time() - start_time
        
        tiles_visited = info.get('tiles_visited', 0)
        print(f"Epizod {episode+1}: {episode_reward:.2f} pts, {steps} kroków, {tiles_visited} płytek, {episode_time:.1f}s")
    
    # Statystyki
    avg_reward = np.mean(total_rewards)
    print(f"\n🎯 PODSUMOWANIE NEAT:")
    print(f"📊 Średnia nagroda: {avg_reward:.2f}")
    print(f"🏆 Najlepszy wynik: {max(total_rewards):.2f}")
    print(f"📉 Najgorszy wynik: {min(total_rewards):.2f}")


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
    train_ppo(env, agent, episodes=args.episodes)

def test_ppo_agent(env, args):
    from stable_baselines3 import PPO
    import time
    from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
    import sys

    env = LapCompletionFixWrapper(env)
    model_path = "models/ppo_carracing1"

    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model nie znaleziony: {model_path}.zip")
        print("Dostępne modele:")
        if os.path.exists("models/"):
            for file in os.listdir("models/"):
                if file.endswith(('.zip', '.pkl')):
                    print(f"  - {file}")
        sys.exit(1)

    try:
        model = PPO.load(model_path)
        print(f"✅ Model wczytany: {model_path}")
    except Exception as e:
        print(f"❌ Błąd wczytywania modelu: {e}")
        sys.exit(1)

    episodes = 5
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
            
            # Opcjonalne spowolnienie dla lepszej wizualizacji
            time.sleep(1/60)
            
            # Zabezpieczenie przed nieskończoną pętlą
            if steps > 1000:
                print("⏰ Timeout - przerwano epizod")
                break

        total_rewards.append(total_reward)
        
        # Ocena wyniku
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

    # Podsumowanie
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
    print("\n✅ Ewaluacja zakończona!")
def train_genetic_agent(env, args):
    """Trenowanie agenta genetycznego"""
    from agents.ga_agent import GeneticAgent
    
    agent = GeneticAgent(env, chromosome_length=600, population_size=30)
    agent.train(num_generations=args.episodes)

def train_pso_agent(env, args):
    """Trenowanie agenta PSO"""
    from agents.pso_agent import PSOAgent
    
    agent = PSOAgent(env, num_particles=20, dimensions=600)
    agent.train(max_iterations=args.episodes)

def test_fuzzy_agent(env, args):
    """Testowanie agenta rozmytego"""
    from agents.fuzzy_agent import FuzzyAgent
    
    agent = FuzzyAgent()
    
    total_rewards = []
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(1000):
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Epizod {episode+1}: {steps} kroków, {episode_reward:.2f} pkt")
    
    print(f"\n🧠 PODSUMOWANIE FUZZY:")
    print(f"📊 Średnia nagroda: {np.mean(total_rewards):.2f}")

def test_genetic_agent(env, args):
    """Testowanie agenta genetycznego"""
    from agents.ga_agent import GeneticAgent
    
    if args.model:
        agent = GeneticAgent.load_best_genome(args.model, env)
    else:
        agent = GeneticAgent.load_best_genome('models/genetic_best_final.pkl', env)
    
    if agent is None:
        print("❌ Nie można wczytać modelu genetycznego")
        return
    
    total_rewards = []
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(1000):
            action = agent.act(observation, step)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Epizod {episode+1}: {steps} kroków, {episode_reward:.2f} pkt")
    
    print(f"\n🧬 PODSUMOWANIE GENETIC:")
    print(f"📊 Średnia nagroda: {np.mean(total_rewards):.2f}")

def test_pso_agent(env, args):
    """Testowanie agenta PSO"""
    from agents.pso_agent import PSOAgent
    
    if args.model:
        agent = PSOAgent.load_model(args.model, env)
    else:
        agent = PSOAgent.load_model('models/pso_best_final.pkl', env)
    
    if agent is None:
        print("❌ Nie można wczytać modelu PSO")
        return
    
    total_rewards = []
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(1000):
            action = agent.act(observation, step)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Epizod {episode+1}: {steps} kroków, {episode_reward:.2f} pkt")
    
    print(f"\n🐝 PODSUMOWANIE PSO:")
    print(f"📊 Średnia nagroda: {np.mean(total_rewards):.2f}")

def test_random_agent(env, args):
    """Testowanie agenta losowego - baseline"""
    from agents.random_agent import RandomAgent
    import time
    import numpy as np
    
    agent = RandomAgent()
    print(f"🎲 Testowanie agenta losowego - {args.episodes} epizodów")
    
    total_rewards = []
    total_steps = []
    completed_laps = 0
    
    for episode in range(args.episodes):
        print(f"\n=== EPIZOD {episode+1}/{args.episodes} ===")
        observation, info = env.reset()
        
        episode_reward = 0
        steps = 0
        start_time = time.time()
        
        for step in range(1000):
            # Agent losowy nie potrzebuje obserwacji - działania są losowe
            action = agent.act()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Wyświetl progress co 200 kroków
            if steps % 200 == 0:
                print(f"  Krok {steps}, Nagroda: {episode_reward:.2f}")
            
            if terminated or truncated:
                break
        
        episode_time = time.time() - start_time
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        # Ocena rezultatu epizodu
        if terminated and episode_reward > 600:
            completed_laps += 1
            result_text = "🏆 TOR UKOŃCZONY! (niesamowite szczęście!)"
        elif terminated and episode_reward > 300:
            result_text = "🍀 Bardzo szczęśliwy przejazd!"
        elif terminated and episode_reward > 100:
            result_text = "🎲 Przyzwoity losowy wynik"
        elif terminated and episode_reward > 0:
            result_text = "✅ Pozytywny wynik"
        elif terminated:
            result_text = "❌ Wypadł z toru (typowe dla losowego)"
        else:
            result_text = "⏱️ Timeout - przekroczono limit czasu"
        
        print(f"Epizod {episode+1} ukończony:")
        print(f"  Kroki: {steps}/1000")
        print(f"  Nagroda: {episode_reward:.2f}")
        print(f"  Czas: {episode_time:.2f}s")
        print(f"  Wynik: {result_text}")
    
    # Podsumowanie końcowe
    print(f"\n{'='*50}")
    print(f"=== PODSUMOWANIE TESTÓW RANDOM ===")
    print(f"{'='*50}")
    
    try:
        mean_reward = sum(total_rewards) / len(total_rewards)
        std_reward = (sum((r - mean_reward)**2 for r in total_rewards) / len(total_rewards))**0.5
        mean_steps = sum(total_steps) / len(total_steps)
        
        print(f"Średnia nagroda: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Najlepszy wynik: {max(total_rewards):.2f}")
        print(f"Najgorszy wynik: {min(total_rewards):.2f}")
        print(f"Średnia liczba kroków: {mean_steps:.1f}")
        print(f"Najdłuższy epizod: {max(total_steps)} kroków")
        print(f"Najkrótszy epizod: {min(total_steps)} kroków")
        
        # Analiza sukcesu
        print(f"\nAnaliza wyników (baseline losowy):")
        successful_runs = sum(1 for r in total_rewards if r > 600)
        good_runs = sum(1 for r in total_rewards if r > 300)
        decent_runs = sum(1 for r in total_rewards if r > 100)
        positive_runs = sum(1 for r in total_rewards if r > 0)
        
        print(f"Ukończone tory (>600 pkt): {successful_runs}/{args.episodes} ({successful_runs/args.episodes*100:.1f}%)")
        print(f"Bardzo dobre (>300 pkt): {good_runs}/{args.episodes} ({good_runs/args.episodes*100:.1f}%)")
        print(f"Przyzwoite (>100 pkt): {decent_runs}/{args.episodes} ({decent_runs/args.episodes*100:.1f}%)")
        print(f"Pozytywne wyniki (>0 pkt): {positive_runs}/{args.episodes} ({positive_runs/args.episodes*100:.1f}%)")
        
        # Ocena jako baseline
        print(f"\n💡 Analiza baseline:")
        if successful_runs >= 1:
            print("🤯 NIESPOTYKANE SZCZĘŚCIE! Agent losowy ukończył tor!")
        elif good_runs >= args.episodes * 0.1:
            print("🍀 Niezwykle szczęśliwy agent losowy!")
        elif decent_runs >= args.episodes * 0.3:
            print("🎲 Typowy agent losowy - czasami ma szczęście")
        elif positive_runs >= args.episodes * 0.5:
            print("📊 Normalny baseline - około połowy wyników pozytywnych")
        else:
            print("❌ Bardzo pechowy agent losowy")
        
        # Znaczenie jako baseline
        print(f"\n📈 Znaczenie jako baseline:")
        print(f"• Każdy inteligentny agent powinien osiągać lepsze wyniki")
        print(f"• Średnia nagroda {mean_reward:.2f} to minimum do pokonania")
        print(f"• Wskaźnik sukcesu {successful_runs/args.episodes*100:.1f}% to próg referencyjny")
        
        if mean_reward > 0:
            print(f"• ✅ Baseline wydaje się rozsądny")
        else:
            print(f"• ⚠️ Bardzo niski baseline - środowisko może być trudne")
        
        return {
            'avg_reward': mean_reward,
            'success_rate': successful_runs/args.episodes*100,
            'total_rewards': total_rewards
        }
    
    except Exception as e:
        print(f"❌ Błąd w obliczeniach: {e}")
        return None
  # Dla innych agentów
if __name__ == "__main__":
    main()