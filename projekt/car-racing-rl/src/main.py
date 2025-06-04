from environments.car_racing_env import CarRacingEnv
from agents.neat_agent import NEATAgent
from agents.random_agent import RandomAgent
from training.train_neat import train_neat, continue_training, continue_from_checkpoint
from evaluation.evaluate import evaluate
import os

def main():


    env = CarRacingEnv(render_mode=None)
    
    # Sprawdź checkpointy zamiast pojedynczego modelu
    import glob
    checkpoints = glob.glob('checkpoints/neat-checkpoint-*')
    
    if checkpoints:
        # Użyj najnowszego checkpointu
        latest_checkpoint = sorted(checkpoints)[-1]
        print(f"Kontynuowanie z checkpointu: {latest_checkpoint}")
        neat_agent, winner = continue_from_checkpoint(latest_checkpoint, env, 10)
    else:
        print("Rozpoczynanie nowego treningu...")
        neat_agent, winner = train_neat(env, generations=10)
        
        # Test...
    
    # Test najlepszego agenta
    print("Testowanie najlepszego agenta NEAT...")
    env_visual = CarRacingEnv(render_mode="human")
    
    observation = env_visual.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(1000):
        action = neat_agent.act(observation)
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