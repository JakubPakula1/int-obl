import numpy as np
import time
import os
import glob
import cv2

def preprocess_state(state):
    """Bardzo szybki preprocessing obrazu"""
    # Zmniejszenie rozmiaru i konwersja na skalę szarości
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # INTER_AREA jest szybsze
    normalized = resized / 255.0
    return normalized.reshape(1, 84, 84, 1)

def train_dqn(env, agent, episodes=10, max_steps=200):  # Zmniejszona liczba kroków
    total_rewards = []
    start_time = time.time()
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        print(f"Epizod {episode}: Stan początkowy wczytany")
        state = preprocess_state(state)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            if step % 50 == 0:
                print(f"  Krok {step}/{max_steps}, Czas: {time.time() - start_time:.2f}s")
            
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            
            done = terminated or truncated or (step == max_steps - 1)
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"  Epizod zakończony po {steps} krokach")
                break
        
        # Statystyki epizodu
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-min(len(total_rewards), 10):])
        
        elapsed_time = time.time() - start_time
        print(f"Epizod: {episode}/{episodes}, Nagroda: {total_reward:.2f}, "
              f"Średnia: {avg_reward:.2f}, Kroki: {steps}, Czas: {elapsed_time:.2f}s")
    print("Zapisywanie końcowego modelu...")
    try:
        agent.save_model(custom_name="dqn_final_model")
    except Exception as e:
        print(f"Błąd podczas zapisywania końcowego modelu: {e}")

def continue_from_checkpoint(checkpoint_path, env, episodes=10):  # Zmniejszony parametr
    """Kontynuuje trenowanie z zapisanego checkpointu"""
    from agents.dqn_agent import DQNAgent
    
    # Wczytanie agenta
    agent = DQNAgent.load(checkpoint_path, (84,84,1), 5)
    
    # Kontynuowanie treningu
    return train_dqn(env, agent, episodes), agent