import numpy as np
import time
import cv2

def preprocess_state(state):
    """Preprocessing obrazu - identyczny jak w DQN"""
    if len(state.shape) == 3:
        # Konwersja do skali szarości
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Resize do 84x84
        resized = cv2.resize(gray, (84, 84))
        # Normalizacja
        normalized = resized.astype(np.float32) / 255.0
        # Dodaj wymiar kanału
        return np.expand_dims(normalized, axis=-1)
    return state

def train_ppo(env, agent, episodes=100, max_steps=1000):
    """Trenowanie agenta PPO"""
    total_rewards = []
    start_time = time.time()
    
    print(f"🚀 Rozpoczynanie treningu PPO na {episodes} epizodów")
    
    for episode in range(1, episodes + 1):
        observation, info = env.reset()
        state = preprocess_state(observation)
        episode_reward = 0
        steps = 0
        
        episode_start_time = time.time()
        
        for step in range(max_steps):
            # Pobierz akcję i wartość stanu
            action, log_prob = agent.get_action(state)
            state_batch = np.expand_dims(state, axis=0)
            value = agent.critic(state_batch, training=False).numpy()[0][0]
            
            # Wykonaj akcję w środowisku
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(next_observation)
            
            done = terminated or truncated or (step == max_steps - 1)
            
            # Zapisz przejście
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Wykonaj aktualizację jeśli bufor jest pełny
            if agent.should_update():
                print(f"📊 Wykonywanie aktualizacji PPO po {len(agent.states)} krokach...")
                agent.update()
            
            if terminated or truncated:
                break
        
        # Zakończenie epizodu
        agent.episodes += 1
        episode_time = time.time() - episode_start_time
        
        # Statystyki epizodu
        total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards[-min(len(total_rewards), 10):])
        
        # Analiza rezultatu
        if terminated and episode_reward > 500:
            result = "🏆 DOSKONAŁY WYNIK!"
        elif terminated and episode_reward > 200:
            result = "🚗 DOBRY WYNIK!"
        elif terminated and episode_reward > 0:
            result = "✅ POZYTYWNY WYNIK"
        elif episode_reward > -50:
            result = "⚠️ SŁABY WYNIK"
        else:
            result = "❌ BARDZO SŁABY WYNIK"
        
        print(f"Epizod {episode}/{episodes}: {steps} kroków, {episode_reward:.2f} pkt, "
              f"średnia: {avg_reward:.2f}, czas: {episode_time:.1f}s - {result}")
        
        # Zapisz model co 20 epizodów
        if agent.episodes % 20 == 0:
            agent.save_model()
            print(f"💾 Model zapisany po epizodzie {agent.episodes}")
    
    # Finalna aktualizacja jeśli są pozostałe dane
    if len(agent.states) > 0:
        print("📊 Finalna aktualizacja PPO...")
        agent.update()
    
    # Zapisz finalny model
    agent.save_model(custom_name="ppo_final_model")
    
    total_time = time.time() - start_time
    print(f"\n🎯 Trening PPO zakończony!")
    print(f"⏱️ Całkowity czas: {total_time:.2f}s")
    print(f"📈 Średnia nagroda: {np.mean(total_rewards):.2f}")
    print(f"🏆 Najlepszy wynik: {max(total_rewards):.2f}")
    
    return agent, total_rewards