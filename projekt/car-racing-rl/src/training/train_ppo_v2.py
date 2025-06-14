import numpy as np
import time
import cv2

def preprocess_state(state):
    """Preprocessing obrazu dla PPO"""
    # Sprawdź czy state to tuple (observation, info)
    if isinstance(state, tuple):
        state = state[0]
    
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return normalized.reshape(84, 84, 1)

def calculate_reward_bonus(info, prev_tiles, step_count):
    """Oblicz bonus za eksplorację"""
    tiles_visited = info.get('tiles_visited', 0)
    bonus = 0
    
    # Bonus za nowe płytki
    if tiles_visited > prev_tiles:
        bonus += (tiles_visited - prev_tiles)
    
    # Bonus za ukończenie toru
    if info.get('lap_completed', False):
        bonus += 100.0
    
    # Penalty za długie stanie w miejscu
    if step_count > 100 and tiles_visited < 10:
        bonus -= 0.2
        
    if tiles_visited >= 10:  # Jeśli agent się porusza
        bonus += 0.1  # Mały bonus za aktywność
    return bonus, tiles_visited

def train_ppo_enhanced(env, agent, episodes=100, max_steps=1000):
    """Ulepszone trenowanie PPO z dodatkowymi nagrodami"""
    from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper
    
    # Dodaj wrapper dla lepszej diagnostyki
    env = LapCompletionFixWrapper(env)
    
    total_rewards = []
    exploration_progress = []
    start_time = time.time()
    
    print(f"🚀 Rozpoczynanie ulepszonego treningu PPO na {episodes} epizodów")
    print(f"🔧 Dodatkowe nagrody za eksplorację i ukończenie toru")
    
    for episode in range(1, episodes + 1):
        observation, info = env.reset()
        print(f"\n🗺️ Epizod {episode}: Tor {info.get('total_tiles', '?')} płytek")
        
        state = preprocess_state(observation)
        episode_reward = 0
        modified_reward = 0  # Nagroda z bonusami
        steps = 0
        prev_tiles = info.get('tiles_visited', 0)
        max_tiles_this_episode = prev_tiles
        
        episode_start_time = time.time()
        
        for step in range(max_steps):
            # Pobierz akcję i wartość stanu
            action, log_prob = agent.get_action(state)
            state_batch = np.expand_dims(state, axis=0)
            value = agent.critic(state_batch, training=False).numpy()[0][0]
            
            # Wykonaj akcję w środowisku
            next_observation, base_reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(next_observation)
            
            # Oblicz bonusy
            reward_bonus, current_tiles = calculate_reward_bonus(info, prev_tiles, step)
            total_reward = base_reward + reward_bonus
            
            # Aktualizuj maksymalną liczbę płytek
            max_tiles_this_episode = max(max_tiles_this_episode, current_tiles)
            prev_tiles = current_tiles
            
            done = terminated or truncated or (step == max_steps - 1)
            
            # Zapisz przejście z zmodyfikowaną nagrodą
            agent.store_transition(state, action, total_reward, value, log_prob, done)
            
            state = next_state
            episode_reward += base_reward  # Oryginalna nagroda
            modified_reward += total_reward  # Nagroda z bonusami
            steps += 1
            
            # Diagnostyka co 250 kroków
            if step % 250 == 0 and step > 0:
                progress_pct = (current_tiles / info.get('total_tiles', 1)) * 100
                print(f"  📊 Krok {step}: Podstawowa={episode_reward:.1f}, "
                      f"Z bonusami={modified_reward:.1f}, Płytki={current_tiles} ({progress_pct:.1f}%)")
            
            # Aktualizacja PPO
            if agent.should_update():
                print(f"📊 Wykonywanie aktualizacji PPO po {len(agent.states)} krokach...")
                agent.update()
            
            if terminated or truncated:
                break
        
        # Zakończenie epizodu
        agent.episodes += 1
        episode_time = time.time() - episode_start_time
        
        # Statystyki
        total_rewards.append(episode_reward)
        exploration_progress.append(max_tiles_this_episode)
        avg_reward = np.mean(total_rewards[-min(len(total_rewards), 10):])
        avg_exploration = np.mean(exploration_progress[-min(len(exploration_progress), 10):])
        
        # Szczegółowa analiza rezultatu
        completion_status = "🏁 UKOŃCZONE" if info.get('lap_completed', False) else "❌ Nieukończone"
        
        if info.get('lap_completed', False):
            result = "🏆 DOSKONAŁY - TOR UKOŃCZONY!"
        elif episode_reward > 200:
            result = "🚗 BARDZO DOBRY WYNIK!"
        elif episode_reward > 50:
            result = "✅ DOBRY WYNIK"
        elif episode_reward > 0:
            result = "➕ POZYTYWNY WYNIK"
        elif episode_reward > -30:
            result = "⚠️ SŁABY WYNIK"
        else:
            result = "❌ BARDZO SŁABY WYNIK"
        
        print(f"Epizod {episode}/{episodes}: {steps} kroków")
        print(f"  💰 Nagroda: {episode_reward:.2f} (z bonusami: {modified_reward:.2f})")
        print(f"  📈 Średnia (10): {avg_reward:.2f}")
        print(f"  🗺️ Płytki: {max_tiles_this_episode}/{info.get('total_tiles', '?')} (śr: {avg_exploration:.1f})")
        print(f"  ⏱️ Czas: {episode_time:.1f}s - {result}")
        print(f"  🎯 Status: {completion_status}")
        
        # Zapisz model co 20 epizodów
        if agent.episodes % 20 == 0:
            agent.save_model()
            print(f"💾 Model zapisany po epizodzie {agent.episodes}")
        
        # Early stopping jeśli bardzo dobre wyniki
        recent_rewards = total_rewards[-5:] if len(total_rewards) >= 5 else total_rewards
        if len(recent_rewards) >= 3 and np.mean(recent_rewards) > 300:
            print(f"🎉 Early stopping - doskonałe wyniki! Średnia z 5 ostatnich: {np.mean(recent_rewards):.2f}")
            break
    
    # Finalna aktualizacja
    if len(agent.states) > 0:
        print("📊 Finalna aktualizacja PPO...")
        agent.update()
    
    # Zapisz finalny model
    agent.save_model(custom_name="ppo_enhanced_final")
    
    # Końcowe statystyki
    total_time = time.time() - start_time
    completed_laps = sum(1 for r in total_rewards if r > 500)  # Oszacowanie ukończonych tras
    
    print(f"\n🎯 Trening PPO Enhanced zakończony!")
    print(f"⏱️ Całkowity czas: {total_time/60:.1f} minut")
    print(f"📈 Średnia nagroda: {np.mean(total_rewards):.2f}")
    print(f"🏆 Najlepszy wynik: {max(total_rewards):.2f}")
    print(f"📊 Średnia eksploracja: {np.mean(exploration_progress):.1f} płytek")
    print(f"🏁 Prawdopodobne ukończenia: {completed_laps}/{episodes}")
    
    # Trend uczenia się
    if len(total_rewards) >= 20:
        early_avg = np.mean(total_rewards[:20])
        late_avg = np.mean(total_rewards[-20:])
        improvement = late_avg - early_avg
        print(f"📈 Poprawa (pierwsze 20 vs ostatnie 20): {improvement:.2f} pts")
    
    return agent, total_rewards, exploration_progress