import numpy as np
import time
import os
import glob
import cv2

def preprocess_state(state):
    """
    Bardzo szybki preprocessing obrazu dla DQN
    
    Args:
        state: Surowy obraz z środowiska CarRacing (96x96x3 RGB)
        
    Returns:
        processed_state: Preprocessowany obraz (84x84x1)
        
    Transformacje:
    1. RGB → Grayscale (3 kanały → 1 kanał, 3x szybsze)
    2. 96x96 → 84x84
    3. [0-255] → [0-1] (normalizacja dla sieci neuronowej)
    4. Reshape do formatu TensorFlow (height, width, channels)

    - Mniejszy obraz = szybszy trening
    """
    # === 1. KONWERSJA RGB → GRAYSCALE ===
    # Zmniejszamy z 3 kanałów (RGB) do 1 kanału (grayscale)
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # (96,96,3) → (96,96)
    
    # === 2. ZMIANA ROZMIARU ===

    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # (96,96) → (84,84)
    
    # === 3. NORMALIZACJA ===
    # Piksele z [0-255] → [0-1] (standard dla sieci neuronowych)
    # Dzielenie przez 255.0 (float) zapewnia wartości zmiennoprzecinkowe
    normalized = resized / 255.0  # (84,84) z wartościami [0,1]
    
    # === 4. RESHAPE DLA TENSORFLOW ===
    # TensorFlow CNN oczekuje (height, width, channels)
    # Dodajemy wymiar kanału: (84,84) → (84,84,1)
    return normalized.reshape(84, 84, 1)

def train_dqn(env, agent, episodes=10, max_steps=1000):
    """
    Główna pętla treningu DQN/DDQN
    
    Args:
        env: Środowisko CarRacing
        agent: Agent DQN z załadowanym modelem
        episodes: Liczba epizodów treningu
        max_steps: Maksymalna liczba kroków na epizod (zapobiega nieskończonym pętlom)
    
    Proces treningu:
    1. Reset środowiska → nowy stan początkowy
    2. Dla każdego kroku: action → step → train → repeat
    3. Zbieranie statystyk i monitoring postępu
    4. Automatyczne zapisywanie co 5 epizodów (w DQNAgent)
    """
    # === INICJALIZACJA TRENINGU ===
    total_rewards = []           # Historia nagród ze wszystkich epizodów
    start_time = time.time()     # Timer do pomiaru czasu treningu
    
    print(f"🚀 Rozpoczynanie treningu DQN: {episodes} epizodów, max {max_steps} kroków/epizod")
    
    # === GŁÓWNA PĘTLA EPIZODÓW ===
    for episode in range(1, episodes + 1):
        print(f"\n=== EPIZOD {episode}/{episodes} ===")
        
        # === 1. RESET ŚRODOWISKA ===
        # Każdy epizod zaczyna się od nowego, losowego stanu początkowego
        state = env.reset()
        print(f"📍 Stan początkowy wczytany (kształt: {state.shape})")
        
        # Preprocessing: (96,96,3) → (84,84,1)
        state = preprocess_state(state)
        
        # Zmienne epizodu
        total_reward = 0    # Suma nagród w tym epizodzie
        steps = 0          # Licznik kroków w epizodzie
        
        # === 2. PĘTLA KROKÓW W EPIZODZIE ===
        for step in range(max_steps):
            # === MONITORING POSTĘPU ===
            # Pokaż progress co 50 kroków (żeby nie zaśmiecać konsoli)
            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  🔄 Krok {step}/{max_steps}, Czas treningu: {elapsed:.2f}s, "
                      f"Nagroda: {total_reward:.2f}, ε: {agent.epsilon:.4f}")
            
            # === 3. WYBIERZ AKCJĘ ===
            # Agent używa epsilon-greedy: losowa vs najlepsza akcja
            action = agent.act(state)  # 0-4 (nic, lewo, prawo, gaz, hamuj)
            
            # === 4. WYKONAJ KROK W ŚRODOWISKU ===
            # CarRacing zwraca: następny_stan, nagroda, czy_skończony, czy_przerwany, info
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Preprocessing następnego stanu
            next_state = preprocess_state(next_state)
            
            # === 5. OKREŚL CZY EPIZOD SIĘ KOŃCZY ===
            # Epizod kończy się gdy:
            # - terminated: agent osiągnął cel lub wyleciał z toru
            # - truncated: przekroczono limit czasu środowiska
            # - step == max_steps-1: nasz limit bezpieczeństwa
            done = terminated or truncated or (step == max_steps - 1)
            
            # === 6. TRENUJ AGENTA ===
            # Zapisz doświadczenie (s,a,r,s',done) i potencjalnie trenuj sieć
            agent.train(state, action, reward, next_state, done)
            
            # === 7. PRZEJDŹ DO NASTĘPNEGO STANU ===
            state = next_state
            total_reward += reward  # Akumuluj nagrody
            steps += 1
            
            # === 8. SPRAWDŹ WARUNKI KOŃCA EPIZODU ===
            if terminated or truncated:
                reason = "🏁 Terminated" if terminated else "⏰ Truncated"
                print(f"  {reason} po {steps} krokach")
                break
        
        # === STATYSTYKI EPIZODU ===
        total_rewards.append(total_reward)
        
        # Średnia z ostatnich 10 epizodów (rolling average)
        recent_rewards = total_rewards[-min(len(total_rewards), 10):]
        avg_reward = np.mean(recent_rewards)
        
        # === RAPORT POSTĘPU ===
        elapsed_time = time.time() - start_time
        print(f"✅ Epizod {episode} ukończony:")
        print(f"   Nagroda: {total_reward:.2f}")
        print(f"   Średnia (10 ep): {avg_reward:.2f}")
        print(f"   Kroki: {steps}/{max_steps}")
        print(f"   Czas treningu: {elapsed_time:.2f}s")
        print(f"   Epsilon: {agent.epsilon:.4f}")
        print(f"   Pamięć: {len(agent.memory)}/{agent.memory.maxlen}")
        
        # === OCENA POSTĘPU ===
        # Jakościowa ocena jak agent sobie radzi
        if total_reward > 600:
            performance = "🏆 DOSKONALE! (ukończył tor)"
        elif total_reward > 300:
            performance = "🚗 DOBRZE (solidna jazda)"
        elif total_reward > 100:
            performance = "🎯 ŚREDNIO (pewien postęp)"
        elif total_reward > 0:
            performance = "🤔 SŁABO (ale pozytywnie)"
        else:
            performance = "❌ BARDZO SŁABO (nagroda ujemna)"
        
        print(f"   Ocena: {performance}")
        
        # === ANALIZA TRENDU ===
        # Sprawdź czy agent się poprawia
        if len(total_rewards) >= 5:
            recent_avg = np.mean(total_rewards[-5:])    # Ostatnie 5
            older_avg = np.mean(total_rewards[-10:-5])   # Poprzednie 5
            
            if len(total_rewards) >= 10:
                if recent_avg > older_avg:
                    trend = "📈 POPRAWA"
                elif recent_avg < older_avg - 50:
                    trend = "📉 POGORSZENIE"
                else:
                    trend = "➡️ STABILNY"
                print(f"   Trend: {trend}")
    
    # === KOŃCOWY ZAPIS ===
    print(f"\n{'='*50}")
    print("🎯 TRENING ZAKOŃCZONY")
    print("="*50)
    print(f"Łączny czas treningu: {time.time() - start_time:.2f}s")
    print(f"Ostatnia nagroda: {total_rewards[-1]:.2f}")
    print(f"Najlepsza nagroda: {max(total_rewards):.2f}")
    print(f"Średnia wszystkich: {np.mean(total_rewards):.2f}")
    
    print("💾 Zapisywanie końcowego modelu...")
    try:
        # Zapisz model jako "dqn_final_model" (niezależnie od liczby epizodów)
        agent.save_model(custom_name="dqn_final_model")
        print("✅ Model końcowy zapisany!")
    except Exception as e:
        print(f"❌ Błąd podczas zapisywania końcowego modelu: {e}")
        import traceback
        traceback.print_exc()
    
    return total_rewards, agent

def continue_from_checkpoint(checkpoint_path, env, episodes=10):
    """
    Kontynuuje trening z zapisanego checkpointu (Resumable Training)
    
    Args:
        checkpoint_path: Ścieżka do zapisanego modelu (.keras)
        env: Środowisko CarRacing
        episodes: Ile dodatkowych epizodów trenować
    
    Returns:
        total_rewards: Historia nagród z kontynuowanego treningu
        agent: Wytrenowany agent
    
    Proces:
    1. Wczytaj zapisany model + parametry treningu
    2. Przywróć stan agenta (epsilon, episodes, memory)
    3. Kontynuuj trening od punktu przerwania
    
    Użycie:
    >> rewards, agent = continue_from_checkpoint("checkpoints/dqn/dqn_model_ep50.keras", env, 20)
    >> # Kontynuuje trening od epizodu 50, dodając 20 epizodów więcej
    """
    print(f"🔄 KONTYNUACJA TRENINGU Z CHECKPOINTU")
    print(f"📂 Ścieżka: {checkpoint_path}")
    
    try:
        # === 1. WCZYTAJ AGENTA Z CHECKPOINTU ===
        from agents.dqn_agent import DQNAgent
        
        # DQNAgent.load() automatycznie:
        # - Ładuje model (.keras) 
        # - Przywraca parametry treningu (_params.json)
        # - Rekonstruuje target_model
        # - Ustawia epsilon, episodes, steps na zapisane wartości
        agent = DQNAgent.load(checkpoint_path, (84,84,1), 5)
        
        print(f"✅ Agent wczytany z {agent.episodes} epizodów treningu")
        print(f"📊 Stan: steps={agent.steps}, ε={agent.epsilon:.4f}")
        
        # === 2. KONTYNUUJ TRENING ===
        # Używaj tej samej funkcji train_dqn, ale agent pamięta swój stan
        print(f"🚀 Kontynuacja treningu na {episodes} dodatkowych epizodów...")
        
        total_rewards, trained_agent = train_dqn(env, agent, episodes)
        
        print(f"✅ Trening kontynuowany! Łącznie epizodów: {trained_agent.episodes}")
        return total_rewards, trained_agent
        
    except Exception as e:
        print(f"❌ BŁĄD podczas kontynuacji treningu: {e}")
        import traceback
        traceback.print_exc()
        raise