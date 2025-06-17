import sys
import os

# Dodaj ścieżkę do głównego katalogu src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
from stable_baselines3 import PPO
import time
from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper

env = gym.make("CarRacing-v3", render_mode="human")
env = LapCompletionFixWrapper(env)

# Sprawdź czy model istnieje
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