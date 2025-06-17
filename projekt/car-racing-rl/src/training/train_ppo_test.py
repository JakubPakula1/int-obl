import sys
import os

# Dodaj ścieżkę do głównego katalogu src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper

MODEL_PATH = "models/ppo_carracing1"

# Przygotowanie środowiska
def make_env():
    env = gym.make("CarRacing-v3")
    env = LapCompletionFixWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

# Sprawdź, czy istnieje zapisany model — jeśli tak, wczytaj go
if os.path.exists(MODEL_PATH + ".zip"):
    print("🔄 Ładowanie istniejącego modelu...")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("🚀 Tworzenie nowego modelu...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")

# Kontynuuj trening
model.learn(total_timesteps=100_000, reset_num_timesteps=False)

# Zapisz ponownie
model.save(MODEL_PATH)
env.close()