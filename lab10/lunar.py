import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(600):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)
   print(action, observation, reward, terminated, truncated)
   if terminated or truncated:
      observation, info = env.reset(seed=42)
env.close()