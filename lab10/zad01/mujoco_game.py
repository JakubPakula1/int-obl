import gymnasium as gym

# Utwórz środowisko
# Instalacja: pip install gymnasium[mujoco]
env = gym.make("HalfCheetah-v4", render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

# Uruchom kilka epizodów
for episode in range(3):
    observation, info = env.reset(seed=episode)
    total_reward = 0
    
    for step in range(1000):
        # Losowa akcja
        action = env.action_space.sample()
        
        # Wykonaj akcję
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Zakończ epizod jeśli trzeba
        if terminated or truncated:
            break
    
    print(f"Epizod {episode+1}: Suma nagród = {total_reward:.2f}")

env.close()