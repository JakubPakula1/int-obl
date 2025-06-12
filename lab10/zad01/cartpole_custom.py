import gymnasium as gym
import numpy as np
#?Stan ciągły + Akcje dyskretne
# Utwórz środowisko
env = gym.make("CartPole-v1", render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

def get_custom_action(observation):
    """
    Custom action strategy for CartPole:
    - Move cart in the direction the pole is falling
    - observation[0] is cart position
    - observation[1] is cart velocity
    - observation[2] is pole angle
    - observation[3] is pole angular velocity
    """
    pole_angle = observation[2]
    pole_velocity = observation[3]
    
    # If pole is falling right (positive angle), move right
    # If pole is falling left (negative angle), move left
    if pole_angle + 0.1 * pole_velocity > 0:
        return 1  # Move right
    else:
        return 0  # Move left

# Uruchom kilka epizodów
for episode in range(5):
    observation, info = env.reset(seed=episode)
    total_reward = 0
    
    for step in range(500):
        # Użyj własnej strategii zamiast losowej akcji
        action = get_custom_action(observation)
        
        # Wykonaj akcję
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Zakończ epizod jeśli trzeba
        if terminated or truncated:
            break
    
    print(f"Epizod {episode+1}: Suma nagród = {total_reward}")

env.close() 