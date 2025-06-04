import gymnasium as gym
import numpy as np

# Utwórz środowisko
env = gym.make("LunarLander-v3", render_mode="human")
print(f"Środowisko: {env.spec.id}")
print(f"Przestrzeń obserwacji: {env.observation_space}")
print(f"Przestrzeń akcji: {env.action_space}")

def get_custom_action(observation):
    """
    Custom action strategy for LunarLander:
    - observation[0] is x position
    - observation[1] is y position
    - observation[2] is x velocity
    - observation[3] is y velocity
    - observation[4] is angle
    - observation[5] is angular velocity
    - observation[6] is left leg contact
    - observation[7] is right leg contact
    """
    x_pos, y_pos = observation[0], observation[1]
    x_vel, y_vel = observation[2], observation[3]
    angle = observation[4]
    angular_vel = observation[5]
    
    # Default action is no-op
    action = 0
    
    # If we're too far left, move right
    if x_pos < -0.2:
        action = 1  # Main engine
    # If we're too far right, move left
    elif x_pos > 0.2:
        action = 3  # Main engine + right
    
    # If we're too high or falling too fast, fire main engine
    if y_pos > 0.5 or y_vel < -0.5:
        if action == 0:  # If no horizontal correction needed
            action = 2  # Main engine
        elif action == 1:  # If moving right
            action = 2  # Main engine
        elif action == 3:  # If moving left
            action = 2  # Main engine
    
    # If we're tilted too much, try to correct
    if abs(angle) > 0.2:
        if angle > 0:  # Tilted right
            action = 1  # Move right to correct
        else:  # Tilted left
            action = 3  # Move left to correct
    
    return action

# Uruchom kilka epizodów
for episode in range(5):
    observation, info = env.reset(seed=episode)
    total_reward = 0
    
    for step in range(1000):
        # Użyj własnej strategii zamiast losowej akcji
        action = get_custom_action(observation)
        
        # Wykonaj akcję
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Zakończ epizod jeśli trzeba
        if terminated or truncated:
            break
    
    print(f"Epizod {episode+1}: Suma nagród = {total_reward:.2f}")

env.close() 