import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


# look here! https://towardsdatascience.com/q-learning-for-beginners-2837b777741
# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)

# Q-learning parameters
alpha = 0.6  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.3  # Exploration rate
num_episodes = 10000
max_steps = 200

# Initialize the Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

print('Q-table before training:')
print(q_table)

goal_position = (3, 3)  # Bottom-right corner

# Function to convert state to 2D position
def state_to_position(state):
    row = state // 4
    col = state % 4
    return (row, col)

# Custom reward function based on Manhattan distance to goal
def custom_reward(state, reward, done):
    if done and reward == 1000.0:
        return reward  # Reward for reaching the goal
    position = state_to_position(state)
    distance = abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1])
    return 1 / (distance + 1)  # Inverse of Manhattan distance

# Custom reward function based on Manhattan distance to goal
def custom_reward(state, reward, done):
    if done and reward == 1000.0:
        return reward  # Reward for reaching the goal
    position = state_to_position(state)
    distance = position[0]+position[1]
    return distance # Inverse of Manhattan distance

# Function to choose the next action
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state, :])  # Exploit

# Training the agent
rewards = []

for episode in range(num_episodes):
    if episode % 100 == 0:
        print(f'Running episode {episode}/{num_episodes}')
    state = env.reset()[0]
    total_rewards = 0

    for step in range(max_steps):
        action = choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)

        # Apply custom reward
        reward = custom_reward(new_state, reward, done)
        
        # Update Q-table
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        
        state = new_state
        total_rewards += reward
        
        if done:
            break

    rewards.append(total_rewards)

print('Q-table after training:')
print(q_table)

# Plotting the results
plt.plot(range(num_episodes), rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Q-learning on FrozenLake 8x8')
plt.show()

# Testing the trained agent
total_test_episodes = 100
total_test_rewards = 0

for episode in range(total_test_episodes):
    state = env.reset()[0]
    episode_rewards = 0
    
    for step in range(max_steps):
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)
        
        episode_rewards += reward
        state = new_state
        
        if done:
            break
            
    total_test_rewards += episode_rewards

print(f"Average reward over {total_test_episodes} test episodes: {total_test_rewards / total_test_episodes}")

env.close()
