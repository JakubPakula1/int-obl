def evaluate(agent, env, num_episodes=100):
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    average_reward = sum(total_rewards) / num_episodes
    print(f'Average Reward over {num_episodes} episodes: {average_reward}')
    return average_reward