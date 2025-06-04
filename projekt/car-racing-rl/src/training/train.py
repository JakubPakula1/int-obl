def train(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    import gym
    from src.agents.dqn_agent import DQNAgent
    from src.environments.car_racing_env import CarRacingEnv

    env = CarRacingEnv()
    agent = DQNAgent()
    train(agent, env)