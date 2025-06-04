import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.action_space = env.action_space
        
    def act(self, state):
        return self.action_space.sample()
        
    def replay(self, state, action, reward, next_state, done):
        pass  # Agent losowy nie uczy się