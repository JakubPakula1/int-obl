import gymnasium as gym
import numpy as np

class CarRacingEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make('CarRacing-v3', render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self):
        observation, info = self.env.reset()
        return observation
        
    def step(self, action):
        return self.env.step(action)
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()