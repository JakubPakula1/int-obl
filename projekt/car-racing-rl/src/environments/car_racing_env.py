import gymnasium as gym
import numpy as np

class CarRacingEnv:
    def __init__(self, render_mode=None):
        self.env = gym.make('CarRacing-v3', render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Mapowanie akcji dyskretnych na ciągłe
        self.discrete_actions = [
            np.array([0.0, 0.0, 0.0]),    # Brak akcji
            np.array([-1.0, 0.0, 0.0]),   # Skręt w lewo
            np.array([1.0, 0.0, 0.0]),    # Skręt w prawo
            np.array([0.0, 1.0, 0.0]),    # Przyspieszenie
            np.array([0.0, 0.0, 1.0])     # Hamowanie
        ]
        
    def reset(self):
        observation, info = self.env.reset()
        return observation
        
    def step(self, action):
        # Konwersja indeksu akcji na wektor akcji
        if isinstance(action, (int, np.integer)):
            continuous_action = self.discrete_actions[action]
            return self.env.step(continuous_action)
        # Jeśli akcja jest już w formacie ciągłym, używamy jej bezpośrednio
        return self.env.step(action)
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()