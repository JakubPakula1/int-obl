import gymnasium as gym
import numpy as np

class CarRacingEnv(gym.Env):
    def __init__(self, render_mode=None, continuous=True, lap_complete_percent=0.95):
        super().__init__()
        # WAŻNE: Użyj wyższego progu w bazowym środowisku
        # ale implementuj własną logikę
        self.env = gym.make('CarRacing-v3', 
                           lap_complete_percent=1.0,  # Wyłącz wbudowaną logikę
                           render_mode=render_mode, 
                           continuous=continuous)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Własna logika ukończenia
        self.custom_lap_complete_percent = lap_complete_percent
        self.tile_visited_count = 0
        self.total_tiles = 0
        self.lap_completed = False
        
    def reset(self):
        observation, info = self.env.reset()
        
        # Pobierz informacje o torze
        if hasattr(self.env.unwrapped, 'track'):
            self.total_tiles = len(self.env.unwrapped.track)
        
        self.tile_visited_count = 0
        self.lap_completed = False
        
        # Dodaj informacje
        info['total_tiles'] = self.total_tiles
        info['lap_complete_percent'] = self.custom_lap_complete_percent
        info['required_tiles'] = int(self.total_tiles * self.custom_lap_complete_percent)
        
        return observation, info
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Śledź płytki (nagroda > 0 = nowa płytka)
        if reward > 0.5:
            self.tile_visited_count += 1
            
        # Sprawdź własne warunki ukończenia
        if not self.lap_completed and self.total_tiles > 0:
            completion_ratio = self.tile_visited_count / self.total_tiles
            if completion_ratio >= self.custom_lap_complete_percent:
                self.lap_completed = True
                terminated = True
                reward += 200  # Bonus za ukończenie
                info['lap_finished'] = True
                info['completion_reason'] = 'custom_percentage_reached'
                print(f"🏁 OKRĄŻENIE UKOŃCZONE! {self.tile_visited_count}/{self.total_tiles} płytek ({completion_ratio*100:.1f}%)")
        
        # Dodaj informacje o postępie
        info['tiles_visited'] = self.tile_visited_count
        info['total_tiles'] = self.total_tiles
        info['completion_progress'] = self.tile_visited_count / max(self.total_tiles, 1)
        info['lap_completed'] = self.lap_completed
        
        return observation, reward, terminated, truncated, info
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()