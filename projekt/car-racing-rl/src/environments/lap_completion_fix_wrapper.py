import gymnasium as gym
import numpy as np

class LapCompletionFixWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lap_complete_percent = 0.95  # Domyślna wartość
        self.tile_visited_count = 0
        self.total_tiles = 0
        self.visited_tiles = set()
        self.lap_completed = False
        self.last_tile_idx = -1
        self.track_discovered = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.tile_visited_count = 0
        self.visited_tiles.clear()
        self.lap_completed = False
        self.last_tile_idx = -1
        self.track_discovered = False
        
        # Spróbuj znaleźć informacje o torze
        self._discover_track_info()
        
        # Dodaj informacje o konfiguracji
        info['total_tiles'] = self.total_tiles
        info['lap_complete_percent'] = self.lap_complete_percent
        info['required_tiles'] = int(self.total_tiles * self.lap_complete_percent) if self.total_tiles > 0 else 0
        
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Spróbuj odkryć tor jeśli jeszcze nie znaleziony
        if not self.track_discovered:
            self._discover_track_info()
        
        # Śledź odwiedzone płytki
        self._track_tile_visits(reward)
        
        # Sprawdź warunki ukończenie okrążenia
        lap_finished = self._check_lap_completion()
        
        # Jeśli okrążenie ukończone, ale środowisko tego nie wykryło
        if lap_finished and not terminated and not self.lap_completed:
            self.lap_completed = True
            terminated = True
            reward += 100  # Bonus za ukończenie
            info['lap_finished'] = True
            info['completion_reason'] = 'percentage_reached'
            
            if self.total_tiles > 0:
                completion_percent = self.tile_visited_count / self.total_tiles * 100
                print(f"🏁 OKRĄŻENIE UKOŃCZONE! Odwiedzone płytki: {self.tile_visited_count}/{self.total_tiles} ({completion_percent:.1f}%)")
            else:
                print(f"🏁 OKRĄŻENIE UKOŃCZONE! Odwiedzone płytki: {self.tile_visited_count}")
        
        # Dodaj szczegółowe informacje
        info['tiles_visited'] = self.tile_visited_count
        info['total_tiles'] = self.total_tiles
        info['completion_progress'] = self.tile_visited_count / max(self.total_tiles, 1)
        info['lap_completed'] = lap_finished or self.lap_completed
        
        return obs, reward, terminated, truncated, info
    
    def _discover_track_info(self):
        """Znajdź informacje o torze w zagnieżdżonym środowisku"""
        try:
            # Metoda 1: Twoja klasa CarRacingEnv -> env -> unwrapped
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'unwrapped'):
                gym_env = self.env.env.unwrapped
                if hasattr(gym_env, 'track') and gym_env.track:
                    self.total_tiles = len(gym_env.track)
                    self.track_discovered = True
                    print(f"🗺️  Tor odkryty (metoda 1)! Płytek: {self.total_tiles}")
                    return
                elif hasattr(gym_env, 'road') and gym_env.road:
                    self.total_tiles = len(gym_env.road)
                    self.track_discovered = True
                    print(f"🗺️  Tor odkryty (metoda 1 - road)! Płytek: {self.total_tiles}")
                    return
            
        except Exception as e:
            print(f"⚠️  Błąd podczas odkrywania toru: {e}")
    
    def _track_tile_visits(self, reward):
        """Śledź odwiedzone płytki na podstawie nagród"""
        # W Car Racing nagroda > 0 oznacza nową płytkę
        if reward > 0.5:  # Próg aby uniknąć małych nagród za inne rzeczy
            self.tile_visited_count += 1
            
    def _check_lap_completion(self):
        """Sprawdź czy okrążenie zostało ukończone"""
        if self.total_tiles == 0:
            # Jeśli nie znamy total_tiles, użyj heurystyki
            # Na podstawie obserwacji - typowe ukończenie przy ~250-300 płytkach
            heuristic_threshold = 250 * self.lap_complete_percent
            return self.tile_visited_count >= heuristic_threshold
            
        # Sprawdź czy osiągnięto wymagany procent
        completion_ratio = self.tile_visited_count / self.total_tiles
        required_ratio = self.lap_complete_percent
        
        return completion_ratio >= required_ratio
    
    def debug_track_info(self):
        """Debuguj informacje o torze - ulepszona wersja"""
        print(f"=== DEBUG TORU ===")
        print(f"env type: {type(self.env)}")
        
        # Sprawdź strukturę zagnieżdżenia
        current_env = self.env
        level = 0
        while current_env and level < 5:  # Zabezpieczenie przed nieskończoną pętlą
            print(f"Level {level}: {type(current_env)}")
            
            # Sprawdź atrybuty toru na tym poziomie
            track_attrs = []
            if hasattr(current_env, 'track'):
                track_attrs.append(f"track: {getattr(current_env, 'track', None)}")
            if hasattr(current_env, 'road'):
                track_attrs.append(f"road: {getattr(current_env, 'road', None)}")
            if hasattr(current_env, 'tile_visited_count'):
                track_attrs.append(f"tile_visited_count: {getattr(current_env, 'tile_visited_count', None)}")
            
            if track_attrs:
                print(f"  Track attrs: {track_attrs}")
            
            # Przejdź do następnego poziomu
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            elif hasattr(current_env, 'unwrapped'):
                current_env = current_env.unwrapped
            else:
                break
            level += 1
        
        print(f"=== KONIEC DEBUG ===")