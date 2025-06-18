import numpy as np

class RandomAgent:
    """Agent wykonujący losowe akcje - baseline do porównań"""
    
    def __init__(self, action_type='continuous'):
        """
        Args:
            action_type: 'continuous' lub 'discrete'
        """
        self.action_type = action_type
        
    def act(self, observation=None):
        """
        Zwraca losową akcję
        Args:
            observation: Obserwacja środowiska (ignorowana przez agenta losowego)
        Returns:
            action: Losowa akcja
        """
        if self.action_type == 'continuous':
            # Continuous action space: [steering, gas, brake]
            # steering: -1.0 do +1.0 (lewo/prawo)
            # gas: 0.0 do +1.0 (nie gaz/pełny gaz)  
            # brake: 0.0 do +1.0 (nie hamuj/pełne hamowanie)
            
            # Losowe akcje z większym prawdopodobieństwem jazdy do przodu
            steering = np.random.uniform(-1.0, 1.0)
            gas = np.random.uniform(0.0, 1.0) if np.random.random() > 0.1 else 0.0  # 90% szans na gaz
            brake = np.random.uniform(0.0, 1.0) if np.random.random() > 0.8 else 0.0  # 20% szans na hamowanie
            
            return np.array([steering, gas, brake], dtype=np.float32)
        
        else:
            # Discrete action space (dla DQN)
            # 0: nic nie rób
            # 1: skręć w lewo  
            # 2: skręć w prawo
            # 3: gaz
            # 4: hamuj
            return np.random.randint(0, 5)
    
    def reset(self):
        """Reset agenta (nic nie robi - agent bezstanowy)"""
        pass
    
    def save(self, filepath):
        """Zapisz model (nic nie robi - agent nie ma parametrów)"""
        print(f"💾 Random agent nie wymaga zapisywania - brak parametrów do zapisania")
    
    @classmethod
    def load(cls, filepath):
        """Wczytaj model (zwraca nowy agent)"""
        print(f"📂 Random agent nie wymaga wczytywania - tworzę nowy agent")
        return cls()