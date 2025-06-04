class RuleBasedAgent:
    def __init__(self):
        pass

    def act(self, observation):
        # Implementacja reguł podejmowania decyzji na podstawie obserwacji
        # Przykładowa logika: 
        # - Jeśli samochód jest blisko ściany, skręć w przeciwną stronę
        # - Jeśli prędkość jest zbyt niska, przyspiesz
        # - W przeciwnym razie, jedź prosto
        action = [0, 0, 0]  # [steering, acceleration, brake]
        
        # Przykładowe reguły
        if observation['distance_to_wall'] < 1.0:
            action[0] = -1  # Skręć w lewo
        elif observation['speed'] < 1.0:
            action[1] = 1   # Przyspiesz
        else:
            action[0] = 0   # Jedź prosto
        
        return action