import numpy as np
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'reward_per_episode': [],
            'steps_to_completion': [],
            'completion_rate': 0.0,
            'training_time': 0.0,
            'convergence_episode': 0,
            'stability_score': 0.0,
            'efficiency_score': 0.0
        }
    
    def calculate_all_metrics(self, rewards, steps, training_time):
        # Wskaźnik ukończenia
        self.completion_rate = sum(1 for r in rewards if r > 600) / len(rewards)
        
        # Stabilność (odchylenie standardowe)
        self.stability_score = 1 / (1 + np.std(rewards))
        
        # Efektywność (nagroda/czas)
        self.efficiency_score = np.mean(rewards) / training_time
        
        # Konwergencja (kiedy osiągnął stabilne wyniki)
        self.convergence_episode = self.find_convergence_point(rewards)