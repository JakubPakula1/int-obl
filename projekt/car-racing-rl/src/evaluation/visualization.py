# src/evaluation/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ComparisonDashboard:
    def __init__(self):
        self.agents_data = {}
        
    def create_comparison_dashboard(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Krzywa uczenia
        self.plot_learning_curves(axes[0,0])
        
        # 2. Box plot wyników finałowych
        self.plot_final_performance_boxplot(axes[0,1])
        
        # 3. Wskaźnik ukończenia toru
        self.plot_completion_rates(axes[0,2])
        
        # 4. Czas treningu vs wydajność
        self.plot_efficiency_scatter(axes[1,0])
        
        # 5. Stabilność wyników
        self.plot_stability_comparison(axes[1,1])
        
        # 6. Radar chart wszystkich metryk
        self.plot_radar_chart(axes[1,2])
        
        plt.tight_layout()
        plt.savefig('results/comparison_dashboard.png', dpi=300, bbox_inches='tight')
        
    def plot_learning_curves(self, ax):
        """Krzywe uczenia dla wszystkich agentów"""
        for agent_name, data in self.agents_data.items():
            episodes = range(1, len(data['rewards']) + 1)
            # Wygładzona krzywa (moving average)
            smoothed = pd.Series(data['rewards']).rolling(window=10).mean()
            ax.plot(episodes, smoothed, label=agent_name, linewidth=2)
        
        ax.set_xlabel('Epizod')
        ax.set_ylabel('Średnia nagroda (10 epizodów)')
        ax.set_title('Krzywe uczenia')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_radar_chart(self, ax):
        """Radar chart porównujący wszystkie metryki"""
        categories = ['Wydajność', 'Stabilność', 'Efektywność', 
                     'Szybkość uczenia', 'Ukończenia', 'Czas treningu']
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Zamknij okrąg
        
        for agent_name, data in self.agents_data.items():
            values = self.normalize_metrics_for_radar(data)
            values += values[:1]  # Zamknij okrąg
            ax.plot(angles, values, 'o-', linewidth=2, label=agent_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Porównanie wszystkich metryk')
        ax.legend()