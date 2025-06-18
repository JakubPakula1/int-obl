import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark import AutoBenchmark

class EnhancedBenchmark(AutoBenchmark):
    def __init__(self, agents=['neat', 'dqn', 'ppo', 'random'], episodes_per_test=25):
        super().__init__(agents, episodes_per_test)
        self.detailed_results = {}
        self.statistical_tests = {}
    
    def run_statistical_analysis(self):
        """Przeprowadź kompleksową analizę statystyczną"""
        from scipy import stats
        
        print("📊 Przeprowadzam analizę statystyczną...")
        
        # Zbierz dane do analizy
        agent_rewards = {}
        for agent in self.agents:
            # Pobierz wyniki z poprzednich testów lub uruchom nowe
            rewards = self.get_agent_rewards(agent)
            agent_rewards[agent] = rewards
        
        # ANOVA - test różnic między wszystkimi grupami
        all_rewards = list(agent_rewards.values())
        f_stat, p_value = stats.f_oneway(*all_rewards)
        
        self.statistical_tests['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Testy t-Studenta między parami
        t_tests = {}
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                t_stat, p_val = stats.ttest_ind(
                    agent_rewards[agent1], 
                    agent_rewards[agent2]
                )
                
                # Effect size (Cohen's d)
                mean1, mean2 = np.mean(agent_rewards[agent1]), np.mean(agent_rewards[agent2])
                std1, std2 = np.std(agent_rewards[agent1]), np.std(agent_rewards[agent2])
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std
                
                t_tests[f"{agent1}_vs_{agent2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'significant': p_val < 0.05,
                    'effect_size': self.interpret_cohens_d(cohens_d)
                }
        
        self.statistical_tests['t_tests'] = t_tests
        return self.statistical_tests
    
    def interpret_cohens_d(self, d):
        """Interpretuj wielkość efektu Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "mały efekt"
        elif abs_d < 0.5:
            return "średni efekt"
        elif abs_d < 0.8:
            return "duży efekt"
        else:
            return "bardzo duży efekt"
    
    def run_robustness_test(self):
        """Test odporności - jak agenci radzą sobie z różnymi warunkami"""
        robustness_results = {}
        
        # Test 1: Różne seedy (różne tory)
        print("🎲 Test różnych torów (seed variation)...")
        for agent in self.agents:
            seed_results = []
            for seed in [42, 123, 456, 789, 999]:
                # Uruchom test z konkretnym seedem
                rewards = self.test_agent_with_seed(agent, seed, episodes=5)
                seed_results.extend(rewards)
            
            robustness_results[agent] = {
                'seed_variance': np.var(seed_results),
                'seed_stability': 1 / (1 + np.var(seed_results)),  # Im mniejsza wariancja, tym lepiej
                'mean_performance': np.mean(seed_results)
            }
        
        return robustness_results
    
    def create_comprehensive_visualizations(self):
        """Stwórz kompleksowe wizualizacje"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        # 1. Box plot porównawczy
        self.plot_performance_comparison(axes[0,0])
        
        # 2. Violin plot
        self.plot_distribution_comparison(axes[0,1])
        
        # 3. Radar chart
        self.plot_radar_comparison(axes[0,2])
        
        # 4. Krzywe uczenia (jeśli dostępne)
        self.plot_learning_curves(axes[1,0])
        
        # 5. Stabilność vs wydajność
        self.plot_stability_vs_performance(axes[1,1])
        
        # 6. Heatmapa porównań
        self.plot_comparison_heatmap(axes[1,2])
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_benchmark_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_research_paper_report(self):
        """Wygeneruj raport w stylu artykułu naukowego"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Analiza Porównawcza Algorytmów Uczenia przez Wzmacnianie w Środowisku CarRacing-v3

**Data analizy:** {timestamp}
**Liczba epizodów testowych:** {self.episodes_per_test} na algorytm
**Środowisko:** OpenAI Gym CarRacing-v3

## Streszczenie

Przeprowadzono kompleksową analizę porównawczą {len(self.agents)} algorytmów uczenia przez wzmacnianie 
w środowisku symulacji wyścigów samochodowych. Badanie obejmowało analizę wydajności, stabilności 
oraz odporności algorytmów na zmienne warunki środowiska.

## Metodologia

### Środowisko testowe
- **Środowisko:** CarRacing-v3 (OpenAI Gymnasium)
- **Metryka sukcesu:** Nagroda > 600 punktów (ukończenie toru)
- **Kryterium dobrego wyniku:** Nagroda > 300 punktów
- **Maksymalny czas epizodu:** 1000 kroków

### Analizowane algorytmy
{self.generate_algorithm_descriptions()}

### Metodyka testowania
1. **Test wydajności podstawowej:** {self.episodes_per_test} epizodów na algorytm
2. **Test stabilności:** Wielokrotne uruchomienia z różnymi seedami
3. **Analiza statystyczna:** ANOVA + testy t-Studenta + wielkość efektu
4. **Test odporności:** Różne warunki początkowe

## Wyniki

### Statystyki opisowe
{self.generate_descriptive_statistics()}

### Analiza statystyczna
{self.generate_statistical_analysis_report()}

### Ranking algorytmów
{self.generate_algorithm_ranking()}

## Dyskusja wyników

{self.generate_discussion()}

## Wnioski

{self.generate_conclusions()}

## Rekomendacje

{self.generate_recommendations()}

---
*Raport wygenerowany automatycznie przez Enhanced Benchmark System*
"""
        
        with open(f'results/research_paper_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md', 'w') as f:
            f.write(report)
        
        print("📄 Raport naukowy zapisany w results/")
        return report
    
    def export_data_for_external_analysis(self):
        """Eksportuj dane do analizy zewnętrznej (R, SPSS, etc.)"""
        
        # DataFrame z wszystkimi wynikami
        all_data = []
        for agent in self.agents:
            rewards = self.get_agent_rewards(agent)
            for i, reward in enumerate(rewards):
                all_data.append({
                    'agent': agent,
                    'episode': i+1,
                    'reward': reward,
                    'success': 1 if reward > 600 else 0,
                    'good_performance': 1 if reward > 300 else 0
                })
        
        df = pd.DataFrame(all_data)
        
        # Zapisz w różnych formatach
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'results/benchmark_data_{timestamp}.csv', index=False)
        df.to_excel(f'results/benchmark_data_{timestamp}.xlsx', index=False)
        
        # Statystyki podsumowujące
        summary_stats = df.groupby('agent').agg({
            'reward': ['mean', 'std', 'min', 'max', 'median'],
            'success': 'mean',
            'good_performance': 'mean'
        }).round(3)
        
        summary_stats.to_csv(f'results/benchmark_summary_{timestamp}.csv')
        
        print(f"📊 Dane eksportowane:")
        print(f"  - results/benchmark_data_{timestamp}.csv")
        print(f"  - results/benchmark_data_{timestamp}.xlsx") 
        print(f"  - results/benchmark_summary_{timestamp}.csv")
        
        return df

if __name__ == "__main__":
    # Uruchom rozszerzony benchmark
    enhanced_benchmark = EnhancedBenchmark(
        agents=['dqn', 'neat', 'ppo', 'random'],
        episodes_per_test=25
    )
    
    # Pełna analiza
    print("🚀 Rozpoczynam rozszerzony benchmark...")
    
    enhanced_benchmark.run_full_comparison()
    enhanced_benchmark.run_statistical_analysis()
    enhanced_benchmark.run_robustness_test()
    enhanced_benchmark.create_comprehensive_visualizations()
    enhanced_benchmark.generate_research_paper_report()
    enhanced_benchmark.export_data_for_external_analysis()
    
    print("✅ Analiza zakończona!")