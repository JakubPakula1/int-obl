# src/evaluation/report_generator.py
class ResearchReportGenerator:
    def __init__(self):
        self.agents_analyzed = []
        self.comparison_data = {}
        
    def generate_research_paper_style_report(self):
        """Generuje raport w stylu paper naukowego"""
        sections = {
            'abstract': self.generate_abstract(),
            'introduction': self.generate_introduction(),
            'methodology': self.generate_methodology(),
            'results': self.generate_results_section(),
            'discussion': self.generate_discussion(),
            'conclusions': self.generate_conclusions()
        }
        
        # Zapisz jako LaTeX i PDF
        self.save_as_latex(sections)
        self.save_as_markdown(sections)
        
    def generate_results_section(self):
        return f"""
        ## Wyniki eksperymentów
        
        ### Wydajność algorytmów
        
        | Algorytm | Śr. nagroda | Wskaźnik ukończenia | Czas treningu | Stabilność |
        |----------|-------------|-------------------|---------------|------------|
        {self.generate_results_table()}
        
        ### Analiza krzywych uczenia
        
        {self.analyze_learning_curves()}
        
        ### Porównanie efektywności
        
        {self.compare_efficiency()}
        """
        
    def generate_statistical_analysis(self):
        """Przeprowadza testy statystyczne między algorytmami"""
        from scipy import stats
        
        results = {}
        
        # Test t-Studenta między parami algorytmów
        for agent1 in self.agents_analyzed:
            for agent2 in self.agents_analyzed:
                if agent1 != agent2:
                    data1 = self.comparison_data[agent1]['final_rewards']
                    data2 = self.comparison_data[agent2]['final_rewards']
                    
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    results[f"{agent1}_vs_{agent2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return results