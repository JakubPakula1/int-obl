# src/evaluation/benchmark.py
class AutoBenchmark:
    def __init__(self, agents=['neat', 'dqn', 'ppo', 'random'], episodes_per_test=50):
        self.agents = agents
        self.episodes_per_test = episodes_per_test
        self.results = {}
        
    def run_full_benchmark(self):
        """Uruchamia pełny benchmark wszystkich agentów"""
        for agent_name in self.agents:
            print(f"\n🧪 Testowanie {agent_name.upper()}...")
            
            # Test wydajności
            performance_results = self.test_agent_performance(agent_name)
            
            # Test stabilności (wielokrotne uruchomienia)
            stability_results = self.test_agent_stability(agent_name, runs=5)
            
            # Test na różnych torach
            track_diversity_results = self.test_track_diversity(agent_name)
            
            self.results[agent_name] = {
                'performance': performance_results,
                'stability': stability_results,
                'diversity': track_diversity_results
            }
            
        # Generuj raport
        self.generate_comprehensive_report()
        
    def test_agent_performance(self, agent_name):
        """Test podstawowej wydajności"""
        import subprocess
        import json
        
        # Uruchom test i zapisz wyniki do JSON
        cmd = f"python src/main.py --agent {agent_name} --mode test --episodes {self.episodes_per_test}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        # Parsuj wyniki (dodaj funkcję parsowania outputu)
        return self.parse_test_output(result.stdout)
        
    def generate_comprehensive_report(self):
        """Generuje kompleksowy raport HTML"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Car Racing RL - Analiza Porównawcza</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric-card { background: #f5f5f5; padding: 20px; margin: 10px; border-radius: 8px; }
                .agent-section { border-left: 4px solid #007acc; padding-left: 20px; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #007acc; color: white; }
            </style>
        </head>
        <body>
            <h1>🏎️ Car Racing RL - Analiza Porównawcza</h1>
            {content}
        </body>
        </html>
        """
        
        content = self.generate_html_content()
        
        with open('results/comprehensive_report.html', 'w') as f:
            f.write(html_template.format(content=content))