# Uruchom kompletną analizę
import sys
sys.path.append('src')

from evaluation.benchmark import AutoBenchmark
from evaluation.visualization import ComparisonDashboard
from evaluation.raport_generator import ResearchReportGenerator
from evaluation.video_recorder import ProgressionRecorder

# 1. Benchmark wszystkich agentów
print("🔬 Uruchamiam benchmark...")
benchmark = AutoBenchmark(
    agents=['dqn', 'neat', 'ppo', 'random'],
    episodes_per_test=25
)
benchmark.run_full_comparison()

# 2. Generuj wizualizacje
print("📊 Tworzę wizualizacje...")
dashboard = ComparisonDashboard()
dashboard.create_comparison_dashboard()

# 3. Generuj raport naukowy
print("📄 Generuję raport...")
report_gen = ResearchReportGenerator()
report_gen.generate_research_paper_style_report()

# 4. Nagraj filmy progresji
print("🎬 Nagrywam filmy...")
for agent in ['dqn', 'neat', 'ppo']:
    recorder = ProgressionRecorder(agent)
    recorder.record_progression_videos()
    recorder.create_progression_montage()

print("✅ Wszystko gotowe!")