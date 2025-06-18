import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_and_plot_json_results(json_file_path):
    """Wczytaj wyniki z JSON i stwórz wykresy"""
    
    # Wczytaj dane
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    rewards = results['rewards']
    
    print(f"📊 Ładowanie wyników dla: {results['model_type']}")
    print(f"📁 Model: {results['model_path']}")
    print(f"🎯 Epizody: {results['total_episodes']}")
    print(f"📈 Średnia nagroda: {np.mean(rewards):.2f}")
    print(f"🏆 Wskaźnik ukończenia: {results['completion_rate']:.1f}%")
    
    # Ustaw styl
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Stwórz figurę z subplotami
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Analiza Wyników {results["model_type"]} - {results["total_episodes"]} epizodów', 
                 fontsize=16, fontweight='bold')
    
    # 1. Nagrody w czasie (górny lewy)
    ax1 = axes[0, 0]
    episodes = list(range(1, len(rewards) + 1))
    
    ax1.plot(episodes, rewards, 'o-', linewidth=2, markersize=4, alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Próg ukończenia (600)')
    
    # Trend line
    z = np.polyfit(episodes, rewards, 1)
    p = np.poly1d(z)
    ax1.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax1.set_title('Nagrody w kolejnych epizodach')
    ax1.set_xlabel('Epizod')
    ax1.set_ylabel('Nagroda')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram nagród (górny środek)
    ax2 = axes[0, 1]
    ax2.hist(rewards, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(x=np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Średnia: {np.mean(rewards):.1f}')
    ax2.axvline(x=np.median(rewards), color='orange', linestyle='--', linewidth=2, 
                label=f'Mediana: {np.median(rewards):.1f}')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5, label='Zero')
    
    ax2.set_title('Rozkład nagród')
    ax2.set_xlabel('Nagroda')
    ax2.set_ylabel('Częstość')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot (górny prawy)
    ax3 = axes[0, 2]
    
    # Podziel na kategorie
    positive_rewards = [r for r in rewards if r > 0]
    negative_rewards = [r for r in rewards if r <= 0]
    good_rewards = [r for r in rewards if r > 50]
    
    data_to_plot = []
    labels = []
    
    if negative_rewards:
        data_to_plot.append(negative_rewards)
        labels.append(f'≤0\n({len(negative_rewards)} ep.)')
    
    if positive_rewards:
        data_to_plot.append(positive_rewards)
        labels.append(f'>0\n({len(positive_rewards)} ep.)')
    
    if good_rewards:
        data_to_plot.append(good_rewards)
        labels.append(f'>50\n({len(good_rewards)} ep.)')
    
    if data_to_plot:
        bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightcoral', 'lightgreen', 'gold']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax3.set_title('Kategorie wyników')
    ax3.set_ylabel('Nagroda')
    ax3.grid(True, alpha=0.3)
    
    # 4. Przesuwające się okno średniej (dolny lewy)
    ax4 = axes[1, 0]
    
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = []
        window_centers = []
        
        for i in range(window_size, len(rewards) + 1):
            window_rewards = rewards[i-window_size:i]
            moving_avg.append(np.mean(window_rewards))
            window_centers.append(i - window_size/2)
        
        ax4.plot(window_centers, moving_avg, 'o-', linewidth=2, markersize=4, color='orange')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_title(f'Średnia krocząca (okno {window_size} epizodów)')
        ax4.set_xlabel('Epizod (środek okna)')
        ax4.set_ylabel('Średnia nagroda')
        ax4.grid(True, alpha=0.3)
    
    # 5. Statystyki tekstowe (dolny środek)
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Oblicz dodatkowe statystyki
    positive_count = len([r for r in rewards if r > 0])
    negative_count = len([r for r in rewards if r <= 0])
    best_reward = max(rewards)
    worst_reward = min(rewards)
    std_reward = np.std(rewards)
    
    stats_text = f"""
SZCZEGÓŁOWE STATYSTYKI

📊 Podstawowe:
• Średnia: {np.mean(rewards):.2f}
• Mediana: {np.median(rewards):.2f}
• Odchylenie std: {std_reward:.2f}
• Najlepszy: {best_reward:.2f}
• Najgorszy: {worst_reward:.2f}

📈 Rozkład:
• Pozytywne: {positive_count}/{len(rewards)} ({positive_count/len(rewards)*100:.1f}%)
• Negatywne: {negative_count}/{len(rewards)} ({negative_count/len(rewards)*100:.1f}%)
• >50 pkt: {len([r for r in rewards if r > 50])} epizodów
• >100 pkt: {len([r for r in rewards if r > 100])} epizodów

🎯 Model:
• Typ: {results['model_type']}
• Ukończenie: {results['completion_rate']:.1f}%
• Śr. kroki: {results['avg_steps']:.0f}
• Śr. kafelki: {results['avg_tiles']:.1f}
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 6. Analiza wzorca (dolny prawy)
    ax6 = axes[1, 2]
    
    # Porównaj pierwsze i drugie połowy
    mid_point = len(rewards) // 2
    first_half = rewards[:mid_point]
    second_half = rewards[mid_point:]
    
    comparison_data = []
    comparison_labels = []
    
    if first_half:
        comparison_data.append(first_half)
        comparison_labels.append(f'Pierwsza połowa\n({len(first_half)} ep.)\nŚr: {np.mean(first_half):.1f}')
    
    if second_half:
        comparison_data.append(second_half)
        comparison_labels.append(f'Druga połowa\n({len(second_half)} ep.)\nŚr: {np.mean(second_half):.1f}')
    
    if comparison_data:
        bp = ax6.boxplot(comparison_data, labels=comparison_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax6.set_title('Porównanie okresów')
    ax6.set_ylabel('Nagroda')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Zapisz wykres
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_file = f'results/{base_name}_plots_{timestamp}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Wykres zapisany: {output_file}")
    
    plt.show()
    
    return results

def create_simple_reward_plot(json_file_path):
    """Stwórz prosty wykres nagród w czasie"""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    rewards = data['results']['rewards']
    episodes = list(range(1, len(rewards) + 1))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(episodes, rewards, 'bo-', linewidth=2, markersize=6, alpha=0.7, label='Nagrody')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='Zero')
    plt.axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Próg ukończenia')
    
    # Trend
    z = np.polyfit(episodes, rewards, 1)
    p = np.poly1d(z)
    plt.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    plt.title(f'Wyniki {data["results"]["model_type"]} - {len(rewards)} epizodów', fontsize=14, fontweight='bold')
    plt.xlabel('Epizod')
    plt.ylabel('Nagroda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dodaj statystyki jako tekst
    avg_reward = np.mean(rewards)
    plt.text(0.05, 0.95, f'Średnia: {avg_reward:.2f}\nNajlepszy: {max(rewards):.2f}\nNajgorszy: {min(rewards):.2f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Zapisz
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_file = f'results/{base_name}_simple_plot_{timestamp}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Prosty wykres zapisany: {output_file}")
    
    plt.show()

def main():
    """Główna funkcja"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generuj wykresy z pliku JSON z wynikami')
    parser.add_argument('json_file', type=str, help='Ścieżka do pliku JSON z wynikami')
    parser.add_argument('--simple', action='store_true', help='Stwórz tylko prosty wykres')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"❌ Plik nie istnieje: {args.json_file}")
        return
    
    print(f"📈 Tworzenie wykresów z: {args.json_file}")
    
    if args.simple:
        create_simple_reward_plot(args.json_file)
    else:
        load_and_plot_json_results(args.json_file)

if __name__ == "__main__":
    main()    