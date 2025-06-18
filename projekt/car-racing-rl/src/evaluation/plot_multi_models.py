import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_all_models_from_json(json_file_path):
    """Wczytaj wyniki wszystkich modeli z JSON i stwórz wykresy dla każdego"""
    
    # Wczytaj dane
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Ładowanie wyników porównawczych z: {os.path.basename(json_file_path)}")
    print("=" * 60)
    
    # Procesuj każdy model
    for model_name, model_data in data.items():
        print(f"\n🔍 Przetwarzanie modelu: {model_name.upper()}")
        create_model_analysis(model_name, model_data, json_file_path)
    
    # Stwórz także wykres porównawczy
    create_comparison_plot(data, json_file_path)

def create_model_analysis(model_name, model_data, original_file_path):
    """Stwórz szczegółową analizę dla pojedynczego modelu"""
    
    rewards = model_data['rewards']
    mean_reward = model_data['mean']
    std_reward = model_data['std']
    completion_rate = model_data['completion_rate']
    
    print(f"   📈 Średnia: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   🏆 Ukończenia: {completion_rate:.1f}%")
    print(f"   📝 Epizody: {len(rewards)}")
    
    # Ustaw styl
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Stwórz figurę z subplotami
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Szczegółowa Analiza Modelu {model_name.upper()} - {len(rewards)} epizodów', 
                 fontsize=16, fontweight='bold')
    
    # 1. Nagrody w czasie (górny lewy)
    ax1 = axes[0, 0]
    episodes = list(range(1, len(rewards) + 1))
    
    ax1.plot(episodes, rewards, 'o-', linewidth=2, markersize=3, alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.axhline(y=600, color='red', linestyle='--', alpha=0.7, label='Próg ukończenia (600)')
    ax1.axhline(y=mean_reward, color='orange', linestyle='--', alpha=0.8, label=f'Średnia: {mean_reward:.1f}')
    
    # Trend line
    if len(rewards) > 1:
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
    ax2.hist(rewards, bins=min(20, len(set(rewards))//2 + 5), alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(x=mean_reward, color='red', linestyle='--', linewidth=2, 
                label=f'Średnia: {mean_reward:.1f}')
    ax2.axvline(x=np.median(rewards), color='orange', linestyle='--', linewidth=2, 
                label=f'Mediana: {np.median(rewards):.1f}')
    
    if model_name != 'random':  # Random ma prawie wszystkie negatywne
        ax2.axvline(x=600, color='green', linestyle='--', alpha=0.7, label='Próg (600)')
    
    ax2.set_title('Rozkład nagród')
    ax2.set_xlabel('Nagroda')
    ax2.set_ylabel('Częstość')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot kategorii (górny prawy)
    ax3 = axes[0, 2]
    
    # Podziel na kategorie w zależności od modelu
    if model_name == 'random':
        # Dla random: negatywne/zero/pozytywne
        negative = [r for r in rewards if r < -50]
        moderate = [r for r in rewards if -50 <= r < 0]
        positive = [r for r in rewards if r >= 0]
        
        data_to_plot = []
        labels = []
        colors = []
        
        if negative:
            data_to_plot.append(negative)
            labels.append(f'< -50\n({len(negative)} ep.)')
            colors.append('red')
        if moderate:
            data_to_plot.append(moderate)
            labels.append(f'-50 do 0\n({len(moderate)} ep.)')
            colors.append('orange')
        if positive:
            data_to_plot.append(positive)
            labels.append(f'≥ 0\n({len(positive)} ep.)')
            colors.append('lightgreen')
    else:
        # Dla innych: słabe/średnie/dobre/doskonałe
        poor = [r for r in rewards if r < 400]
        medium = [r for r in rewards if 400 <= r < 600]
        good = [r for r in rewards if 600 <= r < 900]
        excellent = [r for r in rewards if r >= 900]
        
        data_to_plot = []
        labels = []
        colors = []
        
        if poor:
            data_to_plot.append(poor)
            labels.append(f'< 400\n({len(poor)} ep.)')
            colors.append('lightcoral')
        if medium:
            data_to_plot.append(medium)
            labels.append(f'400-600\n({len(medium)} ep.)')
            colors.append('lightyellow')
        if good:
            data_to_plot.append(good)
            labels.append(f'600-900\n({len(good)} ep.)')
            colors.append('lightgreen')
        if excellent:
            data_to_plot.append(excellent)
            labels.append(f'≥ 900\n({len(excellent)} ep.)')
            colors.append('gold')
    
    if data_to_plot:
        bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax3.set_title('Kategorie wyników')
    ax3.set_ylabel('Nagroda')
    ax3.grid(True, alpha=0.3)
    
    # 4. Przesuwające się okno średniej (dolny lewy)
    ax4 = axes[1, 0]
    
    window_size = min(10, len(rewards)//3)
    if len(rewards) >= window_size and window_size > 1:
        moving_avg = []
        window_centers = []
        
        for i in range(window_size, len(rewards) + 1):
            window_rewards = rewards[i-window_size:i]
            moving_avg.append(np.mean(window_rewards))
            window_centers.append(i - window_size/2)
        
        ax4.plot(window_centers, moving_avg, 'o-', linewidth=2, markersize=4, color='orange')
        ax4.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.7, label=f'Ogólna średnia: {mean_reward:.1f}')
        
        if model_name != 'random':
            ax4.axhline(y=600, color='green', linestyle='--', alpha=0.5, label='Próg (600)')
        
        ax4.set_title(f'Średnia krocząca (okno {window_size} epizodów)')
        ax4.set_xlabel('Epizod (środek okna)')
        ax4.set_ylabel('Średnia nagroda')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Za mało danych\ndla średniej kroczącej', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Średnia krocząca')
    
    # 5. Statystyki tekstowe (dolny środek)
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Oblicz dodatkowe statystyki
    best_reward = max(rewards)
    worst_reward = min(rewards)
    above_600 = len([r for r in rewards if r >= 600])
    above_mean = len([r for r in rewards if r >= mean_reward])
    
    # Percentyle
    q25 = np.percentile(rewards, 25)
    q75 = np.percentile(rewards, 75)
    
    stats_text = f"""
STATYSTYKI {model_name.upper()}

📊 Podstawowe:
• Średnia: {mean_reward:.2f}
• Odchylenie std: {std_reward:.2f}
• Mediana: {np.median(rewards):.2f}
• Q1: {q25:.2f}
• Q3: {q75:.2f}

📈 Extrema:
• Najlepszy: {best_reward:.2f}
• Najgorszy: {worst_reward:.2f}
• Zakres: {best_reward - worst_reward:.2f}

🏆 Wydajność:
• Ukończenia (≥600): {completion_rate:.1f}%
• Powyżej średniej: {above_mean}/{len(rewards)} ({above_mean/len(rewards)*100:.1f}%)
• Współczyn. zmienności: {abs(std_reward/mean_reward)*100 if mean_reward != 0 else 0:.1f}%

📏 Rozkład:
• Epizody: {len(rewards)}
• Sukces/Porażka: {above_600}/{len(rewards) - above_600}
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 6. Analiza wzorca/trendu (dolny prawy)
    ax6 = axes[1, 2]
    
    # Porównaj trzecie części
    third = len(rewards) // 3
    if third > 0:
        first_third = rewards[:third]
        second_third = rewards[third:2*third]
        last_third = rewards[2*third:]
        
        periods = []
        period_labels = []
        
        if first_third:
            periods.append(first_third)
            period_labels.append(f'1/3\n({len(first_third)} ep.)\nŚr: {np.mean(first_third):.1f}')
        
        if second_third:
            periods.append(second_third)
            period_labels.append(f'2/3\n({len(second_third)} ep.)\nŚr: {np.mean(second_third):.1f}')
        
        if last_third:
            periods.append(last_third)
            period_labels.append(f'3/3\n({len(last_third)} ep.)\nŚr: {np.mean(last_third):.1f}')
        
        if periods:
            bp = ax6.boxplot(periods, labels=period_labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        
        ax6.set_title('Postęp w czasie (trzecie)')
        ax6.set_ylabel('Nagroda')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Za mało danych\ndla analizy trendu', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Analiza trendu')
    
    plt.tight_layout()
    
    # Zapisz wykres
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    output_file = f'results/{base_name}_{model_name}_analysis_{timestamp}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Wykres zapisany: {output_file}")
    
    plt.show()

def create_comparison_plot(data, original_file_path):
    """Stwórz wykres porównawczy wszystkich modeli"""
    
    print(f"\n🔄 Tworzenie wykresu porównawczego...")
    
    plt.figure(figsize=(16, 10))
    
    # Przygotuj dane do porównania
    models = list(data.keys())
    means = [data[model]['mean'] for model in models]
    stds = [data[model]['std'] for model in models]
    completion_rates = [data[model]['completion_rate'] for model in models]
    
    # Subplot 1: Średnie wyniki
    plt.subplot(2, 3, 1)
    bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['gold', 'lightgreen', 'lightblue', 'lightcoral'])
    plt.title('Średnie wyniki z odchyleniem standardowym')
    plt.ylabel('Średnia nagroda')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Dodaj wartości na słupkach
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(means)*0.05,
                f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Wskaźniki ukończenia
    plt.subplot(2, 3, 2)
    bars = plt.bar(models, completion_rates, alpha=0.7, 
                   color=['gold', 'lightgreen', 'lightblue', 'lightcoral'])
    plt.title('Wskaźnik ukończenia (%)')
    plt.ylabel('Procent ukończeń')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, rate in zip(bars, completion_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: Box plot porównawczy
    plt.subplot(2, 3, 3)
    all_rewards = [data[model]['rewards'] for model in models]
    bp = plt.boxplot(all_rewards, labels=[model.upper() for model in models], patch_artist=True)
    colors = ['gold', 'lightgreen', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Rozkład wszystkich wyników')
    plt.ylabel('Nagroda')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Histogram porównawczy
    plt.subplot(2, 3, 4)
    colors = ['gold', 'lightgreen', 'lightblue', 'lightcoral']
    for i, model in enumerate(models):
        plt.hist(data[model]['rewards'], alpha=0.6, label=model.upper(), 
                color=colors[i], bins=20)
    
    plt.axvline(x=600, color='red', linestyle='--', alpha=0.7, label='Próg (600)')
    plt.title('Histogramy porównawcze')
    plt.xlabel('Nagroda')
    plt.ylabel('Częstość')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Stabilność (CV)
    plt.subplot(2, 3, 5)
    cv_values = [abs(data[model]['std']/data[model]['mean'])*100 if data[model]['mean'] != 0 else 0 
                 for model in models]
    bars = plt.bar(models, cv_values, alpha=0.7, 
                   color=['gold', 'lightgreen', 'lightblue', 'lightcoral'])
    plt.title('Stabilność (Współczynnik zmienności)')
    plt.ylabel('CV (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, cv in zip(bars, cv_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(cv_values)*0.02,
                f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 6: Ranking table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Stwórz ranking
    ranking_data = []
    for model in models:
        ranking_data.append({
            'Model': model.upper(),
            'Średnia': data[model]['mean'],
            'Ukończenia': data[model]['completion_rate'],
            'Stabilność': cv_values[models.index(model)]
        })
    
    # Sortuj według średniej
    ranking_data.sort(key=lambda x: x['Średnia'], reverse=True)
    
    ranking_text = "🏆 RANKING MODELI\n\n"
    for i, model_data in enumerate(ranking_data, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranking_text += f"{medal} {model_data['Model']}\n"
        ranking_text += f"   Średnia: {model_data['Średnia']:.1f} pkt\n"
        ranking_text += f"   Ukończenia: {model_data['Ukończenia']:.1f}%\n"
        ranking_text += f"   Stabilność: {model_data['Stabilność']:.1f}% CV\n\n"
    
    plt.text(0.05, 0.95, ranking_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.suptitle('Porównanie Wszystkich Modeli RL', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Zapisz wykres porównawczy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    output_file = f'results/{base_name}_comparison_{timestamp}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Wykres porównawczy zapisany: {output_file}")
    
    plt.show()

def main():
    """Główna funkcja"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generuj wykresy dla wszystkich modeli z JSON')
    parser.add_argument('json_file', type=str, help='Ścieżka do pliku JSON z wynikami porównawczymi')
    parser.add_argument('--only_comparison', action='store_true', help='Tylko wykres porównawczy')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"❌ Plik nie istnieje: {args.json_file}")
        return
    
    print(f"📈 Tworzenie wykresów z: {args.json_file}")
    
    if args.only_comparison:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        create_comparison_plot(data, args.json_file)
    else:
        plot_all_models_from_json(args.json_file)

if __name__ == "__main__":
    main()