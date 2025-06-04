from simpful import *
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

class LunarLanderFuzzyController:
    """
    Kontroler rozmyty dla LunarLander-v2 (continuous)
    
    Obserwacje (8 zmiennych):
    0: x pozycja
    1: y pozycja  
    2: x prędkość
    3: y prędkość
    4: kąt
    5: prędkość kątowa
    6: lewa noga dotyka (bool)
    7: prawa noga dotyka (bool)
    
    Akcje (2 zmienne ciągłe):
    0: główny silnik (-1 do 1)
    1: silnik boczny (-1 do 1)
    """
    
    def __init__(self):
        self.setup_fuzzy_systems()
        
    def setup_fuzzy_systems(self):
        """e) Definiowanie zmiennych lingwistycznych"""
        
        # System dla głównego silnika (thrust)
        self.thrust_system = FuzzySystem()
        
        # Zmienne wejściowe dla głównego silnika
        # 1. Wysokość (y pozycja) - normalizowana do [-1, 1]
        height_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="low"),
            FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="medium"), 
            FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="high")
        ], universe_of_discourse=[-1, 1])
        self.thrust_system.add_linguistic_variable("height", height_var)
        
        # 2. Prędkość pionowa (y velocity)
        y_velocity_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-2, b=-2, c=-0.5, d=0), term="falling"),
            FuzzySet(function=Triangular_MF(a=-0.3, b=0, c=0.3), term="stable"),
            FuzzySet(function=Trapezoidal_MF(a=0, b=0.5, c=2, d=2), term="rising")
        ], universe_of_discourse=[-2, 2])
        self.thrust_system.add_linguistic_variable("y_velocity", y_velocity_var)
        
        # 3. Kąt statku
        angle_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="left"),
            FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="straight"),
            FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="right")
        ], universe_of_discourse=[-1, 1])
        self.thrust_system.add_linguistic_variable("angle", angle_var)
        
        # Zmienna wyjściowa - siła głównego silnika
        thrust_output_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="none"),
            FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="weak"),
            FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="strong")
        ], universe_of_discourse=[-1, 1])
        self.thrust_system.add_linguistic_variable("thrust", thrust_output_var)
        
        # System dla silnika bocznego (torque)
        self.torque_system = FuzzySystem()
        
        # Zmienne wejściowe dla silnika bocznego
        # 1. Kąt statku (powtarzamy)
        self.torque_system.add_linguistic_variable("angle", angle_var)
        
        # 2. Prędkość kątowa
        angular_velocity_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-2, b=-2, c=-0.3, d=-0.1), term="spinning_left"),
            FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="stable"),
            FuzzySet(function=Trapezoidal_MF(a=0.1, b=0.3, c=2, d=2), term="spinning_right")
        ], universe_of_discourse=[-2, 2])
        self.torque_system.add_linguistic_variable("angular_velocity", angular_velocity_var)
        
        # 3. Prędkość pozioma
        x_velocity_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-2, b=-2, c=-0.2, d=0.02), term="moving_left"),
            FuzzySet(function=Triangular_MF(a=-0.3, b=0, c=0.3), term="stable"),
            FuzzySet(function=Trapezoidal_MF(a=0.2, b=0.2, c=2, d=2), term="moving_right")
        ], universe_of_discourse=[-2, 2])
        self.torque_system.add_linguistic_variable("x_velocity", x_velocity_var)
        
        # Zmienna wyjściowa - moment obrotowy
        torque_output_var = LinguisticVariable([
            FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="left"),
            FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="none"),
            FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="right")
        ], universe_of_discourse=[-1, 1])
        self.torque_system.add_linguistic_variable("torque", torque_output_var)
        
        self.setup_rules()
    
    def setup_rules(self):
        """f) Definiowanie reguł wnioskowania rozmytego"""
        
        # Reguły dla głównego silnika (bez zmian)
        thrust_rules = [
            "IF (y_velocity IS falling) THEN (thrust IS strong)",
            "IF (height IS low) AND (y_velocity IS falling) THEN (thrust IS strong)",
            "IF (height IS low) AND (y_velocity IS stable) THEN (thrust IS strong)",
            "IF (height IS low) AND (y_velocity IS rising) THEN (thrust IS weak)",
            "IF (height IS medium) AND (y_velocity IS falling) THEN (thrust IS strong)",
            "IF (height IS medium) AND (y_velocity IS stable) THEN (thrust IS weak)",
            "IF (height IS medium) AND (y_velocity IS rising) THEN (thrust IS none)",
            "IF (height IS high) AND (y_velocity IS falling) THEN (thrust IS weak)",
            "IF (height IS high) AND (y_velocity IS stable) THEN (thrust IS none)",
         "IF (height IS high) AND (y_velocity IS rising) THEN (thrust IS none)", 
        ]
        
        # POPRAWIONE reguły dla silnika bocznego
        torque_rules = [
            # Korekta kąta - podstawowe reguły
            "IF (angle IS left) THEN (torque IS right)",
            "IF (angle IS right) THEN (torque IS left)",
            "IF (angle IS straight) THEN (torque IS none)",
            
            # Przeciwdziałanie obrotowi
            "IF (angular_velocity IS spinning_left) THEN (torque IS right)",
            "IF (angular_velocity IS spinning_right) THEN (torque IS left)",
            "IF (angular_velocity IS stable) THEN (torque IS none)",
            
            # Korekta pozioma
            "IF (x_velocity IS moving_left) THEN (torque IS right)",
            "IF (x_velocity IS moving_right) THEN (torque IS left)",
            "IF (x_velocity IS stable) THEN (torque IS none)",
            
            # Kombinowane reguły dla precyzyjnej korekty
            "IF (angle IS left) AND (angular_velocity IS spinning_right) THEN (torque IS none)",
            "IF (angle IS right) AND (angular_velocity IS spinning_left) THEN (torque IS none)",
        ]
        
        self.thrust_system.add_rules(thrust_rules)
        self.torque_system.add_rules(torque_rules)
        
        print("🔧 Systemy rozmyte skonfigurowane:")
        print(f"   • Główny silnik: {len(thrust_rules)} reguł")
        print(f"   • Silnik boczny: {len(torque_rules)} reguł")
    
    def normalize_observation(self, obs):
        """Normalizuje obserwacje do odpowiednich zakresów"""
        x, y, x_vel, y_vel, angle, ang_vel, leg1, leg2 = obs
        
        # Normalizacja do zakresu [-1, 1]
        normalized = {
            'height': np.clip(y, -1, 1),  # y już jest w odpowiednim zakresie
            'y_velocity': np.clip(y_vel, -2, 2),
            'angle': np.clip(angle, -1, 1),
            'angular_velocity': np.clip(ang_vel*2, -2, 2),
            'x_velocity': np.clip(x_vel * 3, -2, 2)
        }
        
        return normalized
    
    def get_action(self, observation):
        """g) Obliczanie akcji na podstawie obserwacji"""
        
        # Normalizuj obserwacje
        norm_obs = self.normalize_observation(observation)
        
        # Ustaw zmienne wejściowe dla systemu głównego silnika
        self.thrust_system.set_variable("height", norm_obs['height'])
        self.thrust_system.set_variable("y_velocity", norm_obs['y_velocity'])
        self.thrust_system.set_variable("angle", norm_obs['angle'])
        
        # Ustaw zmienne wejściowe dla systemu silnika bocznego
        self.torque_system.set_variable("angle", norm_obs['angle'])
        self.torque_system.set_variable("angular_velocity", norm_obs['angular_velocity'])
        self.torque_system.set_variable("x_velocity", norm_obs['x_velocity'])
        
        try:
            # Wykonaj wnioskowanie rozmyte
            thrust_result = self.thrust_system.inference()
            torque_result = self.torque_system.inference()
            
            thrust_action = thrust_result["thrust"]
            torque_action = torque_result["torque"]
            
            # Ogranicz akcje do zakresu [-1, 1]
            thrust_action = np.clip(thrust_action, -1, 1)
            torque_action = np.clip(torque_action, -1, 1)
            
            return np.array([thrust_action, torque_action])
            
        except Exception as e:
            print(f"❌ Błąd wnioskowania: {e}")
            return np.array([0.0, 0.0])  # Akcja domyślna

def plot_membership_functions(controller):
    """e) Wyświetlenie wykresów zmiennych lingwistycznych"""
    
    # Funkcja pomocnicza do rysowania
    def plot_fuzzy_sets(fuzzy_sets, title, x_range, x_label):
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        plt.figure(figsize=(10, 6))
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        for i, fuzzy_set in enumerate(fuzzy_sets):
            y = []
            for xi in x:
                try:
                    y.append(fuzzy_set.get_value(xi))
                except:
                    y.append(0)
            
            plt.plot(x, y, label=fuzzy_set._term, linewidth=2, color=colors[i % len(colors)])
            plt.fill_between(x, y, alpha=0.3, color=colors[i % len(colors)])
        
        plt.title(f"Funkcje przynależności - {title}", fontsize=14, fontweight='bold')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("Stopień przynależności", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()
    
    print("📊 e) Wykresy zmiennych lingwistycznych:")
    
    # Wysokość
    height_sets = [
        FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="low"),
        FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="medium"),
        FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="high")
    ]
    plot_fuzzy_sets(height_sets, "Wysokość", [-1, 1], "Wysokość (znormalizowana)")
    
    # Prędkość pionowa
    y_vel_sets = [
        FuzzySet(function=Trapezoidal_MF(a=-2, b=-2, c=-0.5, d=0), term="falling"),
        FuzzySet(function=Triangular_MF(a=-0.3, b=0, c=0.3), term="stable"),
        FuzzySet(function=Trapezoidal_MF(a=0, b=0.5, c=2, d=2), term="rising")
    ]
    plot_fuzzy_sets(y_vel_sets, "Prędkość pionowa", [-2, 2], "Prędkość Y")
    
    # Kąt
    angle_sets = [
        FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="left"),
        FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="straight"),
        FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="right")
    ]
    plot_fuzzy_sets(angle_sets, "Kąt statku", [-1, 1], "Kąt (znormalizowany)")
    
    # Prędkość kątowa
    ang_vel_sets = [
        FuzzySet(function=Trapezoidal_MF(a=-2, b=-2, c=-0.5, d=0), term="spinning_left"),
        FuzzySet(function=Triangular_MF(a=-0.3, b=0, c=0.3), term="stable"),
        FuzzySet(function=Trapezoidal_MF(a=0, b=0.5, c=2, d=2), term="spinning_right")
    ]
    plot_fuzzy_sets(ang_vel_sets, "Prędkość kątowa", [-2, 2], "Prędkość kątowa")
    
    # Akcje wyjściowe
    thrust_sets = [
        FuzzySet(function=Trapezoidal_MF(a=-1, b=-1, c=-0.3, d=0), term="none"),
        FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term="weak"),
        FuzzySet(function=Trapezoidal_MF(a=0, b=0.3, c=1, d=1), term="strong")
    ]
    plot_fuzzy_sets(thrust_sets, "Siła głównego silnika", [-1, 1], "Siła silnika")

def test_fuzzy_controller():
    """g) Test kontrolera rozmytego w środowisku"""
    
    print("🚀 g) Testowanie kontrolera rozmytego w LunarLander")
    
    # Utwórz środowisko i kontroler
    env = gym.make('LunarLander-v3', continuous=True, render_mode='human')
    controller = LunarLanderFuzzyController()
    
    # Statystyki
    episode_rewards = []
    episode_lengths = []
    
    num_episodes = 5
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n🎮 Epizod {episode + 1}")
        
        while True:
            # Otrzymaj akcję z kontrolera rozmytego
            action = controller.get_action(observation)
            
            # Wykonaj akcję
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Wyświetl informacje co 50 kroków
            if steps % 50 == 0:
                x, y, x_vel, y_vel, angle, ang_vel, leg1, leg2 = observation
                print(f"   Krok {steps:3d}: pos=({x:5.2f},{y:5.2f}), "
                      f"vel=({x_vel:5.2f},{y_vel:5.2f}), "
                      f"angle={angle:5.2f}, reward={total_reward:6.1f}")
            
            time.sleep(0.01)  # Zwolnij dla lepszej wizualizacji
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Ocena lądowania
        if total_reward > 200:
            result = "🏆 DOSKONAŁE LĄDOWANIE!"
        elif total_reward > 100:
            result = "✅ Udane lądowanie"
        elif total_reward > 0:
            result = "⚠️ Słabe lądowanie"
        else:
            result = "❌ Katastrofa"
        
        print(f"   Wynik: {total_reward:6.1f} punktów w {steps} krokach - {result}")
    
    env.close()
    
    # Podsumowanie
    print("\n" + "="*60)
    print("📊 PODSUMOWANIE WYNIKÓW")
    print("="*60)
    print(f"Średnia nagroda: {np.mean(episode_rewards):6.1f} ± {np.std(episode_rewards):5.1f}")
    print(f"Średnia długość: {np.mean(episode_lengths):6.1f} ± {np.std(episode_lengths):5.1f}")
    print(f"Najlepszy wynik: {max(episode_rewards):6.1f}")
    print(f"Najgorszy wynik: {min(episode_rewards):6.1f}")
    
    # Wykres wyników
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, 'bo-', linewidth=2, markersize=8)
    plt.title('Nagrody w epizodach')
    plt.xlabel('Epizod')
    plt.ylabel('Nagroda')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, 'ro-', linewidth=2, markersize=8)
    plt.title('Długość epizodów')
    plt.xlabel('Epizod')
    plt.ylabel('Liczba kroków')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Główna funkcja uruchamiająca wszystkie testy"""
    print("🌙 KONTROLER ROZMYTY DLA LUNAR LANDER")
    print("="*50)
    
    # Stwórz kontroler
    controller = LunarLanderFuzzyController()
    
    # e) Wyświetl wykresy zmiennych lingwistycznych
    # plot_membership_functions(controller)
    
    # f) Informacje o regułach
    print("\n📋 f) Reguły wnioskowania rozmytego:")
    print("   • Używamy operatora AND dla precyzyjnego sterowania")
    print("   • Główny silnik: kontrola wysokości i prędkości pionowej")
    print("   • Silnik boczny: stabilizacja kąta i pozycji poziomej")
    
    # g) Test kontrolera
    input("\nNaciśnij Enter aby rozpocząć test kontrolera...")
    test_fuzzy_controller()

if __name__ == "__main__":
    main() 