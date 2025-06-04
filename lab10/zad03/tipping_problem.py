from simpful import *
import matplotlib.pyplot as plt
import numpy as np

# Utwórz system rozmyty
FS = FuzzySystem()

# Zdefiniuj zmienne lingwistyczne z jawnym uniwersum dyskursu
# Jakość jedzenia (0-10)
food_var = LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=2, d=4), term="poor"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="average"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10, d=10), term="good")
], universe_of_discourse=[0, 10])
FS.add_linguistic_variable("food", food_var)

# Jakość obsługi (0-10)
service_var = LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=2, d=4), term="poor"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="average"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10, d=10), term="good")
], universe_of_discourse=[0, 10])
FS.add_linguistic_variable("service", service_var)

# Wysokość napiwku (0-30%)
tip_var = LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=5, d=10), term="low"),
    FuzzySet(function=Triangular_MF(a=10, b=15, c=20), term="medium"),
    FuzzySet(function=Trapezoidal_MF(a=20, b=25, c=30, d=30), term="high")
], universe_of_discourse=[0, 30])
FS.add_linguistic_variable("tip", tip_var)

# Dodaj reguły (przykład z 9 regułami zamiast 3 dla kompletności)
FS.add_rules([
    "IF (food IS poor) AND (service IS poor) THEN (tip IS low)",
    "IF (food IS average) AND (service IS poor) THEN (tip IS low)",
    "IF (food IS good) AND (service IS poor) THEN (tip IS medium)",
    "IF (food IS poor) AND (service IS average) THEN (tip IS low)",
    "IF (food IS average) AND (service IS average) THEN (tip IS medium)",
    "IF (food IS good) AND (service IS average) THEN (tip IS medium)",
    "IF (food IS poor) AND (service IS good) THEN (tip IS medium)",
    "IF (food IS average) AND (service IS good) THEN (tip IS high)",
    "IF (food IS good) AND (service IS good) THEN (tip IS high)"
])

print("🔧 System rozmyty skonfigurowany z następującymi regułami:")
for i, rule in enumerate(FS._rules, 1):
    print(f"   {i}. {rule}")

# c) Funkcja do wyświetlania wykresów zmiennych lingwistycznych
def plot_membership_functions(fuzzy_sets, title, x_max=10):
    """
    Rysuje funkcje przynależności dla zmiennej lingwistycznej
    """
    x = np.linspace(0, x_max, 1000)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'orange', 'green']
    for i, fuzzy_set in enumerate(fuzzy_sets):
        y = []
        for xi in x:
            try:
                membership_value = fuzzy_set.get_value(xi)
                y.append(membership_value)
            except:
                y.append(0)
        
        plt.plot(x, y, label=fuzzy_set._term, linewidth=3, color=colors[i % len(colors)])
        plt.fill_between(x, y, alpha=0.2, color=colors[i % len(colors)])
    
    plt.title(f"Funkcje przynależności - {title}", fontsize=14, fontweight='bold')
    plt.xlabel("Wartość", fontsize=12)
    plt.ylabel("Stopień przynależności", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

# Wyświetl wykresy funkcji przynależności
print("\n📊 c) Wykresy zmiennych lingwistycznych:")

# Stwórz zbiory dla wizualizacji
food_sets = [
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=2, d=4), term="poor"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="average"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10, d=10), term="good")
]

service_sets = [
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=2, d=4), term="poor"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="average"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10, d=10), term="good")
]

tip_sets = [
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=5, d=10), term="low"),
    FuzzySet(function=Triangular_MF(a=10, b=15, c=20), term="medium"),
    FuzzySet(function=Trapezoidal_MF(a=20, b=25, c=30, d=30), term="high")
]

plot_membership_functions(food_sets, "Jakość jedzenia", 10)
plot_membership_functions(service_sets, "Jakość obsługi", 10)
plot_membership_functions(tip_sets, "Wysokość napiwku", 30)

# d) Testowanie kontrolera
def test_controller():
    """
    d) Przetestuj działanie kontrolera - kilka przykładów
    """
    print("\n" + "="*60)
    print("📋 d) TESTOWANIE KONTROLERA NAPIWKÓW")
    print("="*60)
    print(f"{'Jedzenie':<12} {'Obsługa':<12} {'Napiwek':<15} {'Opis'}")
    print("-" * 60)
    
    # Przykładowe dane wejściowe
    test_cases = [
        (2.0, 3.0, "Słabe jedzenie, słaba obsługa"),
        (5.0, 5.0, "Średnie jedzenie, średnia obsługa"),
        (8.0, 8.0, "Dobre jedzenie, dobra obsługa"),
        (9.0, 2.0, "Bardzo dobre jedzenie, słaba obsługa"),
        (3.0, 9.0, "Słabe jedzenie, bardzo dobra obsługa"),
        (1.0, 1.0, "Bardzo słabe wszystko"),
        (10.0, 10.0, "Perfekcyjne wszystko")
    ]
    
    for food, service, description in test_cases:
        # Ustaw wartości wejściowe
        FS.set_variable("food", food)
        FS.set_variable("service", service)
        
        # Wykonaj wnioskowanie
        tip_result = FS.inference()
        tip_value = tip_result["tip"]
        
        # Wyświetl wynik
        print(f"{food:<12.1f} {service:<12.1f} {tip_value:<7.1f}%      {description}")

# Uruchom testy
test_controller()

print("\n" + "="*60)
print("✅ SYSTEM NAPIWKÓW - PODSUMOWANIE")
print("="*60)
print("📊 System zawiera:")
print("   • 3 zmienne lingwistyczne (jedzenie, obsługa, napiwek)")
print("   • 9 reguł logiki rozmytej")
print("   • Funkcje przynależności: trapezoidalne i trójkątne")
print("\n🎯 System logicznie oblicza napiwki na podstawie jakości!")





# ============================================================
# 📋 d) TESTOWANIE KONTROLERA NAPIWKÓW
# ============================================================
# Jedzenie     Obsługa      Napiwek         Opis
# ------------------------------------------------------------
# 2.0          3.0          4.4    %      Słabe jedzenie, słaba obsługa
# 5.0          5.0          15.0   %      Średnie jedzenie, średnia obsługa
# 8.0          8.0          26.1   %      Dobre jedzenie, dobra obsługa
# 9.0          2.0          15.0   %      Bardzo dobre jedzenie, słaba obsługa
# 3.0          9.0          15.0   %      Słabe jedzenie, bardzo dobra obsługa
# 1.0          1.0          3.9    %      Bardzo słabe wszystko
# 10.0         10.0         26.1   %      Perfekcyjne wszystko

# ============================================================
# ✅ SYSTEM NAPIWKÓW - PODSUMOWANIE
# ============================================================
# 📊 System zawiera:
#    • 3 zmienne lingwistyczne (jedzenie, obsługa, napiwek)
#    • 9 reguł logiki rozmytej
#    • Funkcje przynależności: trapezoidalne i trójkątne