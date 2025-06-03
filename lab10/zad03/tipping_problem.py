from simpful import *
import matplotlib.pyplot as plt
import numpy as np

# Utwórz system rozmyty
FS = FuzzySystem()

# Zdefiniuj zmienne lingwistyczne
# Jakość jedzenia (0-10)
FS.add_linguistic_variable("food", LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=2, d=4), term="poor"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="average"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10, d=10), term="good")
]))

# Jakość obsługi (0-10)
FS.add_linguistic_variable("service", LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=2, d=4), term="poor"),
    FuzzySet(function=Triangular_MF(a=3, b=5, c=7), term="average"),
    FuzzySet(function=Trapezoidal_MF(a=6, b=8, c=10, d=10), term="good")
]))

# Wysokość napiwku (0-30%)
FS.add_linguistic_variable("tip", LinguisticVariable([
    FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=5, d=10), term="low"),
    FuzzySet(function=Triangular_MF(a=10, b=15, c=20), term="medium"),
    FuzzySet(function=Trapezoidal_MF(a=20, b=25, c=30, d=30), term="high")
]))

# Dodaj reguły
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

# Funkcja do rysowania funkcji przynależności
def plot_membership_functions(variable_name, title):
    var = FS.get_variable(variable_name)
    x = np.linspace(0, 10 if variable_name != "tip" else 30, 1000)
    
    plt.figure(figsize=(10, 4))
    for term, fuzzy_set in var.terms.items():
        y = [fuzzy_set.get_value(xi) for xi in x]
        plt.plot(x, y, label=term)
    
    plt.title(f"Funkcje przynależności dla {title}")
    plt.xlabel("Wartość")
    plt.ylabel("Przynależność")
    plt.legend()
    plt.grid(True)
    plt.show()

# Wyświetl wykresy funkcji przynależności
plot_membership_functions("food", "Jakość jedzenia")
plot_membership_functions("service", "Jakość obsługi")
plot_membership_functions("tip", "Wysokość napiwku")

# Przetestuj system na kilku przykładach
test_cases = [
    (3, 3),  # Średnie jedzenie, średnia obsługa
    (8, 8),  # Dobre jedzenie, dobra obsługa
    (2, 2),  # Słabe jedzenie, słaba obsługa
    (9, 4),  # Dobre jedzenie, średnia obsługa
    (4, 9),  # Średnie jedzenie, dobra obsługa
]

print("\nTesty systemu rozmytego:")
print("------------------------")
for food, service in test_cases:
    FS.set_variable("food", food)
    FS.set_variable("service", service)
    tip = FS.inference()["tip"]
    print(f"Jedzenie: {food:.1f}, Obsługa: {service:.1f} -> Napiwek: {tip:.1f}%") 