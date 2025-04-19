from datetime import datetime
import math

def oblicz_biorytmy(dni):
    """Oblicza wartości trzech biorytmów dla podanej liczby dni."""
    fizyczny = math.sin(2 * math.pi * dni / 23)
    emocjonalny = math.sin(2 * math.pi * dni / 28)
    intelektualny = math.sin(2 * math.pi * dni / 33)
    return fizyczny, emocjonalny, intelektualny

def oblicz_dni_zycia(rok, miesiac, dzien):
    """Oblicza liczbę dni od daty urodzenia do dziś."""
    data_urodzenia = datetime(rok, miesiac, dzien)
    dzis = datetime.now()
    roznica = dzis - data_urodzenia
    return roznica.days

# Pobieranie danych od użytkownika
imie = input("Podaj swoje imię: ")
rok = int(input("Podaj rok urodzenia (RRRR): "))
miesiac = int(input("Podaj miesiąc urodzenia (1-12): "))
dzien = int(input("Podaj dzień urodzenia (1-31): "))

# Obliczenia
dni_zycia = oblicz_dni_zycia(rok, miesiac, dzien)
fizyczny, emocjonalny, intelektualny = oblicz_biorytmy(dni_zycia)

# Wyświetlanie wyników
print(f"\nWitaj {imie}!")
print(f"Dziś jest {dni_zycia}. dzień Twojego życia.")
print("\nTwoje biorytmy na dziś:")
print(f"Fizyczny: {fizyczny:.3f}")
print(f"Emocjonalny: {emocjonalny:.3f}")
print(f"Intelektualny: {intelektualny:.3f}")

# Część b) - analiza wyników i przewidywania
for nazwa, wartosc in [("Fizyczny", fizyczny), 
                       ("Emocjonalny", emocjonalny), 
                       ("Intelektualny", intelektualny)]:
    if wartosc > 0.5:
        print(f"\n{nazwa}: Gratulacje! Twój biorytm jest wysoki!")
    elif wartosc < -0.5:
        # Sprawdzenie wartości na następny dzień
        _, nastepny_dzien, _ = oblicz_biorytmy(dni_zycia + 1)
        if nastepny_dzien > wartosc:
            print(f"\n{nazwa}: Dziś nie jest najlepszy dzień, ale nie martw się - jutro będzie lepiej!")
        else:
            print(f"\n{nazwa}: Dziś nie jest najlepszy dzień. Może warto odpocząć?")

# Wystarczył jeden prompt