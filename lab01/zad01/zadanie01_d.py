from datetime import datetime
import math

def get_user_input():
    """Pobiera dane od użytkownika"""
    return {
        'name': input("Podaj imię: "),
        'year': int(input("Podaj rok urodzenia: ")),
        'month': int(input("Podaj miesiąc urodzenia (1-12): ")),
        'day': int(input("Podaj dzień urodzenia: "))
    }

def calculate_days_from_birth(birth_data):
    """Oblicza liczbę dni od daty urodzenia"""
    birth_date = datetime(birth_data['year'], birth_data['month'], birth_data['day'])
    current_date = datetime.now()
    return (current_date - birth_date).days

def calculate_wave(days, period):
    """Oblicza wartość fali biorutmu"""
    return math.sin((2 * math.pi * days) / period)

def check_wave(wave_value, wave_name, days, period):
    """Sprawdza i wyświetla stan biorutmu"""
    print(f"{wave_name}: {wave_value:.3f}")
    
    if wave_value > 0.5:
        print("Świetny wynik!")
    elif wave_value < -0.5:
        print("Nie martw się!")
        next_day_value = calculate_wave(days + 1, period)
        if next_day_value > wave_value:
            print("Jutro będzie lepiej!")

def main():
    """Główna funkcja programu"""
    # Stałe dla okresów biorytmów
    PHYSICAL_PERIOD = 23
    EMOTIONAL_PERIOD = 28
    INTELLECTUAL_PERIOD = 33
    
    # Pobierz dane użytkownika
    user_data = get_user_input()
    days = calculate_days_from_birth(user_data)
    
    # Oblicz biorytmy
    rhythms = {
        "Fizyczny": (calculate_wave(days, PHYSICAL_PERIOD), PHYSICAL_PERIOD),
        "Emocjonalny": (calculate_wave(days, EMOTIONAL_PERIOD), EMOTIONAL_PERIOD),
        "Intelektualny": (calculate_wave(days, INTELLECTUAL_PERIOD), INTELLECTUAL_PERIOD)
    }
    
    # Wyświetl wyniki
    print(f"\nWitaj {user_data['name']}! Żyjesz już na ziemi {days} dni!")
    print("Oto wyniki twoich biorytmów:")
    
    for name, (value, period) in rhythms.items():
        check_wave(value, name, days, period)

if __name__ == "__main__":
    main()