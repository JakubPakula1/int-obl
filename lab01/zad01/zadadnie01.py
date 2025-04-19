from datetime import datetime
import math

name = input("Podaj imie: ")
year = input("Podaj rok urodzenia: ")
month = input("Podaj miesiąc urodzenia(cyfry): ")
day = input("Podaj dzien urodzenia: ")

def calculate_days_from_birth():
    birth_date = datetime(int(year), int(month), int(day))

    current_date = datetime.now()
    days_lived = (current_date - birth_date).days

    return days_lived

def calculate_physical_wave(days):
    return math.sin(((2*math.pi)/23)*days)

def calculate_emotional_wave(days):
    return math.sin(((2*math.pi)/28)*days)

def calculate_intelectual_wave(days):
    return math.sin(((2*math.pi)/33)*days)

def check_wave(wave_value, wave_name, days):
    print(f"{wave_name}: {wave_value}")
    if wave_value > 0.5:
        print("Świetny wynik!")
    elif wave_value < -0.5:
        print("Nie martw się!")
        next_day_value = None
        if wave_name == "Fizyczny":
            next_day_value = calculate_physical_wave(days+1)
        elif wave_name == "Emocjonalny":
            next_day_value = calculate_emotional_wave(days+1)
        elif wave_name == "Intelektualny":
            next_day_value = calculate_intelectual_wave(days+1)
            
        if next_day_value > wave_value:
            print("Jutro będzie lepiej!")

days = calculate_days_from_birth()
physical_wave = calculate_physical_wave(days)
emotional_wave = calculate_emotional_wave(days)
intelectual_wave = calculate_intelectual_wave(days)

print(f"Witaj {name}! Zyjesz juz na ziemi {days}!")
print("Oto wyniki twoich biorytmów: ")
check_wave(physical_wave, "Fizyczny", days)
check_wave(emotional_wave, "Emocjonalny", days)
check_wave(intelectual_wave, "Intelektualny", days)

# Bez narzedzi ok.25min