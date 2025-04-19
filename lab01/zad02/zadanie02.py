import random
import math
import matplotlib.pyplot as plt
import numpy as np
INITAL_HEIGHT = 100
INITIAL_VELOCITY = 50

def pick_target():
    return random.randint(50,340)

def check_hit(target, distance):
    return distance >= target-5 and distance <= target+5

def calculate_distance(angle_degrees):
    angle_rad = math.radians(angle_degrees)

    v0 = INITIAL_VELOCITY
    h0 = INITAL_HEIGHT    
    g = 9.81             
    
    
    sin_alpha = math.sin(angle_rad)
    cos_alpha = math.cos(angle_rad)
    
    sqrt_term = math.sqrt(v0**2 * sin_alpha**2 + 2*g*h0)
    
    distance = (v0 * sin_alpha + sqrt_term) * (v0 * cos_alpha) / g
    
    return round(distance)

def plot_trajectory(angle_degrees, distance, target):
    angle_rad = math.radians(angle_degrees)
    v0 = INITIAL_VELOCITY
    h0 = INITAL_HEIGHT
    g = 9.81

    x = np.linspace(0, distance, 100)
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    t = x / vx
    y = vy*t - (g*t**2)/2 + h0
    max_height = max(y)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Tor lotu')
    plt.plot([target, target], [0, h0], 'r--', label='Cel')
    plt.plot([0], [h0], 'go', label='Trebusz')
    
    plt.grid(True)
    plt.xlabel('Odległość [m]')
    plt.ylabel('Wysokość [m]')
    plt.title(f'Tor lotu pocisku (kąt: {angle_degrees}°)')
    plt.legend()
    plt.axis([0, max(distance, target) + 50, 0, max_height + 50])
    plt.savefig('trajektoria.png', dpi=300,bbox_inches='tight')
    plt.show()

def main():
    target = pick_target()
    game = True
    tries = 0
    print(f"Sprobuj trafic w cel z trebuszu w jak najmiejszej ilosci prób!\n Twój cel znajduje sie w odległosci {target}m")
    
    while game:
        tries+=1
        angle = int(input("Podaj pod jakim kątem chcesz wystrzelic pocisk: "))

        distance = calculate_distance(angle)
        if check_hit(target, distance):
            game=False
        else:
            print(f"Niestety nie udało się twój pocisk poleciał na odległość {distance}")
    plot_trajectory(angle, distance, target)
    print(f"Poscisk poleciał na {distance}m")
    print(f"Brawo trafiłeś cel w {tries} próbach!")

if __name__ == "__main__":
    main()