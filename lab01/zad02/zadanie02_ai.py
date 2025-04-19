import random
import math
import matplotlib.pyplot as plt
import numpy as np

class Trebuchet:
    def __init__(self, height=100, velocity=50):
        self.height = height
        self.velocity = velocity
        self.g = 9.81
        
    def calculate_trajectory(self, angle_degrees, time_points=100):
        """Oblicza punkty trajektorii dla danego kąta."""
        angle_rad = math.radians(angle_degrees)
        vx = self.velocity * math.cos(angle_rad)
        vy = self.velocity * math.sin(angle_rad)
        
        # Oblicz czas lotu
        t_flight = (vy + math.sqrt(vy**2 + 2*self.g*self.height)) / self.g
        t = np.linspace(0, t_flight, time_points)
        
        x = vx * t
        y = self.height + vy*t - 0.5*self.g*t**2
        
        return x, y
        
    def calculate_distance(self, angle_degrees):
        """Oblicza odległość lotu pocisku."""
        x, _ = self.calculate_trajectory(angle_degrees)
        return round(float(x[-1]))

class TrebuchetGame:
    def __init__(self, min_distance=50, max_distance=340):
        self.trebuchet = Trebuchet()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.target = None
        self.tries = 0
        
    def new_target(self):
        """Losuje nowy cel."""
        self.target = random.randint(self.min_distance, self.max_distance)
        self.tries = 0
        
    def check_hit(self, distance):
        """Sprawdza czy trafiono w cel."""
        return abs(distance - self.target) <= 5
        
    def plot_shot(self, angle_degrees):
        """Rysuje wykres trajektorii strzału."""
        distance = self.trebuchet.calculate_distance(angle_degrees)
        x, y = self.trebuchet.calculate_trajectory(angle_degrees)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='Tor lotu')
        plt.plot([self.target, self.target], [0, self.trebuchet.height], 
                'r--', label='Cel')
        plt.plot(0, self.trebuchet.height, 'go', label='Trebusz')
        
        plt.grid(True)
        plt.xlabel('Odległość [m]')
        plt.ylabel('Wysokość [m]')
        plt.title(f'Tor lotu pocisku (kąt: {angle_degrees}°)')
        plt.legend()
        plt.axis([0, max(distance, self.target) + 50, 0, max(y) + 20])
        plt.savefig('trajektoria.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def play(self):
        """Główna pętla gry."""
        self.new_target()
        print(f"Spróbuj trafić w cel z trebusza!\nCel znajduje się w odległości {self.target}m")
        
        while True:
            self.tries += 1
            try:
                angle = int(input("Podaj kąt strzału (w stopniach): "))
                if not 0 <= angle <= 90:
                    raise ValueError("Kąt musi być między 0 a 90 stopni!")
            except ValueError as e:
                print(f"Błąd: {e}")
                continue
                
            distance = self.trebuchet.calculate_distance(angle)
            
            if self.check_hit(distance):
                print(f"\nBrawo! Trafiłeś cel w {self.tries} próbach!")
                self.plot_shot(angle)
                break
            else:
                print(f"Nie trafiłeś! Twój pocisk poleciał na odległość {distance}m")

if __name__ == "__main__":
    game = TrebuchetGame()
    game.play()