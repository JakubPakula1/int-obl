from mazelib import Maze
from mazelib.generate.Prims import Prims  # Algorytm generowania labiryntu

class Labirynth:
    def __init__(self, width, height):
        self.x = 1
        self.y = 1  
        self.width = width
        self.height = height
        self.labirynth = []
        self.reached_goal = False
        self.generate_labirynth()

    def generate_labirynth(self):
        """Generuje labirynt za pomocą biblioteki mazelib."""
        # Tworzenie labiryntu za pomocą algorytmu Prims
        maze = Maze()
        maze.generator = Prims(self.width//2, self.height//2)
        maze.generate()

        # Konwersja labiryntu na macierz
        self.labirynth = []
        for row in maze.grid:
            self.labirynth.append(["#" if cell == 1 else " " for cell in row])

        # Ustawienie punktu startowego i końcowego
        self.labirynth[1][1] = "S"  # Start
        self.labirynth[self.height - 1][self.width - 1] = "E"  # End

    def display_labirynth(self):
        """Wyświetla labirynt w konsoli."""
        print("+" + "--" * self.width + "+")
        for i, row in enumerate(self.labirynth):
            line = "|"
            for j, cell in enumerate(row):
                if (i, j) == (self.x, self.y):
                    line += "P "
                elif cell == "*":
                    line += "\033[92m*\033[0m "  # Zielony kolor dla gwiazdek
                elif cell == " ":
                    line += "o "
                else:
                    line += cell + " "
            line += "|"
            print(line)
        print("+" + "--" * self.width + "+")

    def move(self, direction):
        """Przesuwa gracza w określonym kierunku, jeśli ruch jest możliwy."""
        if self.reached_goal:
            return False
        new_x, new_y = self.x, self.y
        if direction == 0:
            new_x -= 1
        elif direction == 1:
            new_x += 1
        elif direction == 2:
            new_y -= 1
        elif direction == 3:
            new_y += 1
        else:
            return False
        if 0 <= new_x < self.height and 0 <= new_y < self.width:
            if self.labirynth[new_x][new_y] != "#":
                if self.labirynth[self.x][self.y] == " ":
                    self.labirynth[self.x][self.y] = "*"
                self.x, self.y = new_x, new_y
                if self.labirynth[self.x][self.y] == "E":
                    self.reached_goal = True
                    print("Gratulacje! Dotarłeś do końca labiryntu!")
                return True
        return False

    def get_position(self):
        return (self.x, self.y)
    
    def reset_position(self):
        self.x, self.y = 1, 1
        self.reached_goal = False 

    def get_finish(self):
        return (self.height - 1, self.width - 1)
    
    def calculate_distance(self):
        end_x, end_y = self.get_finish()
        current_x, current_y = self.get_position()
        return abs(end_x - current_x) + abs(end_y - current_y)