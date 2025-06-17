# Refaktoryzacja klasy FuzzyAgent z poprawkami ograniczającymi kręcenie się w miejscu

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import cv2


class FuzzyAgent:
    def __init__(self):
        print("🧠 Inicjalizacja Fuzzy Logic Agent...")

        # Zmienne wejściowe
        self.track_position = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'track_position')
        self.track_angle = ctrl.Antecedent(np.arange(-90, 91, 1), 'track_angle')
        self.speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')
        self.distance_to_edge = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'distance_to_edge')

        # Zmienne wyjściowe
        self.steering = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'steering')
        self.throttle = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'throttle')
        self.brake = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'brake')

        self.steering.defuzzify_method = 'centroid'
        self.throttle.defuzzify_method = 'centroid'
        self.brake.defuzzify_method = 'centroid'

        self._define_membership_functions()
        self._define_rules()

        self.prev_steering = 0.0
        self.step_count = 0

        print("✅ Fuzzy Logic Agent utworzony pomyślnie")

    def _define_membership_functions(self):
        self.track_position['left'] = fuzz.trimf(self.track_position.universe, [-1, -0.6, -0.2])
        self.track_position['center'] = fuzz.trimf(self.track_position.universe, [-0.3, 0, 0.3])
        self.track_position['right'] = fuzz.trimf(self.track_position.universe, [0.2, 0.6, 1])

        self.track_angle['sharp_left'] = fuzz.trimf(self.track_angle.universe, [-90, -45, -20])
        self.track_angle['left'] = fuzz.trimf(self.track_angle.universe, [-30, -10, 0])
        self.track_angle['straight'] = fuzz.trimf(self.track_angle.universe, [-15, 0, 15])
        self.track_angle['right'] = fuzz.trimf(self.track_angle.universe, [0, 10, 30])
        self.track_angle['sharp_right'] = fuzz.trimf(self.track_angle.universe, [20, 45, 90])

        self.speed['slow'] = fuzz.trimf(self.speed.universe, [0, 20, 40])
        self.speed['medium'] = fuzz.trimf(self.speed.universe, [30, 50, 70])
        self.speed['fast'] = fuzz.trimf(self.speed.universe, [60, 80, 100])

        self.distance_to_edge['close'] = fuzz.trimf(self.distance_to_edge.universe, [0, 0.2, 0.4])
        self.distance_to_edge['medium'] = fuzz.trimf(self.distance_to_edge.universe, [0.3, 0.5, 0.7])
        self.distance_to_edge['far'] = fuzz.trimf(self.distance_to_edge.universe, [0.6, 0.8, 1])

        self.steering['hard_left'] = fuzz.trimf(self.steering.universe, [-1, -0.8, -0.5])
        self.steering['left'] = fuzz.trimf(self.steering.universe, [-0.6, -0.3, -0.05])
        self.steering['straight'] = fuzz.trimf(self.steering.universe, [-0.1, 0, 0.1])
        self.steering['right'] = fuzz.trimf(self.steering.universe, [0.05, 0.3, 0.6])
        self.steering['hard_right'] = fuzz.trimf(self.steering.universe, [0.5, 0.8, 1])

        self.throttle['none'] = fuzz.trimf(self.throttle.universe, [0, 0.1, 0.2])
        self.throttle['low'] = fuzz.trimf(self.throttle.universe, [0.15, 0.35, 0.55])
        self.throttle['medium'] = fuzz.trimf(self.throttle.universe, [0.45, 0.65, 0.85])
        self.throttle['full'] = fuzz.trimf(self.throttle.universe, [0.75, 0.9, 1])

        self.brake['none'] = fuzz.trimf(self.brake.universe, [0, 0, 0.05])
        self.brake['light'] = fuzz.trimf(self.brake.universe, [0.02, 0.15, 0.3])
        self.brake['heavy'] = fuzz.trimf(self.brake.universe, [0.25, 0.6, 1])

    def _define_rules(self):
        rules = [
            # Steering rules
            ctrl.Rule(self.track_position['left'] & self.track_angle['straight'], self.steering['left']),
            ctrl.Rule(self.track_position['right'] & self.track_angle['straight'], self.steering['right']),
            ctrl.Rule(self.track_position['center'], self.steering['straight']),
            ctrl.Rule(self.track_angle['sharp_left'], self.steering['left']),
            ctrl.Rule(self.track_angle['sharp_right'], self.steering['right']),
            ctrl.Rule(self.track_angle['left'], self.steering['left']),
            ctrl.Rule(self.track_angle['right'], self.steering['right']),
            ctrl.Rule(self.track_angle['straight'], self.steering['straight']),

            # Throttle rules
            ctrl.Rule(self.track_angle['straight'] & self.speed['slow'], self.throttle['full']),
            ctrl.Rule(self.track_angle['straight'] & self.speed['medium'], self.throttle['full']),
            ctrl.Rule(self.track_angle['straight'] & self.speed['fast'], self.throttle['medium']),
            ctrl.Rule(self.track_angle['sharp_left'], self.throttle['low']),
            ctrl.Rule(self.track_angle['sharp_right'], self.throttle['low']),
            ctrl.Rule(self.distance_to_edge['far'], self.throttle['full']),
            ctrl.Rule(self.distance_to_edge['medium'], self.throttle['medium']),
            ctrl.Rule(self.distance_to_edge['close'], self.throttle['low']),

            # Brake rules
            ctrl.Rule(self.track_angle['straight'], self.brake['none']),
            ctrl.Rule(self.track_angle['left'], self.brake['none']),
            ctrl.Rule(self.track_angle['right'], self.brake['none']),
            ctrl.Rule(self.speed['slow'], self.brake['none']),
            ctrl.Rule(self.track_angle['sharp_left'] & self.speed['fast'] & self.distance_to_edge['close'], self.brake['light']),
            ctrl.Rule(self.track_angle['sharp_right'] & self.speed['fast'] & self.distance_to_edge['close'], self.brake['light']),
        ]

        system = ctrl.ControlSystem(rules)
        self.controller = ctrl.ControlSystemSimulation(system)

    def extract_features(self, observation):
        """Wyciąga rzeczywiste cechy z obserwacji"""
        try:
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            
            # 1. Pozycja na torze - analiza dolnej części obrazu
            height, width = gray.shape
            bottom_section = gray[int(height*0.7):, :]
            
            # Znajdź jasne piksele (tor) w dolnej sekcji
            track_mask = (bottom_section > 0.4).astype(np.uint8)
            
            if np.sum(track_mask) > 100:
                # Znajdź środek toru
                y_coords, x_coords = np.where(track_mask)
                if len(x_coords) > 0:
                    track_center = np.mean(x_coords)
                    image_center = width / 2
                    track_position = (track_center - image_center) / (width / 2)
                else:
                    track_position = 0.0
            else:
                track_position = 0.0
            
            # 2. Kąt toru - analiza kierunku toru
            track_angle = self._detect_track_angle(gray)
            
            # 3. Szacowana prędkość na podstawie rozmycia
            speed = self._estimate_speed(gray)
            
            # 4. Odległość od krawędzi
            edge_distance = self._detect_edge_distance(gray)
            
            return {
                'track_position': np.clip(track_position, -1, 1),
                'track_angle': np.clip(track_angle, -90, 90),
                'speed': np.clip(speed, 0, 100),
                'distance_to_edge': np.clip(edge_distance, 0, 1)
            }
            
        except Exception as e:
            print(f"❌ Błąd wyciągania cech: {e}")
            return {
                'track_position': 0.0,
                'track_angle': 0.0,
                'speed': 30.0,
                'distance_to_edge': 0.8
            }
    
    def _detect_track_angle(self, gray_image):
        """Wykrywa kąt toru używając detekcji krawędzi"""
        try:
            # Użyj Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Hough lines do wykrycia linii
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None and len(lines) > 0:
                # Znajdź dominujący kąt
                angles = []
                for rho, theta in lines[:5]:  # Weź pierwsze 5 linii
                    angle = np.degrees(theta) - 90  # Konwertuj do zakresu -90 do 90
                    angles.append(angle)
                
                return np.mean(angles)
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _estimate_speed(self, gray_image):
        """Szacuje prędkość na podstawie rozmycia obrazu"""
        try:
            # Oblicz wariancję Laplacjana (miara ostrości)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            # Im mniejsza wariancja, tym większe rozmycie = większa prędkość
            # Mapuj na zakres 0-100
            speed = max(0, min(100, 100 - laplacian_var * 10))
            return speed
            
        except:
            return 30.0
    
    def _detect_edge_distance(self, gray_image):
        """Wykrywa odległość od krawędzi toru"""
        try:
            height, width = gray_image.shape
            center_x = width // 2
            
            # Sprawdź środkową linię obrazu
            center_line = gray_image[:, center_x]
            
            # Znajdź najciemniejsze piksele (trawa/krawędzie)
            dark_pixels = center_line < 0.3
            
            if np.any(dark_pixels):
                # Znajdź najbliższą ciemną krawędź
                dark_indices = np.where(dark_pixels)[0]
                closest_dark = min(dark_indices, key=lambda x: abs(x - height//2))
                distance = abs(closest_dark - height//2) / (height//2)
                return distance
            else:
                return 1.0  # Daleko od krawędzi
                
        except:
            return 0.8
    
    def act(self, observation):
        """Ulepszona wersja act()"""
        self.step_count += 1
        
        # Wyciągnij rzeczywiste cechy
        features = self.extract_features(observation)
        
        try:
            # Ustaw wejścia kontrolera
            self.controller.input['track_position'] = features['track_position']
            self.controller.input['track_angle'] = features['track_angle']
            self.controller.input['speed'] = features['speed']
            self.controller.input['distance_to_edge'] = features['distance_to_edge']
            
            self.controller.compute()
            
            steering = self.controller.output['steering']
            throttle = self.controller.output['throttle']
            brake = self.controller.output['brake']
            
            # Twoja doskonała logika anti-spin
            if features['speed'] < 10:
                steering *= 0.3
            if features['speed'] < 10 and abs(steering) > 0.5:
                throttle = min(throttle, 0.2)
            if features['speed'] < 2:
                throttle = max(throttle, 0.4)
            else:
                throttle = max(throttle, 0.2)
            if throttle > 0.3:
                brake = 0.0
            
            # Wygładzanie
            steering = 0.5 * steering + 0.5 * self.prev_steering
            self.prev_steering = steering
            steering = np.clip(steering, -0.8, 0.8)
            
            return np.array([steering, throttle, brake])
            
        except Exception as e:
            print(f"❗ Błąd kontrolera: {e}")
            return np.array([0.0, 0.5, 0.0])