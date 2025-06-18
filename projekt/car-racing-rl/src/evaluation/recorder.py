import os
import sys
import numpy as np
import cv2
import gymnasium as gym
from datetime import datetime
import subprocess
import threading
import time
from environments.lap_completion_fix_wrapper import LapCompletionFixWrapper

class CarRacingVideoRecorder:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.recording_episodes = [1, 25, 50, 100, 150, 200]
        self.video_dir = "results/videos"
        os.makedirs(self.video_dir, exist_ok=True)
        
    def record_agent_episodes(self, agent, env_config=None, episodes=3, max_steps=1000):
        """Nagraj kilka epizodów konkretnego agenta"""
        print(f"🎬 Nagrywanie {episodes} epizodów agenta {self.agent_name}")
        
        # Konfiguracja środowiska z nagrywaniem
        if env_config is None:
            env_config = {"render_mode": "rgb_array"}
        
        # Utwórz środowisko
        if self.agent_name == 'dqn':
            from environments.car_racing_env import CarRacingEnv
            env = CarRacingEnv(render_mode="rgb_array", continuous=False)
            env = LapCompletionFixWrapper(env)
        elif self.agent_name == 'ppo':
            import gymnasium as gym
            env = gym.make("CarRacing-v3", render_mode="rgb_array")
            env = LapCompletionFixWrapper(env)
        else:
            from environments.car_racing_env import CarRacingEnv
            env = CarRacingEnv(render_mode="rgb_array", continuous=True)
            env = LapCompletionFixWrapper(env)

        for episode in range(episodes):
            print(f"📹 Nagrywanie epizodu {episode + 1}/{episodes}")
            
            # Nazwa pliku
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.video_dir}/{self.agent_name}_ep{episode+1}_{timestamp}.mp4"
            
            # Nagraj epizod
            self.record_single_episode_direct(agent, env, output_path, max_steps)
        
        env.close()
        print(f"✅ Nagrywanie {self.agent_name} zakończone!")

    def record_single_episode_direct(self, agent, env, output_path, max_steps=1000):
        """Nagraj pojedynczy epizod bezpośrednio - POPRAWIONY"""
        # Reset środowiska
        observation, info = env.reset()
        
        # Pierwszy frame dla konfiguracji video writera
        frame = env.render()
        if frame is None:
            print("❌ Nie można uzyskać pierwszej klatki")
            return
            
        # POPRAWKA: Sprawdź format frame
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:  # Single channel
            frame = np.repeat(frame, 3, axis=2)
        
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        if not video_writer.isOpened():
            print(f"❌ Nie można otworzyć video writera: {output_path}")
            return
        
        episode_reward = 0
        step = 0
        
        try:
            for step in range(max_steps):
                # Pobierz akcję od agenta
                if self.agent_name == 'dqn':
                    action = agent.act(observation)  # DQN ma już wrapper z preprocessingiem
                elif self.agent_name == 'fuzzy':
                    action = agent.act(observation)
                elif self.agent_name == 'genetic':
                    action = agent.act(observation, step)
                elif self.agent_name == 'pso':
                    action = agent.act(observation, step)
                else:
                    action = agent.act(observation)
                
                # Wykonaj krok
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Renderuj i zapisz klatkę
                frame = env.render()
                if frame is not None:
                    # POPRAWKA: Obsługuj różne formaty klatek
                    processed_frame = self.process_frame_for_video(frame, step, episode_reward, action, info)
                    if processed_frame is not None:
                        # Konwertuj RGB na BGR dla OpenCV
                        bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr_frame)
                
                if terminated or truncated:
                    break
            
            video_writer.release()
            print(f"💾 Video zapisane: {output_path}")
            print(f"   📊 Epizod: {step+1} kroków, {episode_reward:.2f} pkt")
            
        except Exception as e:
            print(f"❌ Błąd nagrywania: {e}")
            video_writer.release()

    def process_frame_for_video(self, frame, step, reward, action, info):
        """NOWA METODA: Przetwarza klatkę do odpowiedniego formatu"""
        try:
            # Sprawdź i napraw format klatki
            if frame is None:
                return None
            
            # Konwertuj do RGB jeśli potrzeba
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif frame.shape[2] == 1:  # Single channel jako 3D array
                    frame = np.repeat(frame, 3, axis=2)
                # frame.shape[2] == 3 (RGB) - już OK
            
            # Dodaj informacje na klatkę
            frame_with_info = self.add_info_to_frame(frame, step, reward, action, info)
            
            return frame_with_info
            
        except Exception as e:
            print(f"❌ Błąd przetwarzania klatki: {e}")
            return None

    def add_info_to_frame(self, frame, step, reward, action, info):
        """Dodaj informacje na klatkę - POPRAWIONA Z OBSŁUGĄ RÓŻNYCH TYPÓW AKCJI"""
        try:
            frame_copy = frame.copy()
            
            # Upewnij się że frame jest w formacie uint8
            if frame_copy.dtype != np.uint8:
                frame_copy = (frame_copy * 255).astype(np.uint8)
            
            # Konfiguracja tekstu
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)  # Biały w RGB
            thickness = 2
            
            # Informacje do wyświetlenia
            texts = [
                f"Agent: {self.agent_name.upper()}",
                f"Krok: {step}",
                f"Nagroda: {reward:.2f}",
            ]
            
            # POPRAWKA: Obsługa różnych typów akcji
            if action is not None:
                if isinstance(action, (int, np.integer)):
                    # DQN zwraca pojedynczą liczbę (dyskretną akcję)
                    texts.append(f"Akcja: {action}")
                elif hasattr(action, '__len__') and len(action) >= 3:
                    # Ciągłe akcje [steering, gas, brake]
                    texts.append(f"Akcja: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]")
                elif hasattr(action, '__len__'):
                    # Jakakolwiek inna lista/tablica
                    action_str = ", ".join([f"{a:.2f}" if isinstance(a, (float, np.floating)) else str(a) for a in action])
                    texts.append(f"Akcja: [{action_str}]")
                else:
                    # Pojedyncza wartość (ale nie int)
                    texts.append(f"Akcja: {action:.2f}" if isinstance(action, (float, np.floating)) else f"Akcja: {action}")
            
            # Dodaj informacje ze środowiska
            if info and 'tiles_visited' in info:
                texts.append(f"Kafelki: {info['tiles_visited']}")
            
            # Rysuj tekst
            y_offset = 30
            for text in texts:
                # Dodaj czarne tło pod tekstem dla lepszej czytelności
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                cv2.rectangle(frame_copy, (5, y_offset - 25), 
                            (15 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                
                cv2.putText(frame_copy, text, (10, y_offset), 
                           font, font_scale, color, thickness)
                y_offset += 30
            
            return frame_copy
            
        except Exception as e:
            print(f"❌ Błąd dodawania tekstu: {e}")
            return frame