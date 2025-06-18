import os
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
class ProgressionRecorder:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.recording_episodes = [1, 25, 50, 100, 150, 200]  # Kluczowe epizody
        
    def record_progression_videos(self):
        """Nagrywa filmy progresji dla kluczowych epizodów"""
        for episode in self.recording_episodes:
            model_path = f"checkpoints/{self.agent_name}/{self.agent_name}_model_ep{episode}.keras"
            
            if os.path.exists(model_path):
                output_path = f"results/videos/{self.agent_name}_ep{episode:03d}.mp4"
                self.record_single_episode(model_path, output_path, episode)
                
    def record_single_episode(self, model_path, output_path, episode):
        """Nagrywa pojedynczy epizod"""
        import cv2
        
        # Konfiguracja nagrywania
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (600, 400))
        
        # Uruchom agenta z wizualizacją
        cmd = f"python src/main.py --agent {self.agent_name} --mode test --model {model_path} --episodes 1 --render"
        # Implementacja nagrywania...
        
    def create_progression_montage(self):
        """Tworzy montaż pokazujący progresję"""
        # Połącz wszystkie filmy w jeden montaż
        import moviepy.editor as mp
        
        clips = []
        for episode in self.recording_episodes:
            video_path = f"results/videos/{self.agent_name}_ep{episode:03d}.mp4"
            if os.path.exists(video_path):
                clip = mp.VideoFileClip(video_path).subclip(0, 30)  # 30 sekund z każdego
                # Dodaj tekst z numerem epizodu
                text_clip = mp.TextClip(f"Epizod {episode}", fontsize=30, color='white')
                text_clip = text_clip.set_position(('left', 'top')).set_duration(clip.duration)
                final_clip = mp.CompositeVideoClip([clip, text_clip])
                clips.append(final_clip)
        
        # Połącz wszystkie klipy
        final_montage = mp.concatenate_videoclips(clips)
        final_montage.write_videofile(f"results/{self.agent_name}_progression_montage.mp4")