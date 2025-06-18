#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

import numpy as np
from datetime import datetime
from evaluation.recorder import CarRacingVideoRecorder

def record_selected_agents():
    """Nagraj tylko wybrane agenty: DQN_210ep, NEAT_best, PPO_carracing1, Random"""
    
    print("🎬 NAGRYWANIE WYBRANYCH AGENTÓW")
    print("=" * 50)
    
    # 1. DQN (ep210) - najlepszy wynik!
    print("\n🧪 Nagrywanie DQN (ep210)...")
    try:
        from agents.dqn_agent import DQNAgent
        from training.train_dqn import preprocess_state
        
        recorder = CarRacingVideoRecorder("dqn")
        
        # Sprawdź ścieżki modelu
        model_paths = [
            'models/dqn_model_ep210.keras',
            'checkpoints/dqn/dqn_model_ep210.keras',
            'models/dqn_model_210.keras'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            print(f"🔍 Znaleziono model: {model_path}")
            agent = DQNAgent.load(model_path, (84, 84, 1), 5)
            agent.epsilon = 0.0  # Tylko eksploatacja dla nagrania
            
            # POPRAWIONY wrapper dla DQN
            class DQNWrapper:
                def __init__(self, agent, preprocess_func):
                    self.agent = agent
                    self.preprocess_func = preprocess_func
                    self.last_state = None
                
                def act(self, observation):
                    """Obsłuż preprocessing w wrapper"""
                    # Jeśli obserwacja jest już przetworzonym stanem, użyj go bezpośrednio
                    if len(observation.shape) == 3 and observation.shape[2] == 1:
                        state = observation
                    else:
                        state = self.preprocess_func(observation)
                    
                    self.last_state = state
                    return self.agent.act(state)
            
            wrapped_agent = DQNWrapper(agent, preprocess_state)
            recorder.record_agent_episodes(wrapped_agent, episodes=3, max_steps=1000)
            print(f"✅ DQN nagrany! (średnia z testów: {844.63:.1f} pkt)")
        else:
            print(f"❌ Model DQN nie znaleziony w żadnej z lokalizacji:")
            for path in model_paths:
                print(f"   - {path}")
            
    except Exception as e:
        print(f"❌ Błąd DQN: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. NEAT (best) - dobry wynik
    print("\n🧠 Nagrywanie NEAT (best)...")
    try:
        from agents.neat_agent import NEATAgent
        
        recorder = CarRacingVideoRecorder("neat")
        
        # Sprawdź ścieżki modelu NEAT
        neat_paths = [
            # ('models/neat_best.pkl', 'configs/neat_config.txt'),
            # ('checkpoints/neat_best.pkl', 'configs/neat_config.txt'),
            # ('models/neat_winner.pkl', 'configs/neat_config.txt')
        ]
        
        agent = None
        for model_path, config_path in neat_paths:
            if os.path.exists(model_path) and os.path.exists(config_path):
                print(f"🔍 Próba wczytania: {model_path}")
                agent = NEATAgent.load_model(model_path, config_path)
                if agent is not None:
                    break
        
        if agent is not None:
            recorder.record_agent_episodes(agent, episodes=3, max_steps=1000)
            print(f"✅ NEAT nagrany! (średnia z testów: {455.67:.1f} pkt)")
        else:
            print("❌ Nie udało się załadować żadnego modelu NEAT")
            print("Sprawdzone ścieżki:")
            for model_path, config_path in neat_paths:
                print(f"   - {model_path} ({'✓' if os.path.exists(model_path) else '✗'})")
                print(f"   - {config_path} ({'✓' if os.path.exists(config_path) else '✗'})")
            
    except Exception as e:
        print(f"❌ Błąd NEAT: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. PPO (Stable Baselines3) - bardzo dobry
    print("\n🤖 Nagrywanie PPO (Stable Baselines3)...")
    try:
        from stable_baselines3 import PPO
        
        recorder = CarRacingVideoRecorder("ppo")
        
        # Sprawdź ścieżki modelu PPO
        ppo_paths = [
            # "models/ppo_carracing1",
            # "models/ppo_carracing",
            # "models/ppo_model"
        ]
        
        model = None
        model_path = None
        for path in ppo_paths:
            if os.path.exists(path + ".zip"):
                model_path = path
                print(f"🔍 Znaleziono model PPO: {path}.zip")
                model = PPO.load(path)
                break
        
        if model is not None:
            # Wrapper dla SB3 PPO
            class PPOWrapper:
                def __init__(self, model):
                    self.model = model
                
                def act(self, observation):
                    action, _ = self.model.predict(observation, deterministic=True)
                    return action
            
            wrapped_agent = PPOWrapper(model)
            recorder.record_agent_episodes(wrapped_agent, episodes=3, max_steps=1000)
            print(f"✅ PPO nagrany! (średnia z testów: {525.21:.1f} pkt)")
        else:
            print(f"❌ Model PPO nie znaleziony w żadnej z lokalizacji:")
            for path in ppo_paths:
                print(f"   - {path}.zip ({'✓' if os.path.exists(path + '.zip') else '✗'})")
            
    except Exception as e:
        print(f"❌ Błąd PPO: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Random (baseline)
    print("\n🎲 Nagrywanie Random (baseline)...")
    try:
        from agents.random_agent import RandomAgent
        
        recorder = CarRacingVideoRecorder("random")
        agent = RandomAgent()
        
        recorder.record_agent_episodes(agent, episodes=2, max_steps=500)  # Krótsze dla random
        print(f"✅ Random nagrany! (średnia z testów: {-60.06:.1f} pkt)")
        
    except Exception as e:
        print(f"❌ Błąd Random: {e}")
        import traceback
        traceback.print_exc()

# Reszta funkcji bez zmian...
def create_comparison_montage():
    """Stwórz montaż porównawczy BEZ TEKSTU (bez ImageMagick)"""
    print("\n🎞️ Tworzenie montażu porównawczego (bez tekstu)...")
    
    try:
        import moviepy.editor as mp
        
        agents = ['dqn', 'neat', 'ppo', 'random']
        clips = []
        
        for agent in agents:
            video_dir = f"results/videos"
            
            # Znajdź najnowsze video tego agenta
            import glob
            pattern = f"{video_dir}/{agent}_ep*_*.mp4"
            videos = glob.glob(pattern)
            
            if videos:
                # Weź najnowsze video
                latest_video = max(videos, key=os.path.getctime)
                print(f"   Dodaję {agent}: {latest_video}")
                
                # Załaduj i przytnij do 60 sekund
                try:
                    clip = mp.VideoFileClip(latest_video)
                    duration = min(60, clip.duration)  # Dłuższe segmenty bez tytułów
                    clip = clip.subclip(0, duration)
                    clips.append(clip)
                    
                except Exception as e:
                    print(f"   ❌ Błąd przetwarzania {agent}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠️ Nie znaleziono video dla {agent}")
        
        if clips:
            # Połącz wszystkie klipy bez dodatkowych efektów tekstowych
            print("   🔗 Łączenie klipów...")
            final_montage = mp.concatenate_videoclips(clips)
            
            # Zapisz
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/agents_comparison_simple_{timestamp}.mp4"
            
            print(f"   🎬 Renderowanie montażu...")
            final_montage.write_videofile(output_path, 
                                        fps=30, 
                                        codec='libx264')
            
            print(f"✅ Montaż zapisany: {output_path}")
            print(f"📋 Kolejność agentów: DQN (0-60s) → NEAT (60-120s) → PPO (120-180s) → Random (180-240s)")
            
            # Cleanup
            for clip in clips:
                clip.close()
            final_montage.close()
            
        else:
            print("❌ Brak klipów do montażu")
            
    except Exception as e:
        print(f"❌ Błąd tworzenia montażu: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Główna funkcja"""
    print("🎬 NAGRYWANIE NAJLEPSZYCH AGENTÓW")
    print("Na podstawie wyników z testów:")
    print("• DQN (ep210): 844.6 pkt średnio, 81% ukończeń 🏆")
    print("• PPO (SB3): 525.2 pkt średnio, 37% ukończeń 🤖")
    print("• NEAT: 455.7 pkt średnio, 19% ukończeń 🧠")
    print("• Random: -60.1 pkt średnio, 0% ukończeń 🎲")
    print("")
    
    # Utwórz katalogi
    os.makedirs('results/videos', exist_ok=True)
    
    # Nagraj agentów
    # record_selected_agents()
    
    # Stwórz montaż
    create_comparison_montage()
    
    print("\n✅ Nagrywanie zakończone!")
    print(f"📁 Sprawdź katalog results/videos/ i results/")

if __name__ == "__main__":
    main()