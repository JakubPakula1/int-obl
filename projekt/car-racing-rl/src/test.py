import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.car_racing_env import CarRacingEnv
from agents.neat_agent import NEATAgent

def demo_neat_model():
    """Demonstracja wytrenowanego modelu NEAT"""
    
    model_path = '/Users/kuba/Desktop/Studia UG/semestr_4/inteligencja_obliczeniowa/projekt/car-racing-rl/models/neat-checkpoint-10.pkl'
    config_path = '/Users/kuba/Desktop/Studia UG/semestr_4/inteligencja_obliczeniowa/projekt/car-racing-rl/configs/neat_config.txt' \

    
    # Sprawdź czy pliki istnieją
    if not os.path.exists(model_path):
        print(f"❌ Plik modelu nie istnieje: {model_path}")
        return
        
    if not os.path.exists(config_path):
        print(f"❌ Plik konfiguracji nie istnieje: {config_path}")
        return
    
    print("🤖 Ładowanie modelu NEAT...")
    
    # Utwórz agenta i załaduj model
    neat_agent = NEATAgent(config_path)
    neat_agent.load_model(model_path)
    
    if neat_agent.best_genome is None:
        print("❌ Nie udało się załadować modelu!")
        return
    
    print(f"✅ Model załadowany pomyślnie!")
    print(f"📊 Fitness modelu: {neat_agent.best_genome.fitness}")
    
    # Utwórz środowisko z wizualizacją
    print("\n🏁 Rozpoczynam demonstrację...")
    print("Naciśnij Enter aby rozpocząć...")
    input()
    
    env = CarRacingEnv(render_mode="human")
    
    # Uruchom demonstrację
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n=== 🏎️ Epizod {episode + 1}/{num_episodes} ===")
        
        observation = env.reset()
        total_reward = 0
        steps = 0
        negative_reward_streak = 0
        
        while steps < 1000:
            # Wybierz akcję przy użyciu wytrenowanego modelu
            action = neat_agent.act(observation)
            
            # Wykonaj krok
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Wyświetl postęp co 100 kroków
            if steps % 100 == 0:
                print(f"   Krok {steps}: Nagroda = {total_reward:.2f}")
            
            # Sprawdź czy episode się zakończył
            if terminated or truncated:
                break
                
            # Przerwij jeśli auto długo nie jedzie
            if reward < -0.1:
                negative_reward_streak += 1
                if negative_reward_streak > 100:
                    print("   Auto utknęło - przerywam epizod")
                    break
            else:
                negative_reward_streak = 0
        
        print(f"🏁 Epizod {episode + 1} zakończony:")
        print(f"   📈 Łączna nagroda: {total_reward:.2f}")
        print(f"   🔢 Liczba kroków: {steps}")
        print(f"   ⚡ Średnia nagroda na krok: {total_reward/steps:.3f}")
        
        # Ocena wyników
        if total_reward > 500:
            print("   🏆 Doskonały wynik!")
        elif total_reward > 200:
            print("   🥇 Bardzo dobry wynik!")
        elif total_reward > 0:
            print("   🥈 Dobry wynik!")
        else:
            print("   🥉 Wynik do poprawy")
        
        # Pauza między epizodami
        if episode < num_episodes - 1:
            print("\nNaciśnij Enter dla następnego epizodu...")
            input()
    
    env.close()
    print("\n✅ Demonstracja zakończona!")

if __name__ == "__main__":
    demo_neat_model()