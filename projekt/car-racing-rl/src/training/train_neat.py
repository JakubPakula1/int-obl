from agents.neat_agent import NEATAgent
import os
import neat
import pickle

def train_neat(env, generations=50, config_path='configs/neat_config.txt'):
    """
    Trenowanie agenta NEAT
    
    Args:
        env: Środowisko do treningu
        generations: Liczba pokoleń
        config_path: Ścieżka do konfiguracji NEAT
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Plik konfiguracji nie istnieje: {config_path}")
    
    print("Rozpoczynanie treningu NEAT...")
    
    # Stwórz katalog na checkpointy
    os.makedirs('checkpoints', exist_ok=True)
    
    # Inicjalizuj agenta NEAT
    neat_agent = NEATAgent(config_path)
    
    # Dodaj checkpointer z zapisywaniem co 1 generację
    checkpointer = neat.Checkpointer(generation_interval=1, filename_prefix='checkpoints/neat-checkpoint-')
    neat_agent.population.add_reporter(checkpointer)
    
    # Trenuj
    winner = neat_agent.train(env, generations)
    
    # Zapisz najlepszy model
    os.makedirs('models', exist_ok=True)
    neat_agent.save_model('models/neat_best.pkl')
    
    print(f"Trening zakończony. Najlepszy fitness: {winner.fitness}")
    
    return neat_agent, winner

def continue_training(model_path, env, additional_generations=10, config_path='configs/neat_config.txt'):
    """
    Kontynuuj trening z zapisanego modelu
    
    Args:
        model_path: Ścieżka do zapisanego modelu
        env: Środowisko do treningu
        additional_generations: Dodatkowe generacje
        config_path: Ścieżka do konfiguracji NEAT
    """
    if not os.path.exists(model_path):
        print(f"Model nie istnieje: {model_path}. Rozpoczynanie nowego treningu...")
        return train_neat(env, additional_generations, config_path)
    
    print(f"Kontynuowanie treningu z modelu: {model_path}")
    
    # Inicjalizuj agenta NEAT
    neat_agent = NEATAgent(config_path)
    
    # Wczytaj zapisany model
    try:
        neat_agent.load_model(model_path)
    except Exception as e:
        print(f"Błąd wczytywania modelu: {e}")
        print("Rozpoczynanie nowego treningu...")
        return train_neat(env, additional_generations, config_path)
    
    if neat_agent.best_genome is None:
        print("Nie można wczytać modelu. Rozpoczynanie nowego treningu...")
        return train_neat(env, additional_generations, config_path)
    
    print(f"Wczytano model z fitness: {neat_agent.best_genome.fitness}")
    
    # Kontynuuj trening
    winner = neat_agent.train(env, additional_generations)
    
    # Zapisz zaktualizowany model z timestampem
    import time
    timestamp = int(time.time())
    model_name = f'models/neat_best_{additional_generations}_{timestamp}.pkl'
    neat_agent.save_model(model_name)
    
    print(f"Dodatkowy trening zakończony. Najlepszy fitness: {winner.fitness}")
    print(f"Model zapisany jako: {model_name}")
    
    return neat_agent, winner

def save_neat_checkpoint(neat_agent, generation, checkpoint_dir='checkpoints'):
    """Zapisz checkpoint populacji NEAT"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f'{checkpoint_dir}/neat-checkpoint-{generation}'
    
    if hasattr(neat_agent, 'population') and neat_agent.population:
        neat_agent.population.save_checkpoint(checkpoint_path)
        print(f"Checkpoint zapisany: {checkpoint_path}")
    else:
        print("Brak populacji do zapisania")
        
    return checkpoint_path

def continue_from_checkpoint(checkpoint_path, env, additional_generations=10, config_path='configs/neat_config.txt'):
    """Kontynuuj trening z checkpointu populacji"""
    import os  # ← DODAJ TEN IMPORT
    import pickle  # ← DODAJ TEN IMPORT (używany w linii 185)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nie istnieje: {checkpoint_path}")
    
    print(f"🔄 Kontynuowanie z checkpointu: {checkpoint_path}")
    
    try:
        print("📁 Wczytywanie konfiguracji NEAT...")
        # Wczytaj konfigurację
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path
        )
        print(f"✅ Konfiguracja wczytana: pop_size={config.pop_size}, threshold={config.fitness_threshold}")
        
        print("💾 Przywracanie populacji z checkpointu...")
        # Wczytaj populację z checkpointu
        population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        print(f"✅ Populacja przywrócona: {len(population.population)} osobników")
        
        print("🧬 Tworzenie agenta NEAT...")
        # Stwórz agenta NEAT
        neat_agent = NEATAgent(config_path)
        neat_agent.population = population
        neat_agent.config = config
        print("✅ Agent NEAT utworzony i skonfigurowany")
        
        # Dodaj reportery do populacji
        print("📊 Dodawanie reporterów...")
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Dodaj checkpointer dla dalszego treningu
        checkpointer = neat.Checkpointer(generation_interval=1,
                                       time_interval_seconds=None,
                                       filename_prefix='checkpoints/neat-checkpoint-')
        population.add_reporter(checkpointer)
        print("✅ Reportery dodane")
        
        # Funkcja ewaluacji
        def eval_genomes(genomes, config):
            print(f"🔬 Rozpoczynam ewaluację {len(genomes)} genomów...")
            evaluated = 0
            
            for genome_id, genome in genomes:
                try:
                    genome.fitness = neat_agent.evaluate_genome(genome, config, env)
                    evaluated += 1
                    
                    # Progress co 20 genomów
                    if evaluated % 20 == 0:
                        print(f"  📈 Oceniono {evaluated}/{len(genomes)} genomów")
                    
                    if neat_agent.best_genome is None or genome.fitness > neat_agent.best_genome.fitness:
                        neat_agent.best_genome = genome
                        neat_agent.best_net = neat.nn.FeedForwardNetwork.create(genome, config)
                        print(f"  🌟 Nowy najlepszy genom {genome_id}: fitness={genome.fitness:.2f}")
                        
                except Exception as e:
                    print(f"❌ Błąd podczas ewaluacji genomu {genome_id}: {e}")
                    genome.fitness = -1000  # Kara za błąd
            
            print(f"✅ Ewaluacja zakończona: {evaluated} genomów ocenionych")
        
        print(f"🚀 Rozpoczynanie dodatkowego treningu na {additional_generations} generacji...")
        
        # Kontynuuj trening
        winner = population.run(eval_genomes, additional_generations)
        
        print("🏆 Trening zakończony - przetwarzanie wyników...")
        neat_agent.best_genome = winner
        neat_agent.best_net = neat.nn.FeedForwardNetwork.create(winner, config)
        
        # Zapisz najlepszy model
        print("💾 Zapisywanie najlepszego modelu...")
        os.makedirs('models', exist_ok=True)
        with open('models/neat_best.pkl', 'wb') as f:
            data = {
                'genome': winner,
                'config_path': config_path,
                'fitness': winner.fitness
            }
            pickle.dump(data, f)
        print("✅ Model zapisany: models/neat_best.pkl")
        
        print(f"🎯 Trening z checkpointu zakończony!")
        print(f"📊 Najlepsza fitness: {winner.fitness:.2f}")
        print(f"🧬 Rozmiar sieci: {winner.size()}")
        
        return neat_agent, winner
        
    except FileNotFoundError as e:
        print(f"❌ Nie znaleziono pliku: {e}")
        raise
    except neat.config.ConfigParameter as e:
        print(f"❌ Błąd konfiguracji NEAT: {e}")
        print("💡 Sprawdź czy neat_config.txt ma poprawną składnię")
        raise
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd podczas wczytywania checkpointu: {e}")
        print(f"🔍 Typ błędu: {type(e).__name__}")
        print(f"📍 Checkpoint: {checkpoint_path}")
        print(f"📍 Konfiguracja: {config_path}")
        raise