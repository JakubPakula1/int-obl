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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nie istnieje: {checkpoint_path}")
    
    print(f"Kontynuowanie z checkpointu: {checkpoint_path}")
    
    try:
        # Wczytaj konfigurację
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path
        )
        
        # Wczytaj populację z checkpointu
        population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        
        # Stwórz agenta NEAT
        neat_agent = NEATAgent(config_path)
        neat_agent.population = population
        neat_agent.config = config
        
        # Funkcja ewaluacji
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                try:
                    genome.fitness = neat_agent.evaluate_genome(genome, config, env)
                    
                    if neat_agent.best_genome is None or genome.fitness > neat_agent.best_genome.fitness:
                        neat_agent.best_genome = genome
                        neat_agent.best_net = neat.nn.FeedForwardNetwork.create(genome, config)
                except Exception as e:
                    print(f"Błąd podczas ewaluacji genomu {genome_id}: {e}")
                    genome.fitness = -1000  # Kara za błąd
        
        # Kontynuuj trening
        winner = population.run(eval_genomes, additional_generations)
        neat_agent.best_genome = winner
        neat_agent.best_net = neat.nn.FeedForwardNetwork.create(winner, config)
        
        print(f"Trening z checkpointu zakończony. Najlepszy fitness: {winner.fitness}")
        
        return neat_agent, winner
        
    except Exception as e:
        print(f"Błąd podczas wczytywania checkpointu: {e}")
        raise