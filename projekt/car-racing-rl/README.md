# Projekt Car Racing RL

Projekt Car Racing RL to implementacja algorytmów uczenia maszynowego w środowisku wyścigów samochodowych. Celem projektu jest stworzenie agentów, którzy będą w stanie uczyć się i optymalizować swoje działania w grze Car Racing v2 z biblioteki Gymnasium.

## Struktura projektu

- **src/**: Główna logika projektu.
  - **agents/**: Zawiera różne implementacje agentów.
    - `dqn_agent.py`: Implementacja algorytmu DQN.
    - `ppo_agent.py`: Implementacja algorytmu PPO.
    - `random_agent.py`: Agent losowy.
    - `rule_based_agent.py`: Agent oparty na regułach.
  - **environments/**: Definicje środowisk.
    - `car_racing_env.py`: Środowisko wyścigów samochodowych.
  - **models/**: Definicje modeli sieci neuronowych.
    - `neural_networks.py`: Zawiera definicje sieci dla agentów.
  - **training/**: Funkcje do trenowania agentów.
    - `train.py`: Zawiera funkcję do trenowania agentów.
  - **evaluation/**: Funkcje do oceny wydajności agentów.
    - `evaluate.py`: Zawiera funkcję do oceny agentów.
  - **utils/**: Funkcje pomocnicze.
    - `helpers.py`: Zawiera funkcje pomocnicze.
  - `main.py`: Plik główny uruchamiający projekt.

- **notebooks/**: Notatniki do eksploracji i analizy wyników.
  - `exploration.ipynb`: Notatnik do eksploracji danych.
  - `results_analysis.ipynb`: Notatnik do analizy wyników.

- **results/**: Wyniki eksperymentów.
  - `README.md`: Dokumentacja wyników.

- **configs/**: Pliki konfiguracyjne dla agentów.
  - `dqn_config.yaml`: Konfiguracja dla agenta DQN.
  - `ppo_config.yaml`: Konfiguracja dla agenta PPO.

- **requirements.txt**: Lista wymaganych bibliotek.

- **setup.py**: Plik konfiguracyjny do instalacji pakietu.

## Instrukcje uruchamiania

1. Zainstaluj wymagane biblioteki:
   ```
   pip install -r requirements.txt
   ```

2. Uruchom główny plik projektu:
   ```
   python src/main.py
   ```

3. Możesz również eksplorować dane i wyniki w notatnikach Jupyter.

## Wnioski

Projekt Car Racing RL ma na celu zbadanie różnych algorytmów uczenia maszynowego w kontekście wyścigów samochodowych. Dzięki zastosowaniu różnych agentów i metod treningowych, projekt dostarcza cennych informacji na temat efektywności różnych podejść w problemach sterowania.