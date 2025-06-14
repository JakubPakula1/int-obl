PYTHON = python
SRC_DIR = src

# Zmienne dla poszczególnych agentów
DQN = --agent dqn
NEAT = --agent neat
RANDOM = --agent random

# Tryby działania
TRAIN = --mode train
TEST = --mode test

# Dodatkowe opcje
RENDER = --render
EPISODES_10 = --episodes 10
EPISODES_50 = --episodes 50
EPISODES_100 = --episodes 100
CONTINUE = --continue_training

# ========== TRENOWANIE ==========
# Trenowanie DQN
train-dqn:
	$(PYTHON) $(SRC_DIR)/main.py $(DQN) $(TRAIN) $(EPISODES_50)

# Trenowanie DQN z wizualizacją
train-dqn-visual:
	$(PYTHON) $(SRC_DIR)/main.py $(DQN) $(TRAIN) $(EPISODES_50) $(RENDER)

# Trenowanie NEAT
train-neat:
	$(PYTHON) $(SRC_DIR)/main.py $(NEAT) $(TRAIN) $(EPISODES_20)

# Trenowanie NEAT z wizualizacją
train-neat-visual:
	$(PYTHON) $(SRC_DIR)/main.py $(NEAT) $(TRAIN) $(EPISODES_20) $(RENDER)

# ========== KONTYNUACJA TRENINGU ==========
# Kontynuacja treningu DQN
continue-dqn:
	$(PYTHON) $(SRC_DIR)/main.py $(DQN) $(TRAIN) $(CONTINUE) $(EPISODES_50)

# Kontynuacja treningu DQN z wizualizacją
continue-dqn-visual:
	$(PYTHON) $(SRC_DIR)/main.py $(DQN) $(TRAIN) $(CONTINUE) $(EPISODES_50) $(RENDER)

# Kontynuacja treningu NEAT
continue-neat:
	$(PYTHON) $(SRC_DIR)/main.py $(NEAT) $(TRAIN) $(CONTINUE) $(EPISODES_20)

# ========== TESTOWANIE ==========
# Testowanie najnowszego modelu DQN
test-dqn:
	$(PYTHON) $(SRC_DIR)/main.py $(DQN) $(TEST) $(EPISODES_10)

# Testowanie najnowszego modelu NEAT
test-neat:
	$(PYTHON) $(SRC_DIR)/main.py $(NEAT) $(TEST) $(EPISODES_10)

# Testowanie agenta losowego
test-random:
	$(PYTHON) $(SRC_DIR)/main.py $(RANDOM) $(TEST) $(EPISODES_10)

# ========== TESTOWANIE KONKRETNEGO MODELU ==========
# Testowanie konkretnego modelu DQN
test-dqn-model:
	$(PYTHON) $(SRC_DIR)/main.py $(DQN) $(TEST) --model $(MODEL) $(EPISODES_10)

# Testowanie konkretnego modelu NEAT
test-neat-model:
	$(PYTHON) $(SRC_DIR)/main.py $(NEAT) $(TEST) --model $(MODEL) $(EPISODES_10)

# ========== POMOCNICZE ==========
clean:
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(SRC_DIR)/*/__pycache__

help:
	@echo "Dostępne komendy:"
	@echo "  make train-dqn            - Trenowanie agenta DQN"
	@echo "  make train-dqn-visual     - Trenowanie agenta DQN z wizualizacją"
	@echo "  make train-neat           - Trenowanie agenta NEAT"
	@echo "  make train-neat-visual    - Trenowanie agenta NEAT z wizualizacją"
	@echo "  make continue-dqn         - Kontynuacja treningu DQN"
	@echo "  make continue-dqn-visual  - Kontynuacja treningu DQN z wizualizacją"
	@echo "  make continue-neat        - Kontynuacja treningu NEAT"
	@echo "  make test-dqn             - Testowanie najnowszego modelu DQN"
	@echo "  make test-neat            - Testowanie najnowszego modelu NEAT"
	@echo "  make test-random          - Testowanie agenta losowego"
	@echo "  make test-dqn-model MODEL=ścieżka/do/modelu.h5  - Testowanie	konkretnego modelu DQN"
	@echo "  make test-neat-model MODEL=ścieżka/do/modelu.pkl  - Testowanie konkretnego modelu NEAT"
	@echo "  make clean                - Usunięcie plików tymczasowych"

.PHONY: train-dqn train-dqn-visual train-neat train-neat-visual continue-dqn continue-dqn-visual continue-neat test-dqn test-neat test-random test-dqn-model test-neat-model clean help