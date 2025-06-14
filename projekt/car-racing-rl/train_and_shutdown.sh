#!/bin/bash
# Nazwa pliku: train_and_shutdown.sh

echo "Rozpoczynam trening DQN..."
systemd-inhibit --what=idle:sleep:shutdown make train-dqn
echo "Trening zakończony, wyłączam komputer za 30 sekund..."
sleep 30  # 30 sekund na anulowanie wyłączenia
sudo shutdown -h now