import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# Konfiguracja GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Opcjonalnie: pozwól na dynamiczne wykorzystanie pamięci
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Używam GPU z dynamiczną alokacją pamięci")
    except:
        print("Używam GPU z domyślnymi ustawieniami pamięci")
# Hyperparameters
EPISODES = 1000
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 128
MEMORY_SIZE = 1000000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Neural Network for DQN
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(128, input_dim=state_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Main function
if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    for e in range(EPISODES):
        print(f'Episodes {e}/{EPISODES}')
        state = env.reset()[0]
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0


        while not done :
            #i = i + 1
            action = agent.act(state)
            next_state, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
            agent.replay()
        if (e + 1) % TARGET_UPDATE == 0:
            agent.update_target_model()
    env.close()
