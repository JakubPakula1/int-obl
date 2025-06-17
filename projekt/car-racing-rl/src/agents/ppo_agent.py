import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import os
import json

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99, 
                 epsilon_clip=0.2, epochs=4, batch_size=64, buffer_size=2048,
                 gae_lambda=0.95, checkpoint_dir='checkpoints/ppo'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gae_lambda = gae_lambda
        self.checkpoint_dir = checkpoint_dir
        
        # Pamięć epizodu
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Liczniki
        self.episodes = 0
        self.steps = 0
        self.training_steps = 0
        
        # Budowa modeli
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        # Optymalizatory
        self.actor_optimizer = Adam(learning_rate=learning_rate, clipnorm=0.5)
        self.critic_optimizer = Adam(learning_rate=learning_rate*1.5, clipnorm=0.5)
        
        # Utworzenie katalogu na checkpointy
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"🤖 PPO Agent utworzony:")
        print(f"   📊 Buffer size: {buffer_size}")
        print(f"   🧠 Learning rate: {learning_rate}")
        print(f"   📈 Epochs: {epochs}, Batch size: {batch_size}")

    def build_actor(self):
        """Buduje sieć aktora"""
        inputs = layers.Input(shape=self.state_size)
        
        # Warstwy konwolucyjne - podobne do DQN
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        
        # Warstwy dense
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Wyjścia dla średnich akcji (mu)
        mu_steering = layers.Dense(1, activation='tanh', name='mu_steering')(x)
        mu_gas = layers.Dense(1, activation='sigmoid', name='mu_gas')(x)  
        mu_brake = layers.Dense(1, activation='sigmoid', name='mu_brake')(x)
        
        # Wyjścia dla odchylenia standardowego (sigma)
        sigma_steering = layers.Dense(1, activation='softplus', name='sigma_steering')(x)
        sigma_gas = layers.Dense(1, activation='softplus', name='sigma_gas')(x)
        sigma_brake = layers.Dense(1, activation='softplus', name='sigma_brake')(x)
        
        # Łączenie wyjść
        mu = layers.Concatenate(name='mu')([mu_steering, mu_gas, mu_brake])
        sigma = layers.Concatenate(name='sigma')([sigma_steering, sigma_gas, sigma_brake])
        
        model = Model(inputs=inputs, outputs=[mu, sigma])
        return model

    def build_critic(self):
        """Buduje sieć krytyka"""
        inputs = layers.Input(shape=self.state_size)
        
        # Te same warstwy konwolucyjne co aktor
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        
        # Warstwy dense
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Wyjście - wartość stanu
        value = layers.Dense(1, activation='linear', name='value')(x)
        
        model = Model(inputs=inputs, outputs=value)
        return model

    def get_action(self, state):
        """Wybiera akcję na podstawie polityki"""
        state_batch = np.expand_dims(state, axis=0)
        
        # Przewidywania z aktora
        mu, sigma = self.actor(state_batch, training=False)
        mu = mu.numpy()[0]
        sigma = sigma.numpy()[0]
        
        # Dodaj minimalną wartość sigma dla stabilności
        sigma = np.maximum(sigma, 0.01)
        
        # Próbkowanie z rozkładu normalnego
        action = np.random.normal(mu, sigma)
        
        # Ograniczenie akcji do właściwych zakresów
        action[0] = np.clip(action[0], -1.0, 1.0)   # steering [-1, 1]
        action[1] = np.clip(action[1], 0.0, 1.0)    # gas [0, 1]
        action[2] = np.clip(action[2], 0.0, 1.0)    # brake [0, 1]
        
        # Oblicz log prawdopodobieństwo
        log_prob = self.compute_log_prob(action, mu, sigma)
        
        return action, log_prob

    def compute_log_prob(self, action, mu, sigma):
        """Oblicza log prawdopodobieństwo akcji"""
        log_prob = -0.5 * np.sum(((action - mu) / sigma) ** 2)
        log_prob -= np.sum(np.log(sigma))
        log_prob -= 1.5 * np.log(2 * np.pi)
        return log_prob

    def act(self, state):
        """Interfejs dla testowania - deterministyczna akcja"""
        state_batch = np.expand_dims(state, axis=0)
        mu, _ = self.actor(state_batch, training=False)
        action = mu.numpy()[0]
        
        # Ograniczenie akcji
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.0, 1.0)
        action[2] = np.clip(action[2], 0.0, 1.0)
        
        return action

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Zapisz przejście w pamięci"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def should_update(self):
        """Sprawdza czy należy wykonać aktualizację"""
        return len(self.states) >= self.buffer_size

    def compute_gae(self, rewards, values, dones):
        """Oblicza Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages[i] = gae
        
        return advantages

    def update(self):
        """Główna funkcja aktualizacji PPO"""
        if len(self.states) < self.batch_size:
            return
        
        print(f"🔄 Aktualizacja PPO - batch size: {len(self.states)}")
        
        # Konwersja do numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        old_log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        
        # Oblicz advantages i returns
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalizacja advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Konwersja do tensorów
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        
        # Mini-batch training
        batch_size = min(self.batch_size, len(states))
        indices = np.arange(len(states))
        
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                # Batch tensors
                batch_states = tf.gather(states_tensor, batch_indices)
                batch_actions = tf.gather(actions_tensor, batch_indices)
                batch_advantages = tf.gather(advantages_tensor, batch_indices)
                batch_returns = tf.gather(returns_tensor, batch_indices)
                batch_old_log_probs = tf.gather(old_log_probs_tensor, batch_indices)
                
                # Aktualizacje
                actor_loss = self.update_actor(batch_states, batch_actions, 
                                             batch_advantages, batch_old_log_probs)
                critic_loss = self.update_critic(batch_states, batch_returns)
                
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
        
        self.training_steps += 1
        print(f"✅ Training step {self.training_steps} completed")
        print(f"   📉 Actor loss: {np.mean(actor_losses):.4f}")
        print(f"   📉 Critic loss: {np.mean(critic_losses):.4f}")
        
        # Wyczyść pamięć
        self.clear_memory()

    def update_actor(self, states, actions, advantages, old_log_probs):
        """Aktualizacja aktora"""
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(states, training=True)
            
            # Oblicz nowe log prawdopodobieństwa
            new_log_probs = self.compute_log_prob_tensor(actions, mu, sigma)
            
            # PPO ratio
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
            
            # Policy loss
            policy_loss = -tf.reduce_mean(tf.minimum(
                ratio * advantages,
                clipped_ratio * advantages
            ))
            
            # Entropy bonus
            entropy = tf.reduce_mean(tf.reduce_sum(tf.math.log(sigma + 1e-8), axis=1))
            actor_loss = policy_loss - 0.01 * entropy
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return actor_loss

    def update_critic(self, states, returns):
        """Aktualizacja krytyka"""
        with tf.GradientTape() as tape:
            predicted_values = tf.squeeze(self.critic(states, training=True))
            critic_loss = tf.reduce_mean(tf.square(returns - predicted_values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return critic_loss

    def compute_log_prob_tensor(self, actions, mu, sigma):
        """Log prawdopodobieństwo jako tensor"""
        log_prob = -0.5 * tf.reduce_sum(tf.square((actions - mu) / sigma), axis=1)
        log_prob -= tf.reduce_sum(tf.math.log(sigma), axis=1)
        log_prob -= 1.5 * tf.math.log(2 * np.pi)
        return log_prob

    def clear_memory(self):
        """Czyść pamięć"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def save_model(self, custom_name=None):
        """Zapisz modele"""
        try:
            if custom_name:
                actor_path = f"{self.checkpoint_dir}/{custom_name}_actor.keras"
                critic_path = f"{self.checkpoint_dir}/{custom_name}_critic.keras"
                params_path = f"{self.checkpoint_dir}/{custom_name}_params.json"
            else:
                actor_path = f"{self.checkpoint_dir}/ppo_ep{self.episodes}_actor.keras"
                critic_path = f"{self.checkpoint_dir}/ppo_ep{self.episodes}_critic.keras"
                params_path = f"{self.checkpoint_dir}/ppo_ep{self.episodes}_params.json"
            
            self.actor.save(actor_path)
            self.critic.save(critic_path)
            
            # Zapisz parametry
            params = {
                'episodes': self.episodes,
                'steps': self.steps,
                'training_steps': self.training_steps
            }
            
            with open(params_path, 'w') as f:
                json.dump(params, f)
            
            print(f"💾 Modele PPO zapisane: {actor_path}")
        except Exception as e:
            print(f"❌ Błąd zapisywania: {e}")

    @classmethod
    def load(cls, actor_path, critic_path, state_size, action_size):
        """Ładuj modele"""
        from tensorflow.keras.models import load_model
        
        agent = cls(state_size, action_size)
        
        try:
            agent.actor = load_model(actor_path)
            agent.critic = load_model(critic_path)
            
            # Wczytaj parametry jeśli istnieją
            params_path = actor_path.replace('_actor.keras', '_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                agent.episodes = params.get('episodes', 0)
                agent.steps = params.get('steps', 0)
                agent.training_steps = params.get('training_steps', 0)
                
                print(f"✅ Modele PPO wczytane: {actor_path}")
                print(f"📊 Przywrócone parametry: epizody={agent.episodes}")
            
        except Exception as e:
            print(f"❌ Błąd wczytywania: {e}")
        
        return agent