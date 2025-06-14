import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import os

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
        self.buffer_size = buffer_size  # Zwiększony z 500 do 2048
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
        
        # Optymalizatory - zmniejszone learning rate
        self.actor_optimizer = Adam(learning_rate=learning_rate*0.5, clipnorm=0.5)
        self.critic_optimizer = Adam(learning_rate=learning_rate, clipnorm=0.5)
        
        # Utworzenie katalogu na checkpointy
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"🤖 PPO Agent utworzony:")
        print(f"   Buffer size: {self.buffer_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Epsilon clip: {epsilon_clip}")
    
    def build_actor(self):
        """Buduje sieć aktora - bez Lambda layer"""
        inputs = layers.Input(shape=self.state_size)
        
        # Lżejsze warstwy konwolucyjne
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        
        # Mniejsze warstwy dense
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Wyjścia dla akcji
        mu_steering = layers.Dense(1, activation='tanh', name='mu_steering')(x)
        mu_gas = layers.Dense(1, activation='sigmoid', name='mu_gas')(x)
        mu_brake = layers.Dense(1, activation='sigmoid', name='mu_brake')(x)
        
        # POPRAWKA: Sigma bez Lambda layer - bezpośrednie obliczenie
        sigma_raw = layers.Dense(3, activation='sigmoid', name='sigma_raw')(x)
        
        # Łączenie wyjść
        mu = layers.Concatenate(name='mu')([mu_steering, mu_gas, mu_brake])
        
        model = Model(inputs=inputs, outputs=[mu, sigma_raw])
        return model
    
    def build_critic(self):
        """Buduje sieć krytyka - zoptymalizowana"""
        inputs = layers.Input(shape=self.state_size)
        
        # Te same warstwy konwolucyjne
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), strides=1, activation='relu')(x)
        x = layers.Flatten()(x)
        
        # Warstwy dense
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Wyjście
        value = layers.Dense(1, activation='linear', name='value')(x)
        
        model = Model(inputs=inputs, outputs=value)
        return model
    

    def get_action(self, state):
        """Wybiera akcję - poprawiona bez Lambda layer"""
        state_batch = np.expand_dims(state, axis=0)
        
        # Przewidywania z aktora
        mu, sigma_raw = self.actor(state_batch, training=False)
        mu = mu.numpy()[0]
        sigma_raw = sigma_raw.numpy()[0]
        
        # POPRAWKA: Oblicz sigma tutaj zamiast w Lambda layer
        sigma = sigma_raw * 0.5 + 0.1  # sigma w zakresie [0.1, 0.6]
        
        # Próbkowanie z rozkładu normalnego
        action = np.random.normal(mu, sigma)
        
        # Ograniczenie akcji
        action[0] = np.clip(action[0], -1.0, 1.0)  # steering
        action[1] = np.clip(action[1], 0.0, 1.0)   # gas
        action[2] = np.clip(action[2], 0.0, 1.0)   # brake
        
        # Log prawdopodobieństwo
        log_prob = self.compute_log_prob(action, mu, sigma)
        
        return action, log_prob
    
    def compute_log_prob(self, action, mu, sigma):
        """Oblicza log prawdopodobieństwo"""
        log_prob = -0.5 * np.sum(((action - mu) / sigma) ** 2)
        log_prob -= np.sum(np.log(sigma))
        log_prob -= 1.5 * np.log(2 * np.pi)
        return log_prob
    
    def should_update(self):
        """Sprawdza czy należy wykonać aktualizację"""
        return len(self.states) >= self.buffer_size
    
    def update(self):
        """Aktualizacja PPO - zoptymalizowana"""
        if len(self.states) < 64:  # Minimum dla sensownej aktualizacji
            return
        
        print(f"🔄 Aktualizacja PPO - batch size: {len(self.states)}")
        
        # Konwersja do numpy
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        old_log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        
        # GAE
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalizacja advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
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
                
                # Batch data
                batch_states = tf.convert_to_tensor(states[batch_indices], dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(actions[batch_indices], dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(returns[batch_indices], dtype=tf.float32)
                batch_old_log_probs = tf.convert_to_tensor(old_log_probs[batch_indices], dtype=tf.float32)
                
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
        """Aktualizacja aktora - poprawiona"""
        with tf.GradientTape() as tape:
            mu, sigma_raw = self.actor(states, training=True)
            
            # POPRAWKA: Oblicz sigma w TensorFlow
            sigma = sigma_raw * 0.5 + 0.1
            
            new_log_probs = self.compute_log_prob_tensor(actions, mu, sigma)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
            
            policy_loss = -tf.reduce_mean(tf.minimum(
                ratio * advantages,
                clipped_ratio * advantages
            ))
            
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
    
    def compute_gae(self, rewards, values, dones):
        """GAE computation"""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        
        return advantages
    
    def clear_memory(self):
        """Czyść pamięć"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Zapisz przejście"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def act(self, state):
        """Interfejs dla testowania"""
        action, _ = self.get_action(state)
        return action
    
    def save_model(self, custom_name=None):
        """Zapisz modele"""
        try:
            if custom_name:
                actor_path = os.path.join(self.checkpoint_dir, f"{custom_name}_actor.keras")
                critic_path = os.path.join(self.checkpoint_dir, f"{custom_name}_critic.keras")
                params_path = os.path.join(self.checkpoint_dir, f"{custom_name}_params.json")
            else:
                actor_path = os.path.join(self.checkpoint_dir, f"ppo_actor_ep{self.episodes}.keras")
                critic_path = os.path.join(self.checkpoint_dir, f"ppo_critic_ep{self.episodes}.keras")
                params_path = os.path.join(self.checkpoint_dir, f"ppo_model_ep{self.episodes}_params.json")
            
            self.actor.save(actor_path)
            self.critic.save(critic_path)
            
            import json
            params = {
                'episodes': self.episodes,
                'steps': self.steps,
                'training_steps': self.training_steps,
                'state_size': self.state_size,
                'action_size': self.action_size
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
        import json
        
        agent = cls(state_size, action_size)
        
        try:
            agent.actor = load_model(actor_path)
            agent.critic = load_model(critic_path)
            
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