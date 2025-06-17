from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

class DQNNetwork:
    def __init__(self, input_shape=(84, 84, 1), action_space=5):
        # Znacznie uproszczona architektura sieci
        self.model = Sequential([
            Conv2D(6, (7, 7), strides=3, activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),  # Efektywniejsze niż stride
            Conv2D(12, (4, 4), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(216, activation='relu'),  # Mniejsza warstwa dense
            Dense(action_space, activation='linear')
        ])
        
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        self.model.summary()
    
    def predict(self, state, verbose=0):
        return self.model.predict(state, verbose=verbose)
    
    def fit(self, state, target_f, epochs=1, verbose=0, batch_size=32):
        return self.model.fit(state, target_f, epochs=epochs, verbose=verbose, batch_size=batch_size)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load(cls, model_path):
        from tensorflow.keras.models import load_model
        network = cls()
        network.model = load_model(model_path)
        return network

class PPOActorNetwork:
    def __init__(self, input_shape=(84, 84, 1), action_space=3):
        """
        Sieć aktora dla PPO - generuje politykę dla ciągłych akcji
        action_space=3 dla CarRacing: [steering, gas, brake]
        """
        
        inputs = Input(shape=input_shape)
        
        # Warstwy konwolucyjne
        x = Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = Conv2D(32, (3, 3), strides=1, activation='relu')(x)
        x = Flatten()(x)
        
        # Warstwy dense
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        # Wyjścia dla średnich akcji (mu)
        mu_steering = Dense(1, activation='tanh', name='mu_steering')(x)
        mu_gas = Dense(1, activation='sigmoid', name='mu_gas')(x)
        mu_brake = Dense(1, activation='sigmoid', name='mu_brake')(x)
        
        # Wyjścia dla odchyleń standardowych (sigma)
        sigma_raw = Dense(action_space, activation='sigmoid', name='sigma_raw')(x)
        
        # Łączenie wyjść
        mu = Concatenate(name='mu')([mu_steering, mu_gas, mu_brake])
        
        self.model = Model(inputs=inputs, outputs=[mu, sigma_raw])
        self.model.compile(optimizer=Adam(learning_rate=0.0003))
    
    def predict(self, state, training=False):
        return self.model(state, training=training)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load(cls, model_path):
        from tensorflow.keras.models import load_model
        network = cls()
        network.model = load_model(model_path)
        return network

class PPOCriticNetwork:
    def __init__(self, input_shape=(84, 84, 1)):
        """
        Sieć krytyka dla PPO - ocenia wartość stanu
        """
        
        inputs = Input(shape=input_shape)
        
        # Warstwy konwolucyjne - takie same jak w aktorze
        x = Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)
        x = Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = Conv2D(32, (3, 3), strides=1, activation='relu')(x)
        x = Flatten()(x)
        
        # Warstwy dense
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        # Wyjście - wartość stanu
        value = Dense(1, activation='linear', name='value')(x)
        
        self.model = Model(inputs=inputs, outputs=value)
        self.model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
    
    def predict(self, state, training=False):
        return self.model(state, training=training)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load(cls, model_path):
        from tensorflow.keras.models import load_model
        network = cls()
        network.model = load_model(model_path)
        return network