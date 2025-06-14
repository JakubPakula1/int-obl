from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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