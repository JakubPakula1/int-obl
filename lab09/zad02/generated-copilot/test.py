from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint
from keras.datasets.mnist import load_data
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Input
from keras.layers import Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import load_model
from keras.constraints import Constraint
import keras.backend as K
from matplotlib import pyplot
import tensorflow as tf  # Dodajemy import TensorFlow
# Clipper wag dla WGAN
class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)
    
    def get_config(self):
        return {'clip_value': self.clip_value}

# Funkcja straty Wasserstein
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred) 

# Krytyk (nie dyskryminator) dla WGAN - bez sigmoid na końcu
def define_critic(in_shape=(28,28,1)):
    const = ClipConstraint(0.01)
    model = Sequential()
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', input_shape=in_shape, kernel_constraint=const))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, kernel_constraint=const))  # Bez sigmoid!
    # Kompilacja z loss='wasserstein' zamiast binary_crossentropy
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# Generator dla WGAN
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

# Definiowanie GAN (złożenie generatora i krytyka)
def define_gan(g_model, c_model):
    # Zamrożenie wag krytyka
    c_model.trainable = False
    # Połączenie modeli
    model = Sequential()
    model.add(g_model)
    model.add(c_model)
    # Kompilacja z funkcją straty Wasserstein
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# Ładowanie danych MNIST
def load_real_samples():
    (trainX, _), (_, _) = load_data()
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X

# Generowanie prawdziwych próbek
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # W WGAN używamy etykiet +1 dla prawdziwych próbek
    y = ones((n_samples, 1))
    return X, y

# Generowanie punktów w przestrzeni ukrytej
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Generowanie fałszywych próbek
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    # W WGAN używamy etykiet -1 dla fałszywych próbek
    y = -ones((n_samples, 1))
    return X, y

# Zapisywanie wygenerowanych obrazów
def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

# Ocena wydajności modelu
def summarize_performance(epoch, g_model, c_model, dataset, latent_dim, n_samples=100):
    # Przygotowanie prawdziwych próbek
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # Ocena krytyka na prawdziwych próbkach
    loss_real = c_model.evaluate(X_real, y_real, verbose=0)
    # Przygotowanie fałszywych próbek
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # Ocena krytyka na fałszywych próbkach
    loss_fake = c_model.evaluate(X_fake, y_fake, verbose=0)
    # Podsumowanie wydajności krytyka
    print('>Loss real: %.3f, fake: %.3f' % (loss_real, loss_fake))
    # Zapisywanie obrazów
    save_plot(X_fake, epoch)
    # Zapisywanie modelu generatora
    filename = 'generator_model_%03d.keras' % (epoch+1)
    g_model.save(filename)

# Funkcja treningu WGAN
# Funkcja treningu WGAN z obsługą start_epoch
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128, start_epoch=0):
    batch_count = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    for i in range(start_epoch, n_epochs):
        for j in range(batch_count):
            # Trenowanie krytyka 5x na każde trenowanie generatora
            for _ in range(5):
                # Prawdziwe próbki
                X_real, y_real = generate_real_samples(dataset, half_batch)
                # Fałszywe próbki 
                X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
                # Trenowanie na prawdziwych i fałszywych próbkach
                c_loss1 = c_model.train_on_batch(X_real, y_real)
                c_loss2 = c_model.train_on_batch(X_fake, y_fake)
                c_loss = 0.5 * (c_loss1 + c_loss2)
            
            # Trenowanie generatora
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = -ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            print('>%d, %d/%d, c=%.3f, g=%.3f' % 
                 (i+1, j+1, batch_count, c_loss, g_loss))
                 
        # Ewaluacja modeli po epoce
        if (i+1) % 1 == 0:
            summarize_performance(i, g_model, c_model, dataset, latent_dim)
def continue_training(generator_path, start_epoch, additional_epochs=20):
    # Parametry modelu
    latent_dim = 100
    
    # Załadowanie zapisanego generatora
    custom_objects = {'wasserstein_loss': wasserstein_loss}
    g_model = load_model(generator_path, custom_objects=custom_objects)
    
    # Stworzenie nowego krytyka (nie zapisujemy krytyka w oryginalnym kodzie)
    c_model = define_critic()
    
    # Stworzenie modelu GAN
    gan_model = define_gan(g_model, c_model)
    
    # Ładowanie danych
    dataset = load_real_samples()
    
    # Kontynuacja treningu
    print(f"Kontynuacja treningu od epoki {start_epoch+1}...")
    train(g_model, c_model, gan_model, dataset, latent_dim, 
          n_epochs=start_epoch+additional_epochs, start_epoch=start_epoch)

# Główna funkcja uruchamiająca trening
def main():
    # Ścieżka do ostatniego zapisanego modelu (dostosuj nazwę pliku)
    generator_path = "generator_model_020.keras"  # Model z 20 epoki
    
    # Kontynuuj trening od 20 epoki przez dodatkowe 20 epok
    continue_training(generator_path, start_epoch=20, additional_epochs=100)
# Wywołanie głównej funkcji
if __name__ == '__main__':
    main()