import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical, image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Parametry modelu
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
DATASET_PATH = "dogs-cats-mini"

# Funkcja do wczytywania danych
def load_data(dataset_path, image_size):
    try:
        dataset = image_dataset_from_directory(
            dataset_path,
            image_size=image_size,
            batch_size=32,
            label_mode="int"
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        print(f"Zbiór treningowy: {train_size} batchy")
        print(f"Zbiór walidacyjny: {val_size} batchy")
        return train_dataset, val_dataset
    except Exception as e:
        print(f"Błąd podczas wczytywania danych: {e}")
        return None, None

# Funkcja do wyświetlania przykładowych obrazów
def show_sample_images_with_labels(dataset, num_samples=5):
    plt.figure(figsize=(15, 15))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            ax = plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

# Funkcja do budowy modelu
def build_model1(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), activation='tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# Funkcja do wyświetlania krzywej uczenia się
def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Dokładność treningu')
    plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
    plt.legend()
    plt.title('Dokładność')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')

    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Strata treningu')
    plt.plot(history.history['val_loss'], label='Strata walidacji')
    plt.legend()
    plt.title('Strata')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')

    plt.tight_layout()
    plt.show()

# Funkcja do wyświetlania macierzy konfuzji
def plot_confusion_matrix(model, val_dataset):
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_dataset], axis=0)
    y_pred = np.concatenate([np.argmax(model.predict(x), axis=1) for x, _ in val_dataset], axis=0)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Macierz konfuzji")
    plt.show()


if __name__ == "__main__":
    # Wczytanie danych
    train_dataset, val_dataset = load_data(DATASET_PATH, IMAGE_SIZE)

    # Budowa modelu
    model = build_model((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    model.summary()

    # Przygotowanie danych (one-hot encoding)
    train_dataset = train_dataset.map(lambda x, y: (x, to_categorical(y, num_classes=2)))
    val_dataset = val_dataset.map(lambda x, y: (x, to_categorical(y, num_classes=2)))

    # Trenowanie modelu
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,  
        verbose=1
    )

    # Wyświetlenie krzywej uczenia się
    plot_learning_curves(history)

    # Wyświetlenie macierzy konfuzji
    plot_confusion_matrix(model, val_dataset)