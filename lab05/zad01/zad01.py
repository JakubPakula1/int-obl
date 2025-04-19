import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3,
random_state=11)
# Define the model
model = Sequential([
Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
Dense(64, activation='relu'),
Dense(y_encoded.shape[1], activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

#Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()
# Save the model
model.save('iris_model.h5')
# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#?a)Transformuje dane tak, aby miały średnią równą 0 i odchylenie standardowe równe 1
#?Typowo dla rozkładu normalnego większość wartości mieści się w przedziale od około -3 do +3
#?dane bardziej odpornye na wartości odstające i często zapewnia lepszą zbieżność w procesie uczenia.


#?b)"one hot" to technika przekształcania zmiennych kategorycznych (np. etykiet klas) w format binarny, który może być lepiej przetwarzany przez algorytmy uczenia maszynowego
#?Klasa 0 (Setosa)      → [1, 0, 0]
#?Klasa 1 (Versicolor)  → [0, 1, 0]
#?Klasa 2 (Virginica)   → [0, 0, 1]

#?c)Liczba neuronów w warstwie wejściowej: X_train.shape[1], czyli 4
#?Co oznacza X_train.shape[1]?
#?Jest to liczba cech (kolumn) w zbiorze danych treningowych
#?Liczba neuronów w warstwie wyjściowej: y_encoded.shape[1], czyli 3
#?Jest to liczba klas w zbiorze danych po przekształceniu na format one-hot

#?d)
#? tanh - Test Accuracy: 82.22% - Może zapewnić lepszą zbieżność dla małych zbiorów danych
#? sigmoid - Test Accuracy: 86.67% - Zwraca wartości w zakresie [0, 1]
#? elu - Test Accuracy: 80.00% - Podobna do ReLU, ale łagodniej obsługuje ujemne wartości
#? relu - Test Accuracy: 84.44%

#?e)
#?SGD - Test Accuracy: 82.22%
#?RMSprop - Test Accuracy: 88.89% (Root Mean Square Propagation)
# Utrzymywania średniej ważonej kwadratów poprzednich gradientów dla każdego parametru
# Dzielenia bieżącego gradientu przez pierwiastek tej średniej
#? Tak, można dostosować szybkość uczenia (learning rate).
#? po zmniejszeniu do 0.0001 dokladnos spadła po zwiększeniu do 0.01 wzrosła


#? f) batch_size=?
#? 32 => 84.44%
#? 16 => 86.67%
#? 8 => 88.89%
#? 4 => 86.67%
#TODO Mały rozmiar partii (np. 4):
#? Częstsze aktualizacje wag
#? Większe wahania w krzywych uczenia
#? Wolniejszy trening (więcej kroków w każdej epoce)

#TODO Większy rozmiar partii (np. 16 lub 32):
#? Bardziej stabilne aktualizacje gradientu
#? Gładsze krzywe uczenia
#? Szybszy trening (mniej kroków na epokę)

#? g) 
#? czy model przestaje się poprawiać w późniejszych epokach
#? jak model radzi sobie na danych walidacyjnych w porównaniu do treningowych
#? kiedy najlepiej zatrzymać uczenie
#? dobrze dopasowany , ale zaczyna przeuczać się po około 20 epoce