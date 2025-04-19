import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import load_model
# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Encode the labels
encoder = OneHotEncoder(sparse_output =False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3,
random_state=42)
# Load the pre-trained model
model = load_model('iris_model.h5')
# Continue training the model for 10 more epochs
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
# Save the updated model
model.save('updated_iris_model.h5')
# Evaluate the updated model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

#?Ładuje wcześniej wytrenowany model.
#?Przetwarza dane wejściowe
#?Dzieli dane na zbiory treningowy i testowy.
#?Dalszej trenuje model przez 10 epok.
#?Zapisuje zaktualizowany model.
#?Ocenia model na zbiorze testowym i wyświetla dokładność.