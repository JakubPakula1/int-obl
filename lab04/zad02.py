import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target 
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df['variety'] = [target_names[i] for i in y]
    
    print(f"Załadowano {len(df)} próbek")
    print(f"Klasy: {target_names}")

    return X, y, feature_names, target_names

def train_and_evaluate(inputs, classes, hidden_layers, name):

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
        inputs, classes, train_size=0.7, random_state=292567
    )
    
    clf = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=hidden_layers,
        random_state=1,
        max_iter=2000
    )
    
    clf.fit(train_inputs, train_classes)
    
    # Ocena modelu
    predictions_train = clf.predict(train_inputs)
    predictions_test = clf.predict(test_inputs)
    
    train_score = accuracy_score(predictions_train, train_classes)
    test_score = accuracy_score(predictions_test, test_classes)
    
    # Wyświetlenie wyników
    print(f"\n--- Model: {name} ---")
    print(f"Architektura warstw ukrytych: {hidden_layers}")
    print(f"Dokładność na zbiorze treningowym: {train_score:.4f}")
    print(f"Dokładność na zbiorze testowym: {test_score:.4f}")
    
    return clf, test_score
def main():
    # Wczytanie danych
    X, y, feature_names, target_names = load_data()
    
    print("\nPorównywanie różnych architektur sieci neuronowych:")
    
    # Trening i ewaluacja różnych modeli
    model1, score1 = train_and_evaluate(X, y, (2,), "Jedna warstwa, 2 neurony")
    model2, score2 = train_and_evaluate(X, y, (3,), "Jedna warstwa, 3 neurony")
    model3, score3 = train_and_evaluate(X, y, (3,3), "Dwie warstwy po 3 neurony")
    
    models = {
        "Jedna warstwa, 2 neurony": (model1, score1),
        "Jedna warstwa, 3 neurony": (model2, score2),
        "Dwie warstwy po 3 neurony": (model3, score3)
    }
    
    best_model_name = max(models, key=lambda x: models[x][1])
    
    print("\n--- Podsumowanie ---")
    print(f"Najlepszy model: {best_model_name}")
    print(f"Dokładność: {models[best_model_name][1]:.4f}")

if __name__ == "__main__":
    main()