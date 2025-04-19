import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("iris1_1.csv")

all_inputs = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
all_classes = df["variety"].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
    all_inputs, all_classes, train_size=0.7, random_state=292567
)

classifiers = {
    'Drzewo decyzyjne': DecisionTreeClassifier(),
    'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'k-NN (k=11)': KNeighborsClassifier(n_neighbors=11),
    'Naive Bayes': GaussianNB()
}

accuracies = {}
cms = {}

for name, clf in classifiers.items():
    clf.fit(train_inputs, train_classes)
    
    y_pred = clf.predict(test_inputs)
     
    accuracy = accuracy_score(test_classes, y_pred)
    accuracies[name] = accuracy * 100
    
    print(f"\n{name}:")
    print(f"Dokładność: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = confusion_matrix(test_classes, y_pred)
    cms[name] = cm
    print("Macierz błędów:")
    print(cm)
    
    
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(cms.items()):
    ax = axes[i]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Setosa", "Versicolor", "Virginica"]
    )
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=False)
    ax.set_title(f'{name}\nDokładność: {accuracies[name]:.2f}%')
if len(cms) < len(axes):
    axes[-1].axis('off')
    
plt.tight_layout()
plt.suptitle('Porównanie macierzy pomyłek dla różnych klasyfikatorów', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()

print("\n--- Porównanie dokładności klasyfikatorów ---")
for name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {accuracy:.2f}%")
