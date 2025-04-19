import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')


pca = PCA()
pca.fit(X)


cumulative_variance = np.cumsum(pca.explained_variance_ratio_)


print("\nWariancja wyjaśniona przez każdą składową:")
for i, var in enumerate(pca.explained_variance_ratio_, 1):
    print(f"Składowa {i}: {var:.4f} ({var*100:.2f}%)")

print("\nSkumulowana wariancja:")
for i, cum_var in enumerate(cumulative_variance, 1):
    print(f"Pierwsze {i} składowe: {cum_var:.4f} ({cum_var*100:.2f}%)")


information_loss = 1 - cumulative_variance
print("\nStrata informacji przy usunięciu ostatnich składowych:")
for i in range(1, len(information_loss)):
    print(f"Pozostawienie {i} pierwszych składowych: {information_loss[i-1]:.4f} ({information_loss[i-1]*100:.2f}%)")


X_pca = pca.transform(X)

species = ['Setosa', 'Versicolor', 'Virginica']
colors = ['red', 'blue', 'green']


for i, species_name in enumerate(species):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[i], 
                label=species_name,
                alpha=0.7)

plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')
plt.title('Wizualizacja danych Iris po redukcji do 2D')
plt.legend()
plt.grid(True)
plt.show()