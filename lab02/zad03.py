import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('iris1.csv')


X = df[['sepal.length', 'sepal.width']]
y = df['variety']


plt.figure(figsize=(18, 5))


plt.subplot(131)
for variety in df['variety'].unique():
    mask = y == variety
    plt.scatter(X[mask]['sepal.length'], X[mask]['sepal.width'], 
               label=variety, alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Oryginalne dane')
plt.legend()
plt.grid(True)


plt.subplot(132)
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
df_minmax = pd.DataFrame(X_minmax, columns=['sepal.length', 'sepal.width'])

for variety in df['variety'].unique():
    mask = y == variety
    plt.scatter(df_minmax.loc[mask, 'sepal.length'], 
               df_minmax.loc[mask, 'sepal.width'],
               label=variety, alpha=0.7)
plt.xlabel('Sepal Length (znormalizowany)')
plt.ylabel('Sepal Width (znormalizowany)')
plt.title('Normalizacja Min-Max')
plt.legend()
plt.grid(True)


plt.subplot(133)
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
df_standard = pd.DataFrame(X_standard, columns=['sepal.length', 'sepal.width'])

for variety in df['variety'].unique():
    mask = y == variety
    plt.scatter(df_standard.loc[mask, 'sepal.length'], 
               df_standard.loc[mask, 'sepal.width'],
               label=variety, alpha=0.7)
plt.xlabel('Sepal Length (standaryzowany)')
plt.ylabel('Sepal Width (standaryzowany)')
plt.title('Standaryzacja Z-score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


print("\nStatystyki dla oryginalnych danych:")
print(X.describe())
print("\nStatystyki dla danych znormalizowanych (Min-Max):")
print(df_minmax.describe())
print("\nStatystyki dla danych zeskalowanych (Z-score):")
print(df_standard.describe())