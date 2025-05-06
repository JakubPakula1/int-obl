import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Wczytanie danych
file_path = '/Users/kuba/Desktop/Studia UG/semestr_4/inteligencja_obliczeniowa/lab09/zad01/titanic.csv'
df = pd.read_csv(file_path)

# Upewnienie się, że dane mają tylko wymagane kolumny
df = df[['Class', 'Sex', 'Age', 'Survived']]

# Konwersja danych na format one-hot encoding
df_encoded = pd.get_dummies(df)

# Uruchomienie algorytmu Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)

# Wyszukanie reguł asocjacyjnych
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Posortowanie reguł według ufności
rules = rules.sort_values(by='confidence', ascending=False)

# Filtrowanie reguł wskazujących na przeżycie lub brak przeżycia
interesting_rules = rules[(rules['consequents'].astype(str).str.contains('Survived_Yes')) |
                          (rules['consequents'].astype(str).str.contains('Survived_No'))]

# Wyświetlenie najciekawszych reguł
print("Najciekawsze reguły:")
print(interesting_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Wizualizacja reguł
# Tworzenie wspólnego okna dla dwóch wykresów
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Wykres 1: Scatter plot
scatter = axes[0].scatter(interesting_rules['support'], interesting_rules['confidence'], 
                          c=interesting_rules['lift'], cmap='viridis', s=100)
fig.colorbar(scatter, ax=axes[0], label='Lift')
axes[0].set_title('Reguły asocjacyjne - Scatter plot')
axes[0].set_xlabel('Support')
axes[0].set_ylabel('Confidence')

# Dodanie etykiet do punktów (po indeksie reguły)
for i, rule in interesting_rules.iterrows():
    label = f"ID: {i}"  
    axes[0].annotate(label, (rule['support'], rule['confidence']), fontsize=8, alpha=0.7)


# Wykres 2: Bar plot
top_rules = interesting_rules.head(10)  # Wybierz 10 najciekawszych reguł
axes[1].barh(range(len(top_rules)), top_rules['confidence'], color='skyblue')
axes[1].set_yticks(range(len(top_rules)))
axes[1].set_yticklabels([f"{list(ant)} -> {list(cons)}" for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])])
axes[1].set_xlabel('Confidence')
axes[1].set_title('Top 10 najciekawszych reguł asocjacyjnych')
axes[1].invert_yaxis()  # Odwrócenie osi Y dla lepszej czytelności

# Wyświetlenie wykresów
plt.tight_layout()
plt.show()

