import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")

target_column = ["class"]

predicotrs = list(set(list(df.columns))-set(target_column))

X = df[predicotrs].values
y = df[target_column].values.ravel()

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.30,
random_state=292567)

print("\n----- Model 1: (6,3) neurony, ReLU -----")
mlp1 = MLPClassifier(hidden_layer_sizes=(6,3), activation="relu", solver="adam", max_iter=500, random_state=1)

mlp1.fit(X_train, y_train)

predict_train = mlp1.predict(X_train)
predict_test = mlp1.predict(X_test)

print("\nDokładność na zbiorze treningowym:", accuracy_score(y_train, predict_train))
print("Dokładność na zbiorze testowym:", accuracy_score(y_test, predict_test))

print("\nMacierz błędu (treningowy):")
cm_train = confusion_matrix(y_train, predict_train)
print(cm_train)
print(classification_report(y_train, predict_train))

print("\nMacierz błędu (testowy):")
cm_test = confusion_matrix(y_test, predict_test)
print(cm_test)
print(classification_report(y_test, predict_test))

print("\n----- Model 2: (10,5) neurony, tanh -----")
mlp2 = MLPClassifier(hidden_layer_sizes=(10,5), activation="tanh", 
                    solver="adam", max_iter=500, random_state=1)
mlp2.fit(X_train, y_train)

predict_train2 = mlp2.predict(X_train)
predict_test2 = mlp2.predict(X_test)

print("\nDokładność na zbiorze treningowym:", accuracy_score(y_train, predict_train2))
print("Dokładność na zbiorze testowym:", accuracy_score(y_test, predict_test2))

print("\nMacierz błędu (testowy):")
cm_test2 = confusion_matrix(y_test, predict_test2)
print(cm_test2)
print(classification_report(y_test, predict_test2))


print("\n----- Model 3: (20,10,5) neurony, logistic -----")
mlp3 = MLPClassifier(hidden_layer_sizes=(20,10,5), activation="logistic", 
                    solver="adam", max_iter=2000, random_state=1)
mlp3.fit(X_train, y_train)

predict_train3 = mlp3.predict(X_train)
predict_test3 = mlp3.predict(X_test)

print("\nDokładność na zbiorze treningowym:", accuracy_score(y_train, predict_train3))
print("Dokładność na zbiorze testowym:", accuracy_score(y_test, predict_test3))

print("\nMacierz błędu (testowy):")
cm_test3 = confusion_matrix(y_test, predict_test3)
print(cm_test3)
print(classification_report(y_test, predict_test3))

print("\n----- Analiza błędów -----")
print("Model 1:")
FN1 = cm_test[1,0]  
FP1 = cm_test[0,1]  
print(f"Fałszywie negatywne (FN): {FN1}")
print(f"Fałszywie pozytywne (FP): {FP1}")

print("\nModel 2:")
FN2 = cm_test2[1,0]
FP2 = cm_test2[0,1]
print(f"Fałszywie negatywne (FN): {FN2}")
print(f"Fałszywie pozytywne (FP): {FP2}")

print("\nModel 3:")
FN3 = cm_test3[1,0]
FP3 = cm_test3[0,1]
print(f"Fałszywie negatywne (FN): {FN3}")
print(f"Fałszywie pozytywne (FP): {FP3}")

"""
f) Analiza błędów FP i FN:

W przypadku diagnozy cukrzycy:
- FP Osoba zdrowa została błędnie zidentyfikowana jako chora
- FN Osoba chora została błędnie zidentyfikowana jako zdrowa

Znacznie więcej błędów FN niż FP, oznacza to, że wiele osób chorych nie zostałoby zdiagnozowanych.

FN są gorsze, ponieważ niezdiagnozowana 
choroba może prowadzić do poważnych konsekwencji zdrowotnych. Osoba z cukrzycą, 
która została błędnie sklasyfikowana jako zdrowa, nie otrzyma niezbędnego leczenia.

FP prowadzą tylko do dodatkowych testów.

Wszystkie modele wykazują podobny problem - wysoką liczbę FN w porównaniu do FP,
"""