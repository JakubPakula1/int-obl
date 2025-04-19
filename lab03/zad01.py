import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris1_1.csv")

#podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=292567)

def classify_iris(sl, sw, pl, pw):
    if pl <= 2 and pw < 1:
        return "Setosa"
    elif pl >= 5 and pw >= 1.8:
        return "Virginica"
    else:
        return "Versicolor"
# Posortowanie po typie
# train_df = pd.DataFrame(train_set, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety'])
# print(train_df.sort_values(by='variety').to_string())
good_predictions = 0
len = test_set.shape[0]
for i in range(len):
    if classify_iris(test_set[i][0],test_set[i][1],test_set[i][2],test_set[i][3]) == test_set[i][4]:
        good_predictions = good_predictions + 1
print(f"{good_predictions}/{len}")
print(good_predictions/len*100, "%")

