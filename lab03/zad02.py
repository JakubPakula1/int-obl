from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import graphviz 

df = pd.read_csv("iris1_1.csv")

all_inputs = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
all_classes = df["variety"].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7,random_state=292567)

dtc = tree.DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
score = dtc.score(test_inputs, test_classes)

print(f"Wynik: {score}")

# dot_data = tree.export_graphviz(dtc, 
#                                feature_names=["sepal.length", "sepal.width", "petal.length", "petal.width"],
#                                class_names=["Setosa", "Versicolor", "Virginica"],
#                                filled=True,    # kolorowanie węzłów
#                                rounded=True,   # zaokrąglone krawędzie
#                                special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("iris_decision_tree", format="png", view=True) 

y_pred = dtc.predict(test_inputs)

disp = ConfusionMatrixDisplay.from_predictions(
    test_classes, 
    y_pred,
    display_labels=["Setosa", "Versicolor", "Virginica"]
)
plt.title('Confusion Matrix')
# plt.show()