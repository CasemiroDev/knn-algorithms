import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("knn - classifier/database.csv")

# Normalmente o X são os Inputs e os Y são os Outputs
X = df.drop('risco', axis=1)
y = df.risco

# Avaliando o Classificador e aplicando a predição
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(accuracy_score(y_test,knn.predict(X_test)))
