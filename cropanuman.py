# Importing Libraries
import os

import joblib
import numpy as np
import pandas as pd

file_name = "cp_data.csv"

# Importing Dataset
df = pd.read_csv(os.path.join(os.getcwd(), file_name))

print(df.head())


# Label Encoding the target variable
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
df["label"] = LE.fit_transform(df["label"])

print(LE.classes_)

# Separating attributes and target
X = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values


# Splitting the data into training and test set(20% data in test set)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Standard Scaling the attribute values
from sklearn.preprocessing import StandardScaler

S = StandardScaler()
X_train = S.fit_transform(X_train)
X_test = S.transform(X_test)


# Importing the KNN Model
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(
    n_neighbors=5, weights="uniform", algorithm="auto", p=3, metric="minkowski"
)


from sklearn.tree import DecisionTreeClassifier

DC = DecisionTreeClassifier(criterion="gini")


# Fitting data to the model
Knn.fit(X_train, y_train)
DC.fit(X_train, y_train)


# Predicting Test Data
y_pred1 = Knn.predict(X_test)
y_pred2 = DC.predict(X_test)


# Saving the model
joblib.dump(Knn, "classifier_knn.pkl")
joblib.dump(DC, "classifier_decision_tree.pkl")

# Calculating metrics
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred1))
print(accuracy_score(y_test, y_pred2))

# Accuracy of KNN model - 0.8774193548387097
# Accuracy of Decsion Tree model -0.9064516129032258


temp = int(input("Enter temperature value: "))
hum = int(input("Enter humidity value: "))
ph = int(input("Enter ph value: "))
rain = int(input("Enter rainfall value: "))
knn_pred = Knn.predict([[temp, hum, ph, rain]])
dc_pred = DC.predict([[temp, hum, ph, rain]])
print(LE.classes_[knn_pred[0]])
print(LE.classes_[knn_pred[0]])
