"""
===========================================================
Customer Churn Prediction
Model Training
===========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


print("=" * 60)
print("MODEL TRAINING STARTED")
print("=" * 60)

# Load processed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Models
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC()
}

best_accuracy = 0
best_model = None
best_name = ""

print("\nAccuracy Results\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    print(f"{name:<25} : {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_name = name


os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/churn_model.pkl"
)

print("\n" + "=" * 60)
print("BEST MODEL :", best_name)
print("BEST ACCURACY :", round(best_accuracy,4))
print("=" * 60)

print("\nModel Saved Successfully")