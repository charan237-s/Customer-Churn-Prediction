# ==========================================================
# CUSTOMER CHURN PREDICTION
# PART 1
# Data Loading & Data Cleaning
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# STEP 1 : Load Dataset
# ----------------------------------------------------------

print("=" * 60)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 60)

try:
    df = pd.read_csv("data/customer_churn.csv")
    print("\nDataset Loaded Successfully.\n")
except Exception as e:
    print("Dataset Loading Failed")
    print(e)
    exit()

# ----------------------------------------------------------
# STEP 2 : Display First Rows
# ----------------------------------------------------------

print("=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())

# ----------------------------------------------------------
# STEP 3 : Shape
# ----------------------------------------------------------

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)

rows, cols = df.shape

print(f"Rows    : {rows}")
print(f"Columns : {cols}")

# ----------------------------------------------------------
# STEP 4 : Column Names
# ----------------------------------------------------------

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for column in df.columns:
    print(column)

# ----------------------------------------------------------
# STEP 5 : Dataset Information
# ----------------------------------------------------------

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

# ----------------------------------------------------------
# STEP 6 : Statistical Summary
# ----------------------------------------------------------

print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

# ----------------------------------------------------------
# STEP 7 : Missing Values
# ----------------------------------------------------------

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

# ----------------------------------------------------------
# STEP 8 : Duplicate Records
# ----------------------------------------------------------

print("=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate Rows :", duplicates)

# ----------------------------------------------------------
# STEP 9 : Remove Duplicates
# ----------------------------------------------------------

df = df.drop_duplicates()

print("\nDuplicates Removed Successfully")

print("Current Shape :", df.shape)

# ----------------------------------------------------------
# STEP 10 : Check Data Types
# ----------------------------------------------------------

print("=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# ----------------------------------------------------------
# STEP 11 : Unique Values
# ----------------------------------------------------------

print("=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for col in df.columns:

    print("\n-------------------------------------")
    print(col)
    print("-------------------------------------")

    print(df[col].nunique())

# ----------------------------------------------------------
# STEP 12 : Target Variable
# ----------------------------------------------------------

print("=" * 60)
print("TARGET VARIABLE")
print("=" * 60)

print(df["Churn"].value_counts())

# ----------------------------------------------------------
# STEP 13 : Convert TotalCharges
# ----------------------------------------------------------

if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# ----------------------------------------------------------
# STEP 14 : Fill Missing Values
# ----------------------------------------------------------

df.fillna(df.median(numeric_only=True), inplace=True)

# ----------------------------------------------------------
# STEP 15 : Final Check
# ----------------------------------------------------------

print("=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(df.head())

print()

print(df.shape)

# ----------------------------------------------------------
# STEP 16 : Save Clean Dataset
# ----------------------------------------------------------

df.to_csv(
    "data/customer_churn_clean.csv",
    index=False
)

print("\nClean Dataset Saved Successfully.")

print("\nPART 1 COMPLETED SUCCESSFULLY")

# ==========================================================
# PART 2
# Exploratory Data Analysis (EDA)
# ==========================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

# Better graph style
plt.style.use("ggplot")
sns.set_theme()

print("=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["Churn"].value_counts())

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")

plt.savefig("images/churn_distribution.png")

plt.close()

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="gender",
    hue="Churn"
)

plt.title("Gender vs Churn")

plt.savefig("images/gender_vs_churn.png")

plt.close()

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn"
)

plt.title("Senior Citizen vs Churn")

plt.savefig("images/senior_vs_churn.png")

plt.close()

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Partner",
    hue="Churn"
)

plt.title("Partner vs Churn")

plt.savefig("images/partner_vs_churn.png")

plt.close()

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Dependents",
    hue="Churn"
)

plt.title("Dependents vs Churn")

plt.savefig("images/dependents_vs_churn.png")

plt.close()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Churn")

plt.xticks(rotation=15)

plt.savefig("images/contract_vs_churn.png")

plt.close()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Internet Service vs Churn")

plt.xticks(rotation=20)

plt.savefig("images/internet_vs_churn.png")

plt.close()

plt.figure(figsize=(8,5))

sns.histplot(
    df["MonthlyCharges"],
    bins=30,
    kde=True
)

plt.title("Monthly Charges Distribution")

plt.savefig("images/monthly_charges_distribution.png")

plt.close()

plt.figure(figsize=(8,5))

sns.histplot(
    df["tenure"],
    bins=30,
    kde=True
)

plt.title("Tenure Distribution")

plt.savefig("images/tenure_distribution.png")

plt.close()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")

plt.savefig("images/monthlycharges_boxplot.png")

plt.close()

# Convert categorical columns to numeric temporarily (for correlation plot only)
df_corr = df.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df_corr.columns:
    if df_corr[col].dtype == "object":
        df_corr[col] = le.fit_transform(df_corr[col])

plt.figure(figsize=(15,10))

sns.heatmap(
    df_corr.select_dtypes(include=['number']).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("images/correlation_heatmap.png")

plt.close()

print("=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

print()

print("1. Customers with Month-to-Month contracts usually have higher churn.")
print("2. Customers with longer tenure generally have lower churn.")
print("3. High monthly charges may increase churn risk.")
print("4. Senior citizens can have different churn behavior.")
print("5. Internet service type may influence churn.")
print("\n" + "=" * 60)
print("PART 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ==========================================================
# PART 3
# Feature Encoding, Model Training & Saving
# (This is what app.py needs: models/churn_model.pkl + models/scaler.pkl)
# ==========================================================

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "=" * 60)
print("PART 3: MODEL TRAINING")
print("=" * 60)

df_model = df.copy()

# Drop identifier column if present - not a predictive feature
if "customerID" in df_model.columns:
    df_model = df_model.drop(columns=["customerID"])

# ----------------------------------------------------------
# STEP 1 : Encode categorical columns
# IMPORTANT: these mappings must exactly match the `maps` dict
# used in app.py, so predictions made in the app line up with
# how the model was trained.
# ----------------------------------------------------------

encode_maps = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    },
    "Churn": {"No": 0, "Yes": 1}
}

for col, mapping in encode_maps.items():
    if col in df_model.columns:
        df_model[col] = df_model[col].map(mapping)

# Any values that failed to map (unexpected categories) become NaN - drop them
before = len(df_model)
df_model = df_model.dropna()
after = len(df_model)
if before != after:
    print(f"Dropped {before - after} rows with unrecognized category values.")

# ----------------------------------------------------------
# STEP 2 : Split features / target
# Feature order matches FEATURE_COLUMNS in app.py exactly.
# ----------------------------------------------------------

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

X = df_model[FEATURE_COLUMNS]
y = df_model["Churn"].astype(int)

# ----------------------------------------------------------
# STEP 3 : Train / test split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# STEP 4 : Scale features
# ----------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# STEP 5 : Train model
# ----------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------
# STEP 6 : Evaluate
# ----------------------------------------------------------

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------------------------
# STEP 7 : Save model + scaler for app.py
# ----------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved to 'models/churn_model.pkl' and 'models/scaler.pkl'")
print("\nPART 3 COMPLETED SUCCESSFULLY")