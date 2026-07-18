# ==========================================================
# CUSTOMER CHURN PREDICTION — OFFLINE TRAINING SCRIPT
#
# IMPORTANT: This script is NOT part of the deployed app.
# Run it locally / once, BEFORE deploying app.py, to generate:
#   - data/customer_churn_clean.csv
#   - images/*.png              (EDA charts)
#   - models/churn_model.pkl
#   - models/scaler.pkl
#   - models/metrics.json       (accuracy, precision, recall, f1)
#   - models/feature_importance.csv
#
# app.py never imports or executes this file. Deployment should
# only ever point at app.py. This script is wrapped in a
# `if __name__ == "__main__":` guard so it can never fire by
# accident if something imports it as a module.
#
#   Usage:  python train.py
# ==========================================================

import os
import json
import datetime

import pandas as pd
import numpy as np


# ==========================================================
# PART 1 — LOAD & CLEAN
# ==========================================================

def load_and_clean_data(path="data/customer_churn.csv"):

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    try:
        df = pd.read_csv(path)
        print(f"\nDataset loaded successfully from '{path}'.")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        raise SystemExit(1)

    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    df = df.drop_duplicates()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_before = df.isnull().sum().sum()
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"Missing values filled (numeric median): {missing_before} cells")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/customer_churn_clean.csv", index=False)
    print("Clean dataset saved to 'data/customer_churn_clean.csv'")
    print("PART 1 COMPLETE\n")

    return df


# ==========================================================
# PART 2 — EXPLORATORY DATA ANALYSIS
# ==========================================================

def run_eda(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    os.makedirs("images", exist_ok=True)
    plt.style.use("ggplot")
    sns.set_theme()

    plots = [
        ("Churn", None, "Customer Churn Distribution", "churn_distribution.png"),
        ("gender", "Churn", "Gender vs Churn", "gender_vs_churn.png"),
        ("SeniorCitizen", "Churn", "Senior Citizen vs Churn", "senior_vs_churn.png"),
        ("Partner", "Churn", "Partner vs Churn", "partner_vs_churn.png"),
        ("Dependents", "Churn", "Dependents vs Churn", "dependents_vs_churn.png"),
        ("Contract", "Churn", "Contract Type vs Churn", "contract_vs_churn.png"),
        ("InternetService", "Churn", "Internet Service vs Churn", "internet_vs_churn.png"),
    ]

    for x, hue, title, filename in plots:
        plt.figure(figsize=(7, 5))
        if hue:
            sns.countplot(data=df, x=x, hue=hue)
        else:
            sns.countplot(data=df, x=x)
        plt.title(title)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"images/{filename}")
        plt.close()

    for col, filename, title in [
        ("MonthlyCharges", "monthly_charges_distribution.png", "Monthly Charges Distribution"),
        ("tenure", "tenure_distribution.png", "Tenure Distribution"),
    ]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"images/{filename}")
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.savefig("images/monthlycharges_boxplot.png")
    plt.close()

    df_corr = df.copy()
    if "customerID" in df_corr.columns:
        df_corr = df_corr.drop(columns=["customerID"])
    le = LabelEncoder()
    for col in df_corr.columns:
        if not pd.api.types.is_numeric_dtype(df_corr[col]):
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))

    plt.figure(figsize=(15, 10))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png")
    plt.close()

    print(f"Saved {len(plots) + 3} charts to 'images/'")
    print("PART 2 COMPLETE\n")


# ==========================================================
# PART 3 — ENCODING, MODEL COMPARISON, TRAINING, SAVING
# ==========================================================

# Single source of truth for categorical encoding.
# app.py uses the EXACT same mapping (COLUMN_ENCODINGS) so a
# prediction typed into the app lines up with how the model
# was trained. If you change these, update app.py too.
ENCODE_MAPS = {
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

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def train_and_save_model(df):

    import joblib
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    df_model = df.copy()

    if "customerID" in df_model.columns:
        df_model = df_model.drop(columns=["customerID"])

    for col, mapping in ENCODE_MAPS.items():
        if col in df_model.columns:
            df_model[col] = df_model[col].map(mapping)

    before = len(df_model)
    df_model = df_model.dropna()
    after = len(df_model)
    if before != after:
        print(f"Dropped {before - after} rows with unrecognized category values.")

    X = df_model[FEATURE_COLUMNS]
    y = df_model["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------
    # Compare a few candidate models via cross-validation and
    # keep the best one, instead of training a single fixed
    # model blindly.
    # ------------------------------------------------------

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        ),
    }

    print("\nComparing candidate models (5-fold cross-validation)...")

    best_name, best_model, best_score = None, None, -1.0

    for name, candidate in candidates.items():
        scores = cross_val_score(candidate, X_train_scaled, y_train, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        print(f"  {name:<20} CV Accuracy: {mean_score:.4f}")
        if mean_score > best_score:
            best_name, best_model, best_score = name, candidate, mean_score

    print(f"\nBest model: {best_name} (CV Accuracy: {best_score:.4f})")

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        "model_name": best_name,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": int(len(df_model)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_accuracy": round(float(best_score), 4),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\nTest Set Performance:")
    print(f"  Accuracy  : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score  : {metrics['f1_score'] * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ------------------------------------------------------
    # Feature importance (only available for tree-based models)
    # ------------------------------------------------------

    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
    elif hasattr(best_model, "coef_"):
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": np.abs(best_model.coef_[0])
        }).sort_values("importance", ascending=False)
    else:
        importance_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": 0})

    # ------------------------------------------------------
    # Save everything the app needs
    # ------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importance_df.to_csv("models/feature_importance.csv", index=False)

    print("\nSaved:")
    print("  models/churn_model.pkl")
    print("  models/scaler.pkl")
    print("  models/metrics.json")
    print("  models/feature_importance.csv")
    print("\nPART 3 COMPLETE")


# ==========================================================
# ENTRY POINT
# ==========================================================
# Guarded so nothing runs if this file is ever imported rather
# than executed directly. Deployment (streamlit run app.py)
# never triggers this file at all.

if __name__ == "__main__":

    df = load_and_clean_data()
    run_eda(df)
    train_and_save_model(df)

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE FINISHED — ready to run: streamlit run app.py")
    print("=" * 60)