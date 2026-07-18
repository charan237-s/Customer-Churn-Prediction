import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess():

    print("=" * 60)
    print("PREPROCESSING STARTED")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv("data/customer_churn_clean.csv")

    # Drop customerID if exists
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df.fillna(0, inplace=True)

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    print("Categorical Columns:")
    print(categorical_columns)

    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    print("\nAfter Encoding:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    # Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # Save processed data
    pd.DataFrame(X_train).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)

    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("\nPreprocessing Completed Successfully")


if __name__ == "__main__":
    preprocess()
