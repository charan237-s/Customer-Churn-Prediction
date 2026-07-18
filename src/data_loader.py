"""
===========================================================
Customer Churn Prediction
Data Loading & Cleaning
===========================================================
"""

import os
import pandas as pd


class DataLoader:

    def __init__(self, file_path):

        self.file_path = file_path

    def load_data(self):

        try:

            df = pd.read_csv(self.file_path)

            print("=" * 60)
            print("DATASET LOADED SUCCESSFULLY")
            print("=" * 60)

            return df

        except Exception as e:

            print(e)

            return None

    def dataset_information(self, df):

        print("\nShape")
        print(df.shape)

        print("\nColumns")
        print(df.columns.tolist())

        print("\nInfo")
        print(df.info())

        print("\nStatistics")
        print(df.describe(include="all"))

    def check_missing_values(self, df):

        print("\nMissing Values")

        print(df.isnull().sum())

    def remove_duplicates(self, df):

        duplicate_rows = df.duplicated().sum()

        print("\nDuplicate Rows :", duplicate_rows)

        df = df.drop_duplicates()

        return df

    def convert_total_charges(self, df):

        if "TotalCharges" in df.columns:

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

        return df

    def fill_missing_values(self, df):

        numeric_columns = df.select_dtypes(include=["number"]).columns

        for col in numeric_columns:

            df[col].fillna(df[col].median(), inplace=True)

        categorical_columns = df.select_dtypes(include=["object"]).columns

        for col in categorical_columns:

            df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def save_clean_dataset(self, df):

        os.makedirs("data", exist_ok=True)

        df.to_csv(
            "data/customer_churn_clean.csv",
            index=False
        )

        print("\nClean Dataset Saved")


def main():

    loader = DataLoader("data/customer_churn.csv")

    df = loader.load_data()

    if df is None:

        return

    loader.dataset_information(df)

    loader.check_missing_values(df)

    df = loader.remove_duplicates(df)

    df = loader.convert_total_charges(df)

    df = loader.fill_missing_values(df)

    loader.save_clean_dataset(df)

    print("\nData Loading Completed Successfully")


if __name__ == "__main__":

    main()