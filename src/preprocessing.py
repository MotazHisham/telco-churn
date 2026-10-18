import numpy as np
import pandas as pd


def load_and_clean_data(file_path):
    # 1. Load the dataset
    df = pd.read_csv(file_path)

    # 2. Remove unnecessary customer identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # 3. Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # 4. Handle missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["tenure"] * df["MonthlyCharges"]
    )

    # 5. Map categorical variables to numerical values
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 6. Remove rows with missing target
    df = df.dropna(subset=["Churn"])

    df["Churn"] = df["Churn"].astype(int)

    return df
