import math
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"


def load_model():
    saved_artifact = joblib.load(MODEL_PATH)

    model_pipeline = saved_artifact["model"]
    threshold = saved_artifact["threshold"]

    return model_pipeline, threshold


def predict_customer(customer_data):
    # Load the saved model
    model_pipeline, threshold = load_model()

    # Convert input into a DataFrame
    if isinstance(customer_data, dict):
        results_df = pd.DataFrame([customer_data])
    else:
        results_df = customer_data.copy()

    # Predict churn probability
    probability = model_pipeline.predict_proba(results_df)[:, 1]

    # Apply decision threshold
    offer = (probability >= threshold).astype(int)
    result = np.where(offer == 1, "Churn", "No Churn")

    print("Prediction:", result)
    print("probability:", probability.round(2))

    # for prob in probability:
    # print((prob * 100).round(2), "%")

    results_df["Prediction"] = result
    results_df["Churn Probability"] = probability.round(2)

    return results_df


def rank_customers(customers_df):
    """
    Rank current customers by predicted churn probability.
    Actual Lift requires known churn outcomes and is not
    calculated by this function.
    """

    model_pipeline, threshold = load_model()

    results = customers_df.copy()

    # Predict churn probability for every customer
    results["churn_probability"] = (
        model_pipeline.predict_proba(customers_df)[:, 1]
    )

    # Apply the saved threshold
    results["predicted_churn"] = (
        results["churn_probability"] >= threshold
    ).astype(int)

    # Rank customers from highest to lowest risk
    results = results.sort_values(
        "churn_probability",
        ascending=False
    ).reset_index(drop=True)

    n = len(results)

    # Mark the cumulative top 10% and top 20%
    n_top_10 = math.ceil(n * 0.10)
    n_top_20 = math.ceil(n * 0.20)

    results["risk_group"] = "Other"

    results.loc[:n_top_20 - 1, "risk_group"] = "Top 20%"
    results.loc[:n_top_10 - 1, "risk_group"] = "Top 10%"

    return results


if __name__ == "__main__":

    # 1. Load new customers from CSV
    input_path = ROOT_DIR / "data" / "new_customers.csv"
    customers_df = pd.read_csv(input_path)

    # 2. Remove target column if it exists
    customers_df = customers_df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    # Predict churn for each customer
    result_predict = predict_customer(customers_df)

    # 3. Rank customers
    results = rank_customers(customers_df)

    # 4. Save results
    output_path = ROOT_DIR / "data" / "customer_risk_results.csv"
    results.to_csv(output_path, index=False)

    output_path_predict = ROOT_DIR / "data" / "customer_predict_results.csv"
    result_predict.to_csv(output_path_predict, index=False)

    # 5. Display important columns
    print(results[
        ["churn_probability", "predicted_churn", "risk_group"]
    ])

    print(f"\nResults saved to: {output_path}")
