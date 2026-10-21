import math
from pathlib import Path
import joblib
import pandas as pd

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

    # Convert one customer to a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Predict churn probability
    probability = model_pipeline.predict_proba(customer_df)[:, 1]

    # Apply decision threshold
    offer = int(probability >= threshold)

    result = "Churn" if offer == 1 else "No Churn"

    print("Prediction:", result)

    for prob in probability:
        print((prob * 100).round(2), "%")

    return result, probability


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

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }

    predict_customer(customer)
