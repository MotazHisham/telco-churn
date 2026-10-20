from pathlib import Path
import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"

saved_artifact = joblib.load(MODEL_PATH)

model_pipeline = saved_artifact["model"]
threshold = saved_artifact["threshold"]


def predict_customer(customer_data):

    # Convert customer data to a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Load the saved pipeline
    saved_artifact = joblib.load(MODEL_PATH)
    model_pipeline = saved_artifact["model"]
    threshold = saved_artifact["threshold"]

    # Predict churn probability using the loaded model
    probability = model_pipeline.predict_proba(customer_df)[:, 1]

    # Convert probability to prediction
    offer = (probability >= threshold).astype(int)

    result = "Churn" if offer == 1 else "No Churn"

    print("Prediction:", result)

    for prob in probability:
        print((prob * 100).round(2), "%")

    return result, probability


if __name__ == "__main__":
    # Example customer
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
