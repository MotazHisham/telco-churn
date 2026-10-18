from pathlib import Path
import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"


def predict_customer(customer_data):

    # Convert customer data to a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Load the saved pipeline
    model = joblib.load(MODEL_PATH)

    # Predict churn probability using the loaded model
    probability = model.predict_proba(customer_df)[:, 1]

    # Use the same threshold
    threshold = 0.7

    # Convert probability to prediction
    prediction = (probability >= threshold).astype(int)

    result = "Churn" if prediction == 1 else "No Churn"

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
