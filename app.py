
import numpy as np
import pandas as pd
import streamlit as st
from src.predict import predict_customer, rank_customers

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write("Predict the probability of customer churn.")

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        result, probability = predict_customer(customer)

        from src.predict import load_model

        model_pipeline, threshold = load_model()

        # Get the preprocessor and Logistic Regression model
        preprocessor = model_pipeline.named_steps["preprocessor"]
        classifier = model_pipeline.named_steps["model"]

        # Transform this customer using the fitted preprocessor
        customer_df = pd.DataFrame([customer])
        X_transformed = preprocessor.transform(customer_df)

        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Get feature names and model coefficients
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        # Calculate each feature's contribution to the model score
        contributions = X_transformed[0] * coefficients

        # Keep the three strongest non-zero contributions
        nonzero_indices = np.flatnonzero(
            np.abs(contributions) > 1e-8
        )

        top_indices = sorted(
            nonzero_indices,
            key=lambda i: abs(contributions[i]),
            reverse=True
        )[:3]

        # Display the strongest contributing features
        st.subheader("Top 3 Model Factors")

        for i in top_indices:
            feature = feature_names[i]
            contribution = contributions[i]

            direction = (
                "Pushes toward churn"
                if contribution > 0
                else "Pushes away from churn"
            )

            feature_labels = {
                "num__tenure": "Customer tenure",
                "num__MonthlyCharges": "Monthly charges",
                "num__TotalCharges": "Total charges",
                "num__SeniorCitizen": "Senior citizen status",
                "cat__Contract_Month-to-month": "Monthly contract",
                "cat__Contract_One year": "One-year contract",
                "cat__Contract_Two year": "Two-year contract",
                "cat__InternetService_DSL": "DSL internet service",
                "cat__InternetService_Fiber optic": "Fiber optic internet",
                "cat__InternetService_No": "No internet service",
                "cat__TechSupport_No": "No tech support",
                "cat__TechSupport_Yes": "Has tech support",
                "cat__PaymentMethod_Electronic check": "Electronic check payment",
            }

            readable_name = feature_labels.get(
                feature,
                feature.replace("num__", "").replace(
                    "cat__", "").replace("_", " ")
            )

            st.write(f"**{readable_name}**")
            st.caption(direction)

        churn_probability = float(probability[0])

        st.subheader("Prediction Results")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

        if result == "Churn":
            st.warning("The customer is flagged for retention.")
        else:
            st.success("The customer is not flagged for retention.")

        st.progress(churn_probability)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# =====================================
# Bulk Customer Prediction from CSV
# =====================================

st.divider()

st.header("Bulk Customer Prediction")

st.write(
    "Upload a CSV file to predict and rank "
    "customers by churn risk."
)

uploaded_file = st.file_uploader(
    "Upload customers CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        customers_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Customers")
        st.dataframe(customers_df.head())

        if st.button("Rank Customers by Risk"):

            results = rank_customers(customers_df)

            st.subheader("Ranked Customer Risk List")

            st.dataframe(
                results,
                use_container_width=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="Download Results CSV",
                data=csv_data,
                file_name="customer_churn_risk.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"CSV prediction failed: {error}")
