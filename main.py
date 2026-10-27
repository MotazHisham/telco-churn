
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.predict import predict_customer, rank_customers

app = FastAPI(title="Customer Churn API")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}


@app.post("/predict")
def predict(customers: List[Customer]):
    customers_data = [
        customer.model_dump()
        for customer in customers
    ]

    results_df = predict_customer(
        pd.DataFrame(customers_data)
    )

    return results_df.to_dict(orient="records")


@app.post("/predict/bulk")
def predict_bulk(customers: List[Customer]):
    try:
        if not customers:
            raise HTTPException(
                status_code=400,
                detail="The customer list is empty."
            )

        customers_data = [
            customer.model_dump()
            for customer in customers
        ]

        customers_df = pd.DataFrame(customers_data)

        results = rank_customers(customers_df)

        # Replace missing values with JSON-compatible nulls
        results = results.astype(object).where(
            pd.notna(results),
            None
        )

        return results.to_dict(orient="records")

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
