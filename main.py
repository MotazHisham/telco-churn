
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from src.predict import predict_customer

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
