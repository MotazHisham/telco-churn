# Model Card: Customer Churn Prediction

## 1. Intended Use

### Purpose
This model is designed to predict the probability of customer churn in a telecommunications company.

### Intended Users
The model is intended for use by:
- Customer retention teams.
- Business analysts.
- Customer relationship management (CRM) teams.

### Intended Applications
- Identify customers who are at high risk of leaving the company.
- Prioritize customers for retention campaigns.
- Support data-driven business decisions.
- Help allocate retention resources more effectively.

### Out-of-Scope Uses
The model should not be used as the sole basis for automatically terminating customer accounts or making decisions that negatively affect customers.

---

## 2. Data

### Dataset
- **Dataset Name:** Telco Customer Churn.
- **Total Records:** 7,043 customers.
- **Target Variable:** `Churn`.
- **Problem Type:** Binary Classification.

### Features
The dataset contains customer information, including:

- **Demographics:** Gender, SeniorCitizen, Partner, Dependents.
- **Account Information:** Tenure, Contract, PaperlessBilling, PaymentMethod.
- **Services:** InternetService, OnlineSecurity, TechSupport, StreamingTV, and other subscribed services.
- **Charges:** MonthlyCharges and TotalCharges.

### Target Variable
The `Churn` column indicates whether a customer left the company:
- `Yes` = 1 (Customer churned).
- `No` = 0 (Customer stayed).

### Data Preprocessing
The following preprocessing steps were applied:
- Removed `customerID` because it is an identifier rather than a predictive feature.
- Converted `TotalCharges` to numeric format.
- Handled missing values in `TotalCharges`.
- Converted the target variable into binary values (0 and 1).
- Applied One-Hot Encoding to categorical features.
- Applied Standard Scaling to numerical features.
- Used a preprocessing pipeline to ensure consistent transformations.

### Train-Test Split
- Training set: 80%.
- Testing set: 20%.
- Random State: 42.
- Stratification: Applied using the target variable to preserve class proportions.

---

## 3. Metrics

### Evaluation Results

The model was evaluated on a held-out test set using the following metrics:

| Metric | Score |
|---|---:|
| Accuracy  | 53.51% |
| Precision | 36.13% |
| Recall    | 97.86% |
| F1-score  | 52.78% |
| ROC-AUC   | 84.02% |

### Decision Threshold
- **Selected Threshold:** 0.12.
- The threshold was selected using validation data based on business profit optimization.
- The test set was kept separate for final model evaluation.

### Business Profit Results

- **Selected Threshold:** 0.12.
- **Selection Method:** Business profit optimization using validation data.
- **Best Estimated Profit:** 209,940.10 (business profit units).
- **Evaluation Dataset:** Validation set.

The estimated profit is based on the predefined retention campaign cost and customer lifetime value assumptions. It represents an estimated business outcome, not guaranteed actual revenue.

### Business Profit Assumptions

The expected business profit was estimated using the following assumptions:

- Estimated customer lifetime value: 12 months of MonthlyCharges.
- Retention offer cost: 1 month of MonthlyCharges.
- Successfully retained customer: +11 months of MonthlyCharges.
- Unnecessary retention offer: -1 month of MonthlyCharges.
- Lost customer without a retention offer: -12 months of MonthlyCharges.

These assumptions were used to estimate expected profit and select the decision threshold.

### Performance Interpretation
- High Recall (97.86%) indicates that the model identifies most customers who actually churn.
- Precision (36.13%) indicates that some customers flagged as potential churners will not actually leave.
- ROC-AUC (84.02%) indicates good ability to distinguish between churners and non-churners across different thresholds.

---

## 4. Limitations

### Dataset Limitations
The model was trained on a specific telecommunications dataset. Its performance may vary when applied to customers from different companies, markets, or business environments.

### False Positives
The selected decision threshold (0.12) prioritizes high recall, which may result in a relatively high number of false positives. Some customers identified as high-risk may not actually churn.

### Historical Data
The model relies on historical customer behavior. Changes in market conditions, pricing strategies, or customer preferences may reduce its predictive performance over time.

### No Causal Interpretation
The model identifies statistical patterns and relationships in customer data but does not establish causal relationships between customer characteristics and churn behavior.

### Business Assumptions
The estimated business profit depends on predefined assumptions about retention campaign costs and the financial impact of losing customers. Actual business outcomes may differ.

### Model Monitoring
The model should be periodically evaluated using new customer data to detect performance degradation and changes in customer behavior.

