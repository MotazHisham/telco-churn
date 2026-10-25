# Telco Customer Churn Prediction

A machine learning project that estimates which telecommunications
customers are likely to leave (churn). The goal is to turn customer data
into useful retention insights, rank customers by churn risk, and
explore how a retention offer could affect expected business profit.

## 1. Problem

Customer churn can reduce recurring revenue and increase the need to
acquire replacement customers. A churn model can help a business
identify customers who may be at risk and prioritize them for further
investigation or retention outreach.

This is a **prediction and decision-support project**. A high predicted
probability does not prove that a customer will leave, and contacting a
customer does not guarantee that they will stay.

## 2. Data

The project uses the Telco Customer Churn dataset. It contains customer
information about tenure, services, contracts, payment methods, and
charges.

The target column is `Churn`: - `Yes` / `1`: the customer churned. -
`No` / `0`: the customer did not churn.

Examples of columns:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `customerID`                        Customer identifier; excluded from
                                      model features because it
                                      identifies the record rather than
                                      describing customer behavior.

  `tenure`                            Number of months the customer has
                                      been with the company.

  `MonthlyCharges`                    Current monthly service charges for
                                      the customer.

  `TotalCharges`                      Total charges accumulated over the
                                      customer's service period.

  `Contract`                          Contract type, such as
                                      month-to-month, one-year, or
                                      two-year.

  `InternetService`                   Type of internet service, if any.

  `PaymentMethod`                     Method used to pay the bill.

  `TechSupport`                       Whether the customer has technical
                                      support service.

  `SeniorCitizen`                     Dataset indicator for whether the
                                      customer is a senior citizen.

  `Churn`                             Target label indicating whether the
                                      customer left the company.
  -----------------------------------------------------------------------

The dataset's original source and usage terms should be documented
before public distribution. The raw dataset is expected at
`data/raw/Telco-Customer-Churn.csv`; data files may be excluded from Git
by `.gitignore`.

## 3. Approach

The workflow is organized into these stages:

1.  **Inspect and clean data** --- check column types and missing
    values, convert `TotalCharges` to numeric, handle invalid or missing
    values, and remove `customerID` from model inputs.
2.  **Explore customer behavior (EDA)** --- compare churn rates by
    contract, tenure, monthly charges, internet service, payment method,
    and technical support.
3.  **Prepare features** --- use a preprocessing pipeline for numerical
    and categorical columns. Fit preprocessing only on training data and
    apply the fitted transformations to validation/test data.
4.  **Split the data** --- use a stratified train/test split so the
    churn proportion is approximately preserved.
5.  **Train models** --- Logistic Regression is the baseline model.
    SMOTE and class-weighted training can be evaluated as alternatives
    for class imbalance. Any resampling must happen only inside the
    training workflow, never on validation or test data.
6.  **Evaluate predictions** --- examine precision, recall, F1-score,
    ROC-AUC, confusion matrix, and, where useful, PR-AUC.
7.  **Choose a decision threshold** --- Instead of relying on the
    default classification threshold of 0.50, the decision threshold was selected using a business profit function on the validation set.
    The business assumptions were:
    Retention offer cost: one month of the customer's charges.
    Revenue loss from a churned customer: twelve months of the customer's charges.
    Under these assumptions, a threshold of 0.12 maximized the estimated validation profit.
    Business Trade-off
    The lower threshold prioritizes identifying as many potential churners as possible. This increases recall but also results in more false-positive predictions and retention offers to customers who might have stayed
    A higher threshold, such as 0.70, produces higher precision and accuracy but misses more customers who eventually churn.
    Therefore, the threshold of 0.12 was selected for the profit-oriented retention strategy, subject to validation of the business assumptions
8.  **Translate scores into action** --- rank customers by predicted
    churn probability and evaluate retention targeting assumptions.

## 4. Results

### EDA observations

The current analysis has observed these patterns in the dataset:

-   Month-to-month customers have a higher observed churn rate than
    customers on longer contracts.
-   Customers with shorter tenure, particularly in the first 12 months,
    have higher observed churn.
-   In the analyzed data, mean `MonthlyCharges` was 74.44 for churned
    customers and 61.27 for customers who did not churn.
-   Fiber optic customers have a higher observed churn rate.
-   Electronic check users have the highest observed churn rate among
    the payment methods compared.
-   Customers without technical support have a higher observed churn
    rate.

These are associations in the dataset, not proof that any one factor
causes churn.

### Model metrics

Record the metrics from the **final, held-out test evaluation** here
after the final model and threshold have been selected. Do not use
validation metrics as if they were test results.

  Metric                           Final test result
  ----------- --------------------------------------
  Accuracy      53.51%
  ROC-AUC       84.02%
  Precision     36.13%
  Recall        97.86%
  F1-score      52.78%

If multiple models are compared, include the same metrics for each model
and state which model was selected. Evaluate the test set only after
model selection is complete.

## 5. Business impact

The model can support a retention workflow by:

-   Ranking customers from highest to lowest predicted churn risk.
-   Helping a retention team focus on a limited group of high-risk
    customers.
-   Comparing targeting strategies, such as contacting the top 10% or
    20% of scored customers.
-   Estimating the expected value of offers under explicit assumptions
    about offer cost and the revenue lost when a customer churns.

For the current illustrative business calculation, an offer costs one
month's `MonthlyCharges`, while a lost customer is assumed to represent
12 months of charges. These are assumptions for analysis, not verified
company economics. The calculation also needs to account for whether an
offer actually prevents churn; a churn probability alone cannot
establish that effect.

Business value should be validated with real campaign outcomes,
including offer acceptance, prevented churn, and campaign costs.

## 6. How to run

The exact commands depend on the files currently present in the
repository. The steps below assume the project uses `uv` and that
`pyproject.toml` is in the repository root.

### Prerequisites

-   Python version compatible with `pyproject.toml`
-   `uv`
-   The Telco Customer Churn CSV in `data/raw/`, if the workflow expects
    a local dataset

### Setup on Windows PowerShell

``` powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd telco-churn
uv sync
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual repository URL.

### Run the notebooks

Open the repository in VS Code or Jupyter and run the notebooks in the
intended order. Restart the kernel and use **Run All** to check that the
workflow runs from a clean session.

### Run prediction

The prediction script is intended to score new customer records without
requiring a `Churn` column. Check the current `src/predict.py` interface
and required input columns before running it; use the actual command or
function signature implemented in the repository.

Do not commit private customer data, raw data that is not licensed for
redistribution, or local model artifacts unless the repository is
intended to include them.

## 7. Limitations

-   Model performance depends on the dataset and may not transfer to
    another company or time period.
-   Observed relationships do not establish causation.
-   Missing-value handling and feature engineering introduce assumptions
    that should be documented and tested.
-   SMOTE can change the training distribution; it must not be applied
    to validation or test data.
-   The classification threshold reflects a chosen objective and
    business assumptions; it is not universally optimal.
-   The offer-cost and lost-revenue figures are illustrative and should
    be replaced with verified business inputs.
-   Predicted churn risk does not show that a retention offer will
    prevent churn. That requires campaign or causal evidence.
-   Final test metrics must be added after the final evaluation has been
    run.

## Project structure

The repository may evolve. Update this tree to match the files that are
actually committed:

``` text
telco-churn/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
├── reports/
│   └── figures/
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Author

**Motaz Saleh**

Machine learning project focused on customer churn prediction, model
evaluation, and business-oriented retention analysis.
