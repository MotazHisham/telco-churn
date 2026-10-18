# Customer Churn Prediction

A Machine Learning project that analyzes customer data and predicts customer churn using Logistic Regression. The project explores customer behavior, data preprocessing, class imbalance, and classification evaluation to support data-driven customer retention strategies.

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Key Findings](#key-findings)
- [Business Recommendations](#business-recommendations)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Customer churn occurs when a customer stops using a company's services. Predicting churn can help businesses identify customers who may be at risk of leaving and prioritize customer retention efforts.

This project uses the Telco Customer Churn dataset to explore churn patterns and develop a binary classification model.

The main model used in the current workflow is **Logistic Regression**.

## Business Problem

Customer acquisition can be expensive, making customer retention an important business objective.

This project aims to answer the following questions:

- Which customer characteristics are associated with churn?
- Can customer data help identify customers who are likely to leave?
- How does class imbalance affect churn prediction?
- How can model predictions support customer retention strategies?

The model identifies patterns in historical data. These patterns do not necessarily indicate causal relationships.

## Objectives

- Inspect and clean customer data.
- Perform Exploratory Data Analysis (EDA).
- Analyze churn patterns across customer segments.
- Prepare numerical and categorical features for machine learning.
- Handle class imbalance using SMOTE.
- Train and evaluate a Logistic Regression classifier.
- Evaluate performance using multiple classification metrics.
- Translate analytical findings into business recommendations.

## Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer information related to demographics, services, contracts, payment methods, and billing.

Examples of analyzed features include:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `PaymentMethod`
- `TechSupport`
- `SeniorCitizen`

The target variable is `Churn`:

| Value | Meaning |
|---|---|
| `0` | Customer did not churn |
| `1` | Customer churned |

The original dataset source and license should be added here before publishing the repository.

## Technologies Used

- **Python** — Main programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Exploratory data visualization, where used
- **Scikit-learn** — Preprocessing, Logistic Regression, and model evaluation
- **Imbalanced-learn** — SMOTE for class imbalance
- **Joblib** — Saving and loading trained models
- **Jupyter Notebook / Google Colab / VS Code** — Development environments

## Project Structure

The intended project structure is:

```text
Customer_Churn_Project/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── churn_clean.csv
│
├── notebooks/
│   ├── 01_Data_Inspection.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Model_Training_and_Evaluation.ipynb
│
│   
├── src/
│   ├── processing.py
│   
│   

├── models/
│   └── logistic_regression.pkl
│
├── reports/
│   ├── figures/
│   └── final_report.md
│
├── README.md
└── requirements.txt
```

*Note: Adjust the tree to match the files actually present in your repository.*

## Workflow

### 1. Data Inspection

- Inspect dataset dimensions and column types.
- Review missing values and duplicate records.
- Examine target class distribution.

### 2. Data Cleaning

- Convert `TotalCharges` to a numeric data type.
- Handle missing or invalid values.
- Remove the `customerID` identifier.
- Encode selected binary categorical variables.

### 3. Exploratory Data Analysis

Analyze churn patterns across:

- Contract types
- Customer tenure
- Monthly charges
- Internet service types
- Payment methods
- Technical support availability

### 4. Feature Preprocessing

The current workflow uses a `ColumnTransformer` to apply:

- `StandardScaler` to numerical features.
- `OneHotEncoder` to categorical features.

The preprocessor is fitted on the training data and then used to transform the test data.

### 5. Train-Test Split

The data is split into training and testing subsets using an 80:20 ratio, with stratification to preserve approximate target class proportions.

### 6. Class Imbalance

SMOTE is explored to generate synthetic minority-class examples in the training data.

Resampling should only be applied to training data. The test data should retain its original class distribution.

### 7. Model Development

Logistic Regression is used as the classification model.

Current model configuration:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### 8. Model Evaluation

The project evaluates model predictions using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

The current evaluation workflow also explores a classification threshold of `0.7`.

## Exploratory Data Analysis

The current analysis identified the following patterns:

- Month-to-month customers showed higher observed churn rates than customers with longer-term contracts.
- Customers with shorter tenure, especially during the first 12 months, showed higher churn rates.
- The recorded mean monthly charge was `74.44` for churned customers and `61.27` for customers who did not churn.
- Fiber optic customers showed higher observed churn rates.
- Electronic check users showed the highest observed churn rate among the payment methods compared.
- Customers without technical support showed higher observed churn rates.

These are associations observed in the dataset and should not be interpreted as proof of causation.

## Model Evaluation

The evaluation workflow calculates Accuracy, Precision, Recall, F1-score, and ROC-AUC.

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Reliability of positive churn predictions |
| Recall | Proportion of actual churners detected |
| F1-score | Balance between Precision and Recall |
| ROC-AUC | Ranking ability across classification thresholds |

**Final metrics:** Add the actual evaluation results after running the final model evaluation. No final values are reported here because the current evaluation function calculates the metrics without printing or returning them.

The selected threshold of `0.7` classifies a customer as a potential churner when the predicted probability is at least 70%. The threshold should be validated against business objectives and the cost of false-positive and false-negative predictions.

## Key Findings

The analysis suggests that churn is associated with several customer characteristics, including contract type, tenure, monthly charges, internet service, payment method, and technical support.

These findings can help identify customer segments for further investigation. Additional validation is needed before using them to make business decisions.

## Business Recommendations

1. Improve onboarding and customer support during the first year.
2. Investigate the reasons for higher churn among month-to-month customers.
3. Review pricing and perceived value among customers with higher monthly charges.
4. Investigate service quality and customer experience among Fiber optic users.
5. Evaluate whether improved technical support can reduce churn.
6. Investigate billing or payment friction among electronic check users.
7. Use validated model scores to prioritize retention outreach.
8. Measure the impact of retention campaigns before scaling them.

These recommendations are hypotheses for testing, not guaranteed outcomes.

## How to Run

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Customer_Churn_Project
```

Replace the placeholder with your actual repository URL.

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scriptsctivate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the libraries required by the project, including pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn (if used), and joblib.

### 4. Prepare the Dataset

Place the dataset in:

```text
data/raw/Telco-Customer-Churn.csv
```

Update the input path if your dataset is stored elsewhere.

### 5. Run the Workflow

Open the notebooks in the `notebooks/` directory in numerical order, or run the relevant Python scripts from `src/`.

The exact commands depend on the entry points implemented in your repository.

**Important:** The current evaluation function loads a pre-trained model from a local Windows path. Update this path to a project-relative path and ensure the matching model artifact is available before running inference.

## Limitations

- Final test-set metric values still need to be recorded.
- The saved model must be compatible with the preprocessing transformations.
- The current processing function generates SMOTE-balanced training data but does not use it to fit the loaded model.
- Imputation assumptions should be validated.
- A single test split may not fully represent future performance.
- Observed associations do not establish causal relationships.
- The classification threshold requires validation against real business costs.

## Future Improvements

- Add cross-validation and systematic hyperparameter tuning.
- Compare Logistic Regression with additional classification algorithms.
- Evaluate Precision-Recall curves and PR-AUC.
- Select the classification threshold using validation data.
- Save preprocessing and the estimator as one complete pipeline.
- Add automated data-quality and inference checks.
- Save evaluation plots under `reports/figures/`.
- Add reproducible setup instructions and dataset attribution.
- Evaluate retention strategies using actual business outcomes.

## Author

**Motaz Saleh**

Machine Learning project focused on customer churn analysis, classification, and data-driven business insights.

## License

Add an appropriate license before distributing the repository. Confirm that the dataset's terms also permit your intended use.
