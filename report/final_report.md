# Customer Churn Prediction — Final Report

## 1. Executive Summary

This project focuses on predicting customer churn using the Telco Customer Churn dataset. The objective is to identify customers who are likely to leave a telecommunications service and provide insights that can support customer retention strategies.

The project includes data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing, model training, and model evaluation using Logistic Regression.

The workflow uses numerical feature scaling, categorical feature encoding, and a stratified train-test split. SMOTE was also explored to address class imbalance. Model performance is evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix.

**Note:** Final evaluation metrics must be added after running the complete evaluation workflow. The values should not be assumed or estimated.

## 2. Business Problem

Customer churn is a major challenge for telecommunications companies because losing customers can reduce revenue and increase customer acquisition costs.

The goal of this project is to build a classification model that predicts whether a customer is likely to churn based on available customer and service information.

The target variable is:

- `0`: Customer did not churn.
- `1`: Customer churned.

The predictions can help a business prioritize retention campaigns and investigate customer segments with higher churn rates.

## 3. Dataset Overview

The project uses the Telco Customer Churn dataset.

The dataset includes information about customer demographics, account characteristics, subscribed services, contract types, payment methods, and billing charges.

Important features investigated include:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `PaymentMethod`
- `TechSupport`
- `SeniorCitizen`

The `customerID` column is removed because it serves as a customer identifier rather than a direct predictive feature.

The exact dataset source, number of records, and final number of features should be documented from the dataset used in the final experiment.

## 4. Data Cleaning and Preprocessing

### 4.1 Data Type Conversion

The `TotalCharges` column is converted to a numeric data type using `pd.to_numeric()` with `errors="coerce"`.

Invalid values are converted to missing values so they can be handled explicitly.

### 4.2 Missing Value Handling

Missing `TotalCharges` values are estimated using:

`TotalCharges = tenure × MonthlyCharges`

Zero-tenure values are replaced with the mean of non-zero tenure values in the current implementation.

These choices should be reviewed because zero tenure may represent a valid new customer, and estimated total charges may not equal actual billed charges.

### 4.3 Categorical Encoding

Binary variables are mapped to numeric values:

- `Churn`: Yes = 1, No = 0
- `Partner`: Yes = 1, No = 0
- `Dependents`: Yes = 1, No = 0
- `PhoneService`: Yes = 1, No = 0
- `gender`: Male = 1, Female = 0

Other categorical features are encoded using One-Hot Encoding.

### 4.4 Train-Test Split

The dataset is split into training and testing subsets using an 80:20 ratio.

Stratification is applied to preserve the approximate class distribution in both subsets. A random state of 42 is used for reproducibility.

### 4.5 Feature Scaling and Encoding

A `ColumnTransformer` applies:

- `StandardScaler` to numerical features.
- `OneHotEncoder` to categorical features.

The preprocessor is fitted on the training features and then used to transform both training and test features.

## 5. Exploratory Data Analysis

The exploratory analysis identified several patterns associated with customer churn.

### 5.1 Contract Type

Customers with month-to-month contracts showed higher churn rates than customers with longer-term contracts.

### 5.2 Customer Tenure

Customers with shorter tenure, especially during the first 12 months, showed higher churn rates.

### 5.3 Monthly Charges

The recorded analysis showed the following average monthly charges:

| Churn Status | Mean Monthly Charges |
|---|---:|
| No Churn | 61.27 |
| Churn | 74.44 |

Customers who churned had a higher average monthly charge in the analyzed data.

### 5.4 Internet Service

Customers using Fiber optic service showed higher observed churn rates.

### 5.5 Payment Method

Electronic check users showed the highest observed churn rate among the payment methods compared.

### 5.6 Technical Support

Customers without technical support showed higher observed churn rates.

These patterns represent associations in the dataset and do not prove that the listed factors directly cause churn.

## 6. Feature Engineering

The project explored grouping customer tenure into intervals to simplify churn analysis across customer lifecycle stages.

A `TenureGroup` feature can help compare churn rates between newer and longer-standing customers.

The final feature list should be checked to confirm which engineered features were included in the trained model.

## 7. Model Development

### 7.1 Logistic Regression

Logistic Regression was selected as a classification algorithm for predicting customer churn.

The model configuration is:

- `max_iter = 1000`
- `random_state = 42`

The higher maximum iteration limit gives the optimization algorithm more opportunities to converge during training.

### 7.2 Handling Class Imbalance

SMOTE was explored to address the imbalance between churn and non-churn classes.

SMOTE generates synthetic minority-class examples and should only be applied to the training data.

The evaluation dataset must remain unchanged to provide a realistic estimate of model performance.

**Implementation note:** In the current `processing()` function, SMOTE creates balanced training data, but the saved Logistic Regression model is loaded without being fitted again. Therefore, the balanced data does not affect the loaded model unless it was used during an earlier training step.

## 8. Model Evaluation

The model is evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the overall proportion of correct predictions. |
| Precision | Measures how many predicted churners actually churned. |
| Recall | Measures how many actual churners were identified. |
| F1-score | Balances Precision and Recall. |
| ROC-AUC | Measures the model's ability to rank churners above non-churners across thresholds. |

A confusion matrix is also generated to examine True Positives, True Negatives, False Positives, and False Negatives.

The model's predicted probabilities are used to plot the ROC curve and calculate ROC-AUC.

### Classification Threshold

The current evaluation uses a threshold of 0.70.

Customers with predicted churn probabilities of at least 70% are classified as potential churners.

A higher threshold generally reduces the number of positive predictions and may improve Precision while reducing Recall. The actual effect must be measured on the evaluation data.

### Final Evaluation Results

The following table should be completed using the actual output from the final model evaluation.

| Metric | Final Result |
|---|---:|
| Accuracy | To be measured |
| Precision | To be measured |
| Recall | To be measured |
| F1-score | To be measured |
| ROC-AUC | To be measured |
| Classification Threshold | 0.70 |

The results must be generated from the final saved model and the matching preprocessing workflow before publication.

## 9. Business Insights and Recommendations

Based on the exploratory analysis, the following actions are worth investigating:

1. **Improve early retention:** Introduce onboarding support and monitor customer satisfaction during the first year.
2. **Review month-to-month contracts:** Investigate why these customers leave more frequently and test suitable retention offers.
3. **Review higher-charge customers:** Evaluate perceived value, pricing concerns, and service usage.
4. **Investigate Fiber optic churn:** Examine customer feedback, service quality, and pricing.
5. **Improve technical support:** Test whether proactive troubleshooting reduces churn among customers without support.
6. **Investigate electronic check users:** Examine billing friction and customer characteristics before changing payment processes.
7. **Prioritize outreach:** Use validated churn scores to prioritize retention campaigns, subject to business review.

These recommendations are hypotheses that require business validation. The dataset alone cannot demonstrate that any particular intervention will reduce churn.

## 10. Limitations

- The final evaluation metrics have not yet been recorded in this report.
- The saved model's exact training configuration needs to be verified.
- SMOTE-generated data is not used to fit the model inside the current processing function.
- Zero-tenure replacement and total-charge imputation may introduce inaccuracies.
- Some cleaning operations occur before the train-test split.
- A single train-test split may not fully represent future performance.
- Observed relationships do not establish causality.
- The selected threshold of 0.70 requires validation against business objectives and campaign costs.

## 11. Future Improvements

- Validate the full training and inference pipeline.
- Compare Logistic Regression with additional classification algorithms.
- Use cross-validation for more reliable performance estimates.
- Evaluate the Precision-Recall curve and PR-AUC.
- Select the classification threshold using validation data.
- Save the fitted preprocessor and model together in a single pipeline.
- Save confusion matrix and ROC curve figures under `reports/figures/`.
- Add automated checks for missing values and unexpected categories.
- Evaluate retention campaigns using actual business outcomes.

## 12. Conclusion

This project demonstrates a structured machine-learning workflow for analyzing and predicting customer churn. The analysis highlights customer segments associated with higher observed churn rates, while Logistic Regression provides a baseline classification approach.

The project also explores feature preprocessing, class-imbalance handling, probability thresholds, and multiple evaluation metrics.

Before the report is finalized for publication, the saved model and preprocessing steps should be verified for compatibility, and the actual test-set metrics should be added. These steps will make the results more reproducible and useful for business decision-making.