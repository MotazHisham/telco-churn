from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE

from src.preprocessing import load_and_clean_data
from src.feature_engineering import build_preprocessor

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


def train_model():
    # 1. Load and clean data
    df = load_and_clean_data(DATA_PATH)

    # 2. Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. Split before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Build preprocessing + model pipeline
    model_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 5. Train
    model_pipeline.fit(X_train, y_train)

    # 6. Predict
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    # Choose a threshold
    threshold = 0.12

    # Convert probabilities to predictions
    y_pred_threshold = (y_prob >= threshold).astype(int)

    # 7. Evaluate
    print("Accuracy:", round(accuracy_score(y_test, y_pred_threshold), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_threshold))

    # 8. Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model_pipeline,
            "threshold": 0.12
        },
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
