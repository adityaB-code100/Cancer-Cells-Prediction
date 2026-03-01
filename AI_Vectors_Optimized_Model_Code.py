import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = "AI_Vectors_Final_Prediction_Dataset.csv"
MODEL_PATH = "AI_Vectors_GradientBoosting_Model.joblib"
TARGET_COL = "Diagnosis"
SPLIT_COL = "Split"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in dataset.")
    return df


def split_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, SPLIT_COL}]
    X = df[feature_cols]
    y = df[TARGET_COL]

    if SPLIT_COL in df.columns:
        split = df[SPLIT_COL].astype(str).str.lower()
        train_mask = split.eq("train")
        test_mask = split.eq("test")
        if train_mask.any() and test_mask.any():
            return (
                X.loc[train_mask],
                X.loc[test_mask],
                y.loc[train_mask],
                y.loc[test_mask],
                feature_cols,
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = ImbPipeline(
        [
            ("smote", SMOTE(random_state=42)),
            (
                "clf",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)

    print("\n=== Model Metrics ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))


def main():
    df = load_data(DATASET_PATH)
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)

    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(
        {
            "model": model,
            "model_params": model.get_params(),
            "feature_columns": feature_cols,
            "target_column": TARGET_COL,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
