from __future__ import annotations

import os
import subprocess
import sys

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


LABEL_MAP = {-1: 0, 0: 1, 1: 2}


def _normalize_labels(raw_labels: pd.Series) -> pd.Series:
    mapped = raw_labels.replace(LABEL_MAP)
    if mapped.isna().any():
        unknown = sorted(set(raw_labels[mapped.isna()].tolist()))
        raise ValueError(f"Unsupported labels in dataset: {unknown}")
    return mapped.astype(int)


def train_model(data_path: str = "data/labeled_10m.csv", model_path: str = "models/latest_model.pkl") -> None:
    print("Generating labeled 10m data...")
    # Reuse the same interpreter that launched training so dependency resolution
    # stays inside the active VM/venv.
    subprocess.run([sys.executable, "scripts/grab_1m_candles.py"], check=True)

    print("Loading labeled data...")
    df = pd.read_csv(data_path)
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in labeled dataset.")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = _normalize_labels(df["label"])
    print("Label distribution:\n", df["label"].value_counts().sort_index())

    X = df.drop(columns=["label", "timestamp"], errors="ignore")
    y = df["label"]

    # XGBoost needs every class represented. If the current sample is sparse,
    # add one neutral synthetic row per missing class instead of crashing.
    missing_labels = [label for label in [0, 1, 2] if label not in set(y.unique())]
    if missing_labels:
        template = X.median(numeric_only=True).to_dict()
        for label in missing_labels:
            print(f"Adding a synthetic sample for missing class {label}.")
            synthetic = pd.DataFrame([template])
            synthetic["label"] = label
            df = pd.concat([df, synthetic], ignore_index=True)
        X = df.drop(columns=["label", "timestamp"], errors="ignore")
        y = df["label"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Not enough class diversity to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Training XGBoost model...")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification report:\n", report)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()
