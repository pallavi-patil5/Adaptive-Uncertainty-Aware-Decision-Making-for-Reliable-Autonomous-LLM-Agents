# src/policy/train_classifier.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

FEATURE_COLUMNS = [
    "uncertainty", "self_consistency_uncertainty", "self_eval_uncertainty",
    "evidence_coverage", "top1_distance", "support_score",
    "contradiction_prob", "entailment_prob",
    "ambiguity", "complexity",
]


def train(path="data/processed/feature_table.csv"):
    df = pd.read_csv(path)

    if len(df) < 20:
        print(f"WARNING: only {len(df)} rows — this is a pipeline smoke test, not a real trained model.")
        print("Expand the Custom Decision Dataset before trusting any accuracy number below.\n")

    X = df[FEATURE_COLUMNS].fillna(0)
    y_raw = df["expected_action"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
    else:
        # too few rows to hold out a test set meaningfully — train on everything, just confirm it runs
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(
        y_test, preds, labels=range(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0,
    ))

    joblib.dump(model, "data/processed/policy_classifier.joblib")
    joblib.dump(encoder, "data/processed/policy_label_encoder.joblib")
    print("Saved model + label encoder to data/processed/")


if __name__ == "__main__":
    train()