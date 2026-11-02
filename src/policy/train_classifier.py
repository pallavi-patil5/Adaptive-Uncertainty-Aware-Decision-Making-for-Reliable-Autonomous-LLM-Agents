# src/policy/train_classifier.py
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os


FEATURE_COLUMNS = [
    "uncertainty", "self_consistency_uncertainty", "self_eval_uncertainty",
    "evidence_coverage", "top1_distance", "support_score",
    "contradiction_prob", "entailment_prob",
    "ambiguity", "complexity", "is_claim_check",
]


def _load_split(path):
    df = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + ["expected_action"] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _report(model, encoder, df, split_name):
    X = df[FEATURE_COLUMNS].fillna(0)
    y = encoder.transform(df["expected_action"])
    predictions = model.predict(X)
    print(f"\n{split_name} classification report:")
    print(classification_report(
        y,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    ))


def train(
    train_path="data/processed/train_feature_table.csv",
    validation_path="data/processed/val_feature_table.csv",
    test_path="data/processed/test_feature_table.csv",
):
    train_df = _load_split(train_path)
    validation_df = _load_split(validation_path)
    test_df = _load_split(test_path)

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train_raw = train_df["expected_action"]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)

    unknown_actions = (
        set(validation_df["expected_action"]) | set(test_df["expected_action"])
    ) - set(encoder.classes_)
    if unknown_actions:
        raise ValueError(f"Held-out splits contain actions missing from training: {sorted(unknown_actions)}")

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)
    

    print(f"Fitted target: expected_action | training rows: {len(train_df)}")
    print(f"Training action counts:\n{y_train_raw.value_counts().sort_index()}")
    _report(model, encoder, validation_df, "Validation")
    _report(model, encoder, test_df, "Test")

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(model, "data/processed/policy_classifier.joblib")
    joblib.dump(encoder, "data/processed/policy_label_encoder.joblib")
    print("Saved model + label encoder to data/processed/")


if __name__ == "__main__":
    train()