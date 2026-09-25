"""Predict fight outcomes for two fighters."""

import pickle
from pathlib import Path
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Load the three things train_model.py saved
model = joblib.load(???)
scaler = joblib.load(???)
feature_cols = joblib.load(???)

# Load the full fighter history (NOT filtered yet)
history = pd.read_csv(???, parse_dates=["event_date"])

print("Model loaded:", type(model))
print("Number of features:", len(feature_cols))
print("Fighter history shape:", history.shape)


from __future__ import annotations


def load_model(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def build_prediction_row(fighter_a: str, fighter_b: str, stats: dict) -> pd.DataFrame:
    """Create a single-row DataFrame matching the training feature schema."""
    row = {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
    }
    row.update(stats)
    return pd.DataFrame([row])


def predict_fight(model_path: str | Path, fighter_a: str, fighter_b: str, stats: dict):
    model = load_model(model_path)
    row = build_prediction_row(fighter_a, fighter_b, stats)
    prob = model.predict_proba(row)
    return prob


if __name__ == "__main__":
    model_path = Path("models/fight_predictor.pkl")
    example_stats = {
        "fighter_a_striking_accuracy": 0.5,
        "fighter_a_takedown_accuracy": 0.4,
        "fighter_b_striking_accuracy": 0.45,
        "fighter_b_takedown_accuracy": 0.35,
    }
    probs = predict_fight(model_path, "GSP", "McGregor", example_stats)
    print(probs)
