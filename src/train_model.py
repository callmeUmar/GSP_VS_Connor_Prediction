"""Train and persist a fight prediction model."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load final training-ready dataset."""
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    """Placeholder training function.

    Replace this with the selected model and training logic once the dataset is
    ready.
    """
    from sklearn.dummy import DummyClassifier

    target = "target" if "target" in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    return model


def save_model(model, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    data_path = Path("data/processed/final_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError("Training dataset not found. Build it first in data/processed/")

    df = load_training_data(data_path)
    model = train_model(df)
    save_model(model, Path("models/fight_predictor.pkl"))
    print("Model saved to models/fight_predictor.pkl")
