"""Build pre-fight features from raw fight data.

This module is intentionally reusable and should be the place where data
cleaning and feature construction logic lives before model training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw fight data from CSV."""
    return pd.read_csv(path)


def clean_fight_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning steps for raw fight dataset."""
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower().replace(" ", "_") for col in cleaned.columns]
    cleaned = cleaned.replace({"nan": None, "": None})
    return cleaned


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a reusable feature matrix for modeling.

    This is a placeholder implementation that can be replaced with the actual
    fight-dataset engineering logic as the project evolves.
    """
    features = df.copy()
    return features


if __name__ == "__main__":
    raw_path = Path("data/raw")
    if not raw_path.exists():
        raise FileNotFoundError("Raw data directory is missing. Add your Kaggle CSVs to data/raw/")

    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw/")

    df = load_raw_data(csv_files[0])
    cleaned = clean_fight_data(df)
    features = build_features(cleaned)
    out_path = Path("data/interim") / "features.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved interim features to {out_path}")
