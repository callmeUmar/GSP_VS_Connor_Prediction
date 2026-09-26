"""Predict fight outcomes for two fighters."""
from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Load the three things train_model.py saved
model = joblib.load(MODELS_DIR / "win_model.pkl")
scaler = joblib.load(MODELS_DIR / "win_scaler.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

# Load the full fighter history (NOT filtered yet)
history = pd.read_csv(PROCESSED_DIR / "fighter_history.csv", parse_dates=["event_date"])

print("Model loaded:", type(model))
print("Number of features:", len(feature_cols))
print("Fighter history shape:", history.shape)


gsp_id = "6506c1d34da9c013"
mcgregor_id = "f4c49976c75c5ab2"

gsp_history = history[history["fighter_id"] == gsp_id]
mcgregor_history = history[history["fighter_id"] == mcgregor_id]

print("GSP fights:", len(gsp_history))
print("McGregor fights:", len(mcgregor_history))

raw_cols = ["pre_slpm", "pre_sapm", "pre_str_acc", "pre_str_def",
            "pre_td_avg", "pre_td_acc", "pre_td_def", "pre_sub_avg",
            "pre_kd_avg", "pre_ctrl_pct", "pre_win_pct",
            "age_at_fight", "height_inches", "reach_inches", "prior_fight_count"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(gsp_history[["event_date", "opponent_name"] + raw_cols].to_string(index=False))
print()
print(mcgregor_history[["event_date", "opponent_name"] + raw_cols].to_string(index=False))