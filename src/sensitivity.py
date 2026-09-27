"""Sensitivity check: does the GSP/McGregor result hold across different peak definitions?"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

model = joblib.load(MODELS_DIR / "win_model.pkl")
scaler = joblib.load(MODELS_DIR / "win_scaler.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

history = pd.read_csv(PROCESSED_DIR / "fighter_history.csv", parse_dates=["event_date"])

gsp_id = "6506c1d34da9c013"
mcgregor_id = "f4c49976c75c5ab2"

raw_cols = ["pre_slpm", "pre_sapm", "pre_str_acc", "pre_str_def",
            "pre_td_avg", "pre_td_acc", "pre_td_def", "pre_sub_avg",
            "pre_kd_avg", "pre_ctrl_pct", "pre_win_pct",
            "age_at_fight", "height_inches", "reach_inches", "prior_fight_count",
            "pre_elo"]

gsp_window = history[
    (history["fighter_id"] == gsp_id)
    & (history["event_date"] >= "2008-01-01")
    & (history["event_date"] <= "2013-12-31")
].dropna(subset=raw_cols)

mcgregor_window = history[
    (history["fighter_id"] == mcgregor_id)
    & (history["event_date"] >= "2015-01-01")
    & (history["event_date"] <= "2016-12-31")
].dropna(subset=raw_cols)

print(f"GSP fights in window:      {len(gsp_window)}")
print(f"McGregor fights in window: {len(mcgregor_window)}")
print(f"Total matchups to test:    {len(gsp_window) * len(mcgregor_window)}")
print()

results = []

for _, gsp_row in gsp_window.iterrows():
    for _, mcg_row in mcgregor_window.iterrows():
        diff = gsp_row[raw_cols] - mcg_row[raw_cols]
        X = pd.DataFrame([diff.values], columns=feature_cols)
        prob = model.predict_proba(scaler.transform(X))[0][1]

        results.append({
            "gsp_date": gsp_row["event_date"].date(),
            "gsp_vs": gsp_row["opponent_name"],
            "mcg_date": mcg_row["event_date"].date(),
            "mcg_vs": mcg_row["opponent_name"],
            "gsp_win_prob": prob,
        })

res_df = pd.DataFrame(results)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
print(res_df.to_string(index=False))

print()
print(f"Min:    {res_df['gsp_win_prob'].min():.1%}")
print(f"Max:    {res_df['gsp_win_prob'].max():.1%}")
print(f"Mean:   {res_df['gsp_win_prob'].mean():.1%}")
print(f"Median: {res_df['gsp_win_prob'].median():.1%}")
print(f"Std:    {res_df['gsp_win_prob'].std():.1%}")

# How often is GSP favored at all?
favored = (res_df["gsp_win_prob"] > 0.5).mean()
print(f"\nGSP favored in {favored:.0%} of matchup combinations")