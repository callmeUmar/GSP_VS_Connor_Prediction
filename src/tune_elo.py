# src/tune_elo.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import build_features as bf

master_df = bf.load_master()
long_base = bf.add_recent_form(
    bf.add_defensive_stats(
        bf.build_fighter_history(
            bf.add_per_fight_totals(bf.to_long_format(master_df))
        )
    )
)
long_base["height_inches"] = long_base["height"].apply(bf.height_to_inches)

for k in [16, 20, 32, 40, 64]:
    long_df = bf.add_elo(long_base.copy(), k=k)
    data = bf.build_matchup_dataset(long_df, master_df)
    data = data.sort_values("event_date").reset_index(drop=True)

    feature_cols = [c for c in data.columns if c.startswith("diff_")]
    X, y = data[feature_cols], data["target_win"]

    scores = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(accuracy_score(y.iloc[test_idx], pipe.predict(X.iloc[test_idx])))

    print(f"K={k:3d}: mean CV accuracy = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")