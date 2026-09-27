"""
build_features.py

Turns master.csv (one row per fight, both corners' final stats) into a
leakage-free training dataset: for every fight, each fighter's stats are
computed using ONLY their fights that happened strictly before that date.

Usage:
    python build_features.py
Outputs:
    data/processed/training_data.csv   -- one row per fight, diff features + target
    data/processed/fighter_history.csv -- long-format per-fighter pre-fight snapshots
                                           (reused later to build GSP/McGregor profiles)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_master():
    df = pd.read_csv(RAW_DIR / "master.csv", parse_dates=["event_date"])
    df = df[df["result_status"] == "win"].copy()
    df = df[df["method"].isin(
        ["Decision - Unanimous", "Decision - Split", "Decision - Majority",
         "KO/TKO", "Submission", "TKO - Doctor's Stoppage"]
    )].copy()
    return df


def to_long_format(df):
    r_cols = {c: c[2:] for c in df.columns if c.startswith("r_")}
    b_cols = {c: c[2:] for c in df.columns if c.startswith("b_")}

    shared_cols = [
        "fight_id", "event_id", "event_name", "event_date", "weight_class",
        "title_fight", "winner_id", "method", "finish_round", "rounds_fought",
    ]

    r_rows = df[shared_cols + list(r_cols.keys())].rename(columns=r_cols)
    r_rows["opponent_id"] = df["b_fighter_id"].values
    r_rows["opponent_name"] = df["b_fighter_name"].values

    b_rows = df[shared_cols + list(b_cols.keys())].rename(columns=b_cols)
    b_rows["opponent_id"] = df["r_fighter_id"].values
    b_rows["opponent_name"] = df["r_fighter_name"].values

    long_df = pd.concat([r_rows, b_rows], ignore_index=True)
    long_df["won"] = (long_df["fighter_id"] == long_df["winner_id"]).astype(int)

    long_df["fight_seconds"] = long_df["rounds_fought"] * 5 * 60
    long_df.loc[long_df["fight_seconds"] <= 0, "fight_seconds"] = 5 * 60

    return long_df


def add_per_fight_totals(long_df):
    rename_map = {
        "total_sig_landed": "sig_landed",
        "total_sig_atmp": "sig_atmp",
        "total_total_str_landed": "str_landed",
        "total_total_str_atmp": "str_atmp",
        "total_td_success": "td_landed",
        "total_td_atmp": "td_atmp",
        "total_sub_att": "sub_att",
        "total_kd": "kd",
        "total_rev": "rev",
        "total_ctrl_seconds": "ctrl_seconds",
    }
    long_df = long_df.rename(columns=rename_map)
    return long_df


def build_fighter_history(long_df):
    long_df = long_df.sort_values(["fighter_id", "event_date"]).reset_index(drop=True)

    stat_cols = ["sig_landed", "sig_atmp", "str_landed", "str_atmp",
                 "td_landed", "td_atmp", "sub_att", "kd", "rev", "ctrl_seconds",
                 "fight_seconds", "won"]

    grouped = long_df.groupby("fighter_id")

    for col in stat_cols:
        long_df[f"prior_{col}_sum"] = grouped[col].transform(
            lambda s: s.shift(1).cumsum()
        )

    long_df["prior_fight_count"] = grouped.cumcount()

    minutes = long_df["prior_fight_seconds_sum"] / 60.0
    long_df["pre_slpm"] = long_df["prior_sig_landed_sum"] / minutes
    long_df["pre_sapm"] = np.nan
    long_df["pre_str_acc"] = long_df["prior_sig_landed_sum"] / long_df["prior_sig_atmp_sum"]
    long_df["pre_td_avg"] = long_df["prior_td_landed_sum"] / minutes * 15
    long_df["pre_td_acc"] = long_df["prior_td_landed_sum"] / long_df["prior_td_atmp_sum"]
    long_df["pre_sub_avg"] = long_df["prior_sub_att_sum"] / minutes * 15
    long_df["pre_kd_avg"] = long_df["prior_kd_sum"] / minutes * 15
    long_df["pre_ctrl_pct"] = long_df["prior_ctrl_seconds_sum"] / long_df["prior_fight_seconds_sum"]
    long_df["pre_win_pct"] = long_df["prior_won_sum"] / long_df["prior_fight_count"]

    long_df["dob"] = pd.to_datetime(long_df["dob"], errors="coerce")
    long_df["age_at_fight"] = (long_df["event_date"] - long_df["dob"]).dt.days / 365.25

    return long_df

def add_recent_form(long_df, window=3):
    """
    Recent-form features: same stats as the career versions, but over
    only the last `window` fights. Career averages blur "who he was"
    with "who he is now"; this separates them.
    """
    long_df = long_df.sort_values(["fighter_id", "event_date"]).reset_index(drop=True)
    grouped = long_df.groupby("fighter_id")

    # Sum strikes and minutes separately over the window, THEN divide --
    # this weights a 15-min war more than a 60-second KO, matching how
    # UFC Stats computes per-minute rates.
    for col in ["sig_landed", "opp_sig_landed", "fight_seconds", "won"]:
        long_df[f"recent_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )

    recent_minutes = long_df["recent_fight_seconds"] / 60.0
    long_df["recent_slpm_3"] = long_df["recent_sig_landed"] / recent_minutes
    long_df["recent_sapm_3"] = long_df["recent_opp_sig_landed"] / recent_minutes

    # recent win pct: wins in window / fights actually in window
    long_df["recent_fights_in_window"] = grouped["won"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).count()
    )
    long_df["recent_win_pct_3"] = long_df["recent_won"] / long_df["recent_fights_in_window"]

    # layoff: days since previous fight (ring rust)
    long_df["days_since_last_fight"] = grouped["event_date"].transform(
        lambda s: (s - s.shift(1)).dt.days
    )

    return long_df

def add_defensive_stats(long_df):
    opp = long_df[["fight_id", "fighter_id", "sig_landed", "sig_atmp",
                    "td_landed", "td_atmp"]].rename(columns={
        "fighter_id": "opponent_id",
        "sig_landed": "opp_sig_landed",
        "sig_atmp": "opp_sig_atmp",
        "td_landed": "opp_td_landed",
        "td_atmp": "opp_td_atmp",
    })
    long_df = long_df.merge(opp, on=["fight_id", "opponent_id"], how="left")

    long_df = long_df.sort_values(["fighter_id", "event_date"]).reset_index(drop=True)
    grouped = long_df.groupby("fighter_id")

    for col in ["opp_sig_landed", "opp_sig_atmp", "opp_td_landed", "opp_td_atmp"]:
        long_df[f"prior_{col}_sum"] = grouped[col].transform(lambda s: s.shift(1).cumsum())

    minutes = long_df["prior_fight_seconds_sum"] / 60.0
    long_df["pre_sapm"] = long_df["prior_opp_sig_landed_sum"] / minutes
    long_df["pre_str_def"] = 1 - (long_df["prior_opp_sig_landed_sum"] / long_df["prior_opp_sig_atmp_sum"])
    long_df["pre_td_def"] = 1 - (long_df["prior_opp_td_landed_sum"] / long_df["prior_opp_td_atmp_sum"])

    return long_df


FEATURE_COLS = [
    "pre_slpm", "pre_sapm", "pre_str_acc", "pre_str_def",
    "pre_td_avg", "pre_td_acc", "pre_td_def", "pre_sub_avg",
    "pre_kd_avg", "pre_ctrl_pct", "pre_win_pct",
    "age_at_fight", "height_inches", "reach_inches", "prior_fight_count",
    "recent_slpm_3", "recent_sapm_3", "recent_win_pct_3", "days_since_last_fight",
]



def height_to_inches(h):
    if pd.isna(h):
        return np.nan
    try:
        feet, inches = h.replace('"', '').split("'")
        return int(feet.strip()) * 12 + int(inches.strip())
    except Exception:
        return np.nan


def build_matchup_dataset(long_df, master_df):
    feat = long_df[["fight_id", "fighter_id"] + FEATURE_COLS].copy()

    fights = master_df[["fight_id", "r_fighter_id", "b_fighter_id", "winner_id",
                         "method", "event_date", "weight_class"]].copy()

    fights = fights.merge(
        feat.add_prefix("r_"), left_on=["fight_id", "r_fighter_id"],
        right_on=["r_fight_id", "r_fighter_id"], how="left"
    ).drop(columns=["r_fight_id"])
    fights = fights.merge(
        feat.add_prefix("b_"), left_on=["fight_id", "b_fighter_id"],
        right_on=["b_fight_id", "b_fighter_id"], how="left"
    ).drop(columns=["b_fight_id"])

    fights = fights.dropna(subset=[f"r_{c}" for c in FEATURE_COLS] +
                                   [f"b_{c}" for c in FEATURE_COLS], how="any")

    rng = np.random.default_rng(RANDOM_SEED)
    swap = rng.random(len(fights)) < 0.5

    for c in FEATURE_COLS:
        r_val = fights[f"r_{c}"].values
        b_val = fights[f"b_{c}"].values
        f1 = np.where(swap, b_val, r_val)
        f2 = np.where(swap, r_val, b_val)
        fights[f"diff_{c}"] = f1 - f2

    f1_won = np.where(swap,
                       (fights["winner_id"] == fights["b_fighter_id"]).astype(int),
                       (fights["winner_id"] == fights["r_fighter_id"]).astype(int))
    fights["target_win"] = f1_won

    def method_group(m):
        if m == "KO/TKO":
            return "KO/TKO"
        if m == "Submission":
            return "Submission"
        return "Decision"
    fights["target_method"] = fights["method"].apply(method_group)

    diff_cols = [f"diff_{c}" for c in FEATURE_COLS]
    out = fights[["fight_id", "event_date", "weight_class"] + diff_cols +
                 ["target_win", "target_method"]].copy()
    return out


def main():
    print("Loading master.csv ...")
    master_df = load_master()
    print(f"  {len(master_df)} clean win/loss fights")

    print("Reshaping to long format ...")
    long_df = to_long_format(master_df)
    long_df = add_per_fight_totals(long_df)

    print("Computing pre-fight cumulative stats (no leakage) ...")
    long_df = build_fighter_history(long_df)
    long_df = add_defensive_stats(long_df)
    long_df = add_recent_form(long_df)
    long_df["height_inches"] = long_df["height"].apply(height_to_inches)

    print("Saving fighter history (for GSP/McGregor profiles later) ...")
    long_df.to_csv(PROCESSED_DIR / "fighter_history.csv", index=False)

    print("Building matchup training dataset ...")
    training_data = build_matchup_dataset(long_df, master_df)
    training_data.to_csv(PROCESSED_DIR / "training_data.csv", index=False)

    print(f"\nDone. Training rows: {len(training_data)}")
    print(f"Target win rate (should be ~50% due to random swap): {training_data['target_win'].mean():.3f}")
    print(f"\nMethod breakdown:\n{training_data['target_method'].value_counts()}")
    print(f"\nSaved to:\n  {PROCESSED_DIR / 'training_data.csv'}\n  {PROCESSED_DIR / 'fighter_history.csv'}")


if __name__ == "__main__":
    main()