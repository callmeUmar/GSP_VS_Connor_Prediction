"""
plot_sensitivity.py

Builds the GSP-vs-McGregor sensitivity grid (every prime-era snapshot of one
fighter against every prime-era snapshot of the other) and saves it as an
annotated heatmap.

Usage:
    python3 src/plot_sensitivity.py
Output:
    outputs/plots/sensitivity_heatmap.png
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # write files, don't try to open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GSP_ID = "6506c1d34da9c013"
MCGREGOR_ID = "f4c49976c75c5ab2"

GSP_WINDOW = ("2008-01-01", "2013-12-31")
MCGREGOR_WINDOW = ("2015-01-01", "2016-12-31")

RAW_COLS = ["pre_slpm", "pre_sapm", "pre_str_acc", "pre_str_def",
            "pre_td_avg", "pre_td_acc", "pre_td_def", "pre_sub_avg",
            "pre_kd_avg", "pre_ctrl_pct", "pre_win_pct",
            "age_at_fight", "height_inches", "reach_inches",
            "prior_fight_count", "pre_elo"]


def load_artifacts():
    model = joblib.load(MODELS_DIR / "win_model.pkl")
    scaler = joblib.load(MODELS_DIR / "win_scaler.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    history = pd.read_csv(PROCESSED_DIR / "fighter_history.csv",
                          parse_dates=["event_date"])
    return model, scaler, feature_cols, history


def get_window(history, fighter_id, window):
    start, end = window
    sub = history[
        (history["fighter_id"] == fighter_id)
        & (history["event_date"] >= start)
        & (history["event_date"] <= end)
    ].dropna(subset=RAW_COLS)
    return sub.sort_values("event_date")


def build_grid(model, scaler, feature_cols, gsp_window, mcg_window):
    """Return a DataFrame of GSP win probabilities, rows = GSP, cols = McGregor."""
    grid = np.zeros((len(gsp_window), len(mcg_window)))

    for i, (_, gsp_row) in enumerate(gsp_window.iterrows()):
        for j, (_, mcg_row) in enumerate(mcg_window.iterrows()):
            diff = gsp_row[RAW_COLS] - mcg_row[RAW_COLS]
            X = pd.DataFrame([diff.values], columns=feature_cols)
            grid[i, j] = model.predict_proba(scaler.transform(X))[0][1]

    row_labels = [f"{r.opponent_name}\n{r.event_date.year}"
                  for r in gsp_window.itertuples()]
    col_labels = [f"{r.opponent_name}\n{r.event_date.strftime('%b %Y')}"
                  for r in mcg_window.itertuples()]

    return pd.DataFrame(grid, index=row_labels, columns=col_labels)


def plot_heatmap(grid_df, outpath):
    n_rows, n_cols = grid_df.shape

    fig, ax = plt.subplots(figsize=(1.55 * n_cols + 3.2, 0.78 * n_rows + 3.0))

    # Anchor the colour scale at 0.5 so "is GSP favoured at all?" is readable
    # straight off the colour, not just the numbers.
    cmap = LinearSegmentedColormap.from_list(
        "fight", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    )
    im = ax.imshow(grid_df.values, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(grid_df.columns, fontsize=8.5)
    ax.set_yticklabels(grid_df.index, fontsize=8.5)
    ax.tick_params(axis="both", length=0)

    # Thin white gridlines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate every cell; flip text colour on the dark end for contrast
    for i in range(n_rows):
        for j in range(n_cols):
            val = grid_df.values[i, j]
            ax.text(j, i, f"{val:.0%}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > 0.78 else "#1a1a1a")

    ax.set_xlabel("McGregor's stats entering...", fontsize=10.5, labelpad=10)
    ax.set_ylabel("GSP's stats entering...", fontsize=10.5, labelpad=10)

    vals = grid_df.values
    ax.set_title(
        "GSP vs McGregor: does the result depend on which \"prime\" you pick?\n",
        fontsize=13.5, fontweight="bold", loc="left", pad=18
    )
    ax.text(
        0, 1.012,
        f"GSP win probability across all {vals.size} prime-era combinations  ·  "
        f"range {vals.min():.0%}–{vals.max():.0%}  ·  mean {vals.mean():.0%}  ·  "
        f"GSP favoured in {(vals > 0.5).mean():.0%} of them",
        transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02, shrink=0.62)
    cbar.set_label("P(GSP wins)", fontsize=9.5)
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.set_ticklabels(["50%\n(coin flip)", "60%", "70%", "80%", "90%", "100%"])
    cbar.ax.tick_params(labelsize=8.5, length=0)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    model, scaler, feature_cols, history = load_artifacts()

    gsp_window = get_window(history, GSP_ID, GSP_WINDOW)
    mcg_window = get_window(history, MCGREGOR_ID, MCGREGOR_WINDOW)
    print(f"GSP snapshots:      {len(gsp_window)}")
    print(f"McGregor snapshots: {len(mcg_window)}")
    print(f"Matchups:           {len(gsp_window) * len(mcg_window)}")

    grid_df = build_grid(model, scaler, feature_cols, gsp_window, mcg_window)

    vals = grid_df.values
    print(f"\nMin  {vals.min():.1%}   Max {vals.max():.1%}   "
          f"Mean {vals.mean():.1%}   Std {vals.std():.1%}")
    print(f"GSP favoured in {(vals > 0.5).mean():.0%} of matchups")

    outpath = PLOTS_DIR / "sensitivity_heatmap.png"
    plot_heatmap(grid_df, outpath)
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()