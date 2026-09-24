# GSP vs McGregor

This project is organized around a lightweight MMA fight-prediction workflow using a data science pipeline:

- raw Kaggle CSV files stay in `data/raw/`
- intermediate cleaned data is stored in `data/interim/`
- final processed/training-ready data goes to `data/processed/`
- reusable modeling code lives in `src/`
- trained artifacts are saved in `models/`
- output plots are stored in `outputs/plots/`

## Folder layout

- `data/raw/` — original files, untouched
- `data/interim/` — cleaned but not feature-engineered data
- `data/processed/` — final dataset used for training
- `notebooks/` — exploration, feature engineering, modeling, and dream-fight notebooks
- `src/` — reusable Python scripts for feature creation, training, and prediction
- `models/` — saved model artifacts
- `outputs/plots/` — charts and visual outputs

## Quick start

1. Place your raw CSV files in `data/raw/`.
2. Use the notebooks in `notebooks/` for data exploration and feature logic.
3. Run the preprocessing script in `src/build_features.py` to create interim features.
4. Train the model with `src/train_model.py`.
5. Use `src/predict.py` to score a matchup.

## Notes

This repository is intentionally scaffolded so you can evolve the project from raw dataset ingestion to a full predictive pipeline.
