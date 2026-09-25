"""
train_model.py
"""
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

df = pd.read_csv(PROCESSED_DIR / "training_data.csv", parse_dates=["event_date"])

FEATURE_COLS = [c for c in df.columns if c.startswith("diff_")]
TARGET_COL = "target_win"

CUTOFF_DATE = "2022-01-01"
train_df = df[df["event_date"] < CUTOFF_DATE]
test_df = df[df["event_date"] >= CUTOFF_DATE]

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

print(f"Train: {len(X_train)} fights ({train_df['event_date'].min().date()} to {train_df['event_date'].max().date()})")
print(f"Test:  {len(X_test)} fights ({test_df['event_date'].min().date()} to {test_df['event_date'].max().date()})")

# ---------------------------------------------------------------
# Step 2: baseline model -- logistic regression on standardized features
# ---------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train -- never let test data influence scaling
X_test_scaled = scaler.transform(X_test)          # test just gets transformed with train's mean/std

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]
pred_label = logreg.predict(X_test_scaled)

print("\n--- Logistic Regression baseline ---")
print(f"Accuracy: {accuracy_score(y_test, pred_label):.3f}")
print(f"AUC:      {roc_auc_score(y_test, pred_proba):.3f}")
print(f"Log loss: {log_loss(y_test, pred_proba):.3f}")

# ---------------------------------------------------------------
# Step 3: a stronger model -- gradient boosting
# Tree-based models don't need scaling (they split on raw thresholds),
# so we feed them the original, unscaled features.
# ---------------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,          # shallow trees -- keeps it from memorizing noise
    learning_rate=0.05,
    random_state=42,
)
gb.fit(X_train, y_train)

gb_pred_proba = gb.predict_proba(X_test)[:, 1]
gb_pred_label = gb.predict(X_test)

print("\n--- Gradient Boosting ---")
print(f"Accuracy: {accuracy_score(y_test, gb_pred_label):.3f}")
print(f"AUC:      {roc_auc_score(y_test, gb_pred_proba):.3f}")
print(f"Log loss: {log_loss(y_test, gb_pred_proba):.3f}")

# Sanity check: is gradient boosting overfitting? Compare train vs test accuracy.
gb_train_acc = accuracy_score(y_train, gb.predict(X_train))
logreg_train_acc = accuracy_score(y_train, logreg.predict(X_train_scaled))
print(f"\n--- Overfitting check (train accuracy vs test accuracy) ---")
print(f"Logistic Regression: train={logreg_train_acc:.3f}  test={accuracy_score(y_test, pred_label):.3f}  gap={logreg_train_acc - accuracy_score(y_test, pred_label):.3f}")
print(f"Gradient Boosting:   train={gb_train_acc:.3f}  test={accuracy_score(y_test, gb_pred_label):.3f}  gap={gb_train_acc - accuracy_score(y_test, gb_pred_label):.3f}")

# ---------------------------------------------------------------
# Step 4: interpret the winning model -- logistic regression coefficients
# Because features were standardized, coefficients are directly comparable:
# a bigger absolute value = a bigger effect on the win prediction.
# ---------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": logreg.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

print("\n--- What matters most (logistic regression coefficients) ---")
print(coef_df.to_string(index=False))

# ---------------------------------------------------------------
# Step 5: refit on ALL data (train + test) and save
# Once we've picked a model and validated it honestly on held-out data,
# it's standard practice to retrain on everything before deploying it --
# more data = a better final model, and we've already proven it generalizes.
# ---------------------------------------------------------------
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

X_all, y_all = df[FEATURE_COLS], df[TARGET_COL]
final_scaler = StandardScaler().fit(X_all)
final_model = LogisticRegression(max_iter=1000).fit(final_scaler.transform(X_all), y_all)

joblib.dump(final_model, MODELS_DIR / "win_model.pkl")
joblib.dump(final_scaler, MODELS_DIR / "win_scaler.pkl")
joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")

print(f"\nSaved final model to {MODELS_DIR}")
