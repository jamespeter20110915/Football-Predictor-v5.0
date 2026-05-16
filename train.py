"""
Stage 3 — Model Training

Time-based split across 12 seasons:
  Train: 2014-15 through 2022-23 (9 seasons)
  Val:   2023-24              (1 season)
  Test:  2024-25 through 2025-26 (2 seasons)

Models:
  1. XGBClassifier  — match result (H / D / A), objective='multi:softprob'
  2. XGBRegressor   — home goals,     objective='count:poisson'
  3. XGBRegressor   — away goals,     objective='count:poisson'

All use early stopping on validation set (patience=50).
NaN values are left as-is — XGBoost handles them natively.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23",
]
VAL_SEASON = "2023-24"
TEST_SEASONS = ["2024-25", "2025-26"]

MODEL_DIR = "models"

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

EARLY_STOPPING_ROUNDS = 50

# Columns excluded from features
ID_COLS = {"date", "league", "season", "home_team", "away_team"}
TARGET_COLS = {
    "target_result", "target_home_goals",
    "target_away_goals", "target_total_goals",
}

# ---------------------------------------------------------------------------
# 1. Data loading & time-based split
# ---------------------------------------------------------------------------

def load_and_split(features_path: str) -> dict:
    """Read features.parquet and split by season. Returns a dict of arrays."""
    df = pd.read_parquet(features_path)
    print(f"Loaded features: {df.shape}")

    # Drop rows where season is null (shouldn't happen, but defensive)
    df = df.dropna(subset=["season"])

    mask_train = df["season"].isin(TRAIN_SEASONS)
    mask_val = df["season"] == VAL_SEASON
    mask_test = df["season"].isin(TEST_SEASONS)

    parts = {}
    for name, mask in [("train", mask_train), ("val", mask_val), ("test", mask_test)]:
        subset = df[mask].copy()
        parts[name] = {
            "df": subset,
            "y_result": subset["target_result"].map({"H": 0, "D": 1, "A": 2}).values.astype(int),
            "y_home":   subset["target_home_goals"].values.astype(float),
            "y_away":   subset["target_away_goals"].values.astype(float),
            "n": len(subset),
            "date_range": f"{subset['date'].min()}  →  {subset['date'].max()}",
        }

    for k in ["train", "val", "test"]:
        print(f"  {k:5s}: {parts[k]['n']:5d} matches  [{parts[k]['date_range']}]")

    return parts


# ---------------------------------------------------------------------------
# 2. Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, feature_cols: list,
                     league_encoder: LabelEncoder = None) -> np.ndarray:
    """
    Build the feature matrix X from a DataFrame subset.

    - Drops ID and target columns
    - Label-encodes league (fit on first call, transform on subsequent)
    - Returns float32 ndarray with columns in feature_cols order
    """
    if league_encoder is None:
        league_encoder = LabelEncoder()
        league_encoder.fit(df["league"])

    league_encoded = league_encoder.transform(df["league"])

    # Build feature DataFrame excluding ID + target columns
    drop_cols = [c for c in ID_COLS | TARGET_COLS if c in df.columns]
    X_df = df.drop(columns=drop_cols).copy()

    # Replace league string column with encoded integer
    X_df["league_encoded"] = league_encoded

    # Ensure column order matches saved feature_cols
    # On first call (train), derive and save. On subsequent, enforce order.
    if feature_cols is None:
        feature_cols = list(X_df.columns)
    else:
        X_df = X_df[feature_cols]

    return X_df.astype(np.float32).values, feature_cols, league_encoder


# ---------------------------------------------------------------------------
# 3. Train classifier (H/D/A)
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train, X_val, y_val, params: dict) -> tuple:
    """Train XGBClassifier with early stopping on validation mlogloss."""
    clf_params = {
        **params,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }

    model = xgb.XGBClassifier(**clf_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )
    return model, model.evals_result()


# ---------------------------------------------------------------------------
# 4. Train regressors (home goals, away goals)
# ---------------------------------------------------------------------------

def train_regressor(X_train, y_train, X_val, y_val, params: dict, name: str) -> tuple:
    """Train XGBRegressor with Poisson objective. Early stopping on val RMSE."""
    reg_params = {
        **params,
        "objective": "count:poisson",
        "eval_metric": "rmse",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }

    model = xgb.XGBRegressor(**reg_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )
    return model, model.evals_result()


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(model, X_test, y_test) -> dict:
    """Compute all classifier metrics."""
    y_pred = model.predict(X_test)           # class labels 0/1/2
    y_proba = model.predict_proba(X_test)     # (N, 3)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Log loss
    ll = log_loss(y_test, y_proba)

    # Multi-class Brier score (one-vs-rest, average)
    y_onehot = np.eye(3)[y_test]
    brier = np.mean([
        brier_score_loss(y_onehot[:, k], y_proba[:, k])
        for k in range(3)
    ])

    # Per-class metrics
    report = classification_report(
        y_test, y_pred,
        target_names=["H", "D", "A"],
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "per_class": {k: v for k, v in report.items()
                      if k in ("H", "D", "A")},
        "weighted_avg": report["weighted avg"],
        "macro_avg": report["macro avg"],
    }


def evaluate_regressor(model, X_test, y_test, name: str) -> dict:
    """Compute regression metrics for a single regressor."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Poisson deviance: 2 * mean(y_pred - y_true + y_true * log(y_true/y_pred))
    # Clamp predictions to avoid log(0)
    eps = 1e-10
    y_pred_safe = np.clip(y_pred, eps, None)
    poisson_dev = 2.0 * np.mean(
        y_pred_safe - y_test + y_test * np.log((y_test + eps) / y_pred_safe)
    )

    return {
        "name": name,
        "rmse": float(rmse),
        "mae": float(mae),
        "poisson_deviance": float(poisson_dev),
        "sum_actual": float(y_test.sum()),
        "sum_predicted": float(y_pred.sum()),
    }


# ---------------------------------------------------------------------------
# 6. Baseline: b365 implied probability
# ---------------------------------------------------------------------------

def evaluate_b365_baseline(df_test: pd.DataFrame) -> dict:
    """
    Use Bet365 implied probabilities as a predictor.

    - Class prediction: argmax of (p_home, p_draw, p_away)
    - b365 probabilities were computed in features.py as mkt_b365_p_*
    """
    if not {"mkt_b365_p_home", "mkt_b365_p_draw", "mkt_b365_p_away"}.issubset(df_test.columns):
        return {"error": "b365 market features not found in test set"}

    y_true_str = df_test["target_result"]
    y_true = y_true_str.map({"H": 0, "D": 1, "A": 2}).values

    p_h = df_test["mkt_b365_p_home"].values
    p_d = df_test["mkt_b365_p_draw"].values
    p_a = df_test["mkt_b365_p_away"].values

    # Filter rows where any b365 prob is NaN
    valid = ~(np.isnan(p_h) | np.isnan(p_d) | np.isnan(p_a))
    y_true = y_true[valid]
    p_h, p_d, p_a = p_h[valid], p_d[valid], p_a[valid]
    proba = np.column_stack([p_h, p_d, p_a])
    y_pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, proba)

    return {
        "accuracy": acc,
        "log_loss": ll,
        "n_matches": int(valid.sum()),
        "note": "Based on Bet365 implied probabilities (vig-adjusted)",
    }


# ---------------------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(clf, reg_h, reg_a, feature_cols, league_encoder, parts, metrics):
    """Save all models and metadata to models/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Models
    clf.save_model(f"{MODEL_DIR}/clf_result.json")
    reg_h.save_model(f"{MODEL_DIR}/reg_home_goals.json")
    reg_a.save_model(f"{MODEL_DIR}/reg_away_goals.json")
    print(f"Models saved to {MODEL_DIR}/")

    # Feature columns
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Label encoder
    le_data = {
        "classes_": league_encoder.classes_.tolist(),
        "mapping": {name: int(i) for i, name in enumerate(league_encoder.classes_)},
    }
    with open(f"{MODEL_DIR}/label_encoder.json", "w") as f:
        json.dump(le_data, f, indent=2)

    # Training report
    report = {
        "data_split": {
            "train": {"n": parts["train"]["n"], "date_range": parts["train"]["date_range"]},
            "val":   {"n": parts["val"]["n"],   "date_range": parts["val"]["date_range"]},
            "test":  {"n": parts["test"]["n"],  "date_range": parts["test"]["date_range"]},
        },
        "best_iterations": {
            "classifier": int(clf.best_iteration) if clf.best_iteration else None,
            "reg_home": int(reg_h.best_iteration) if reg_h.best_iteration else None,
            "reg_away": int(reg_a.best_iteration) if reg_a.best_iteration else None,
        },
        "metrics": metrics,
    }
    with open(f"{MODEL_DIR}/training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {MODEL_DIR}/training_report.json")


# ---------------------------------------------------------------------------
# 8. Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model, feature_cols, top_n=15, label="Model"):
    """Print top-N feature importances by gain."""
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        print(f"  {label}: no importance scores available")
        return

    # Map f0, f1, ... to feature names
    mapped = {}
    for k, v in importance.items():
        if k.startswith("f"):
            idx = int(k[1:])
            name = feature_cols[idx] if idx < len(feature_cols) else k
            mapped[name] = v
        else:
            mapped[k] = v

    sorted_imp = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n  {label} — Top {top_n} feature importances (gain):")
    for rank, (feat, gain) in enumerate(sorted_imp, 1):
        print(f"    {rank:2d}. {feat:<30s} {gain:>10.1f}")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    features_path = "data/processed/features.parquet"

    # --- Load & split ---
    print("=" * 70)
    print("1. DATA SPLIT")
    print("=" * 70)
    parts = load_and_split(features_path)

    # --- Prepare features ---
    print("\n" + "=" * 70)
    print("2. FEATURE PREPARATION")
    print("=" * 70)

    # Fit encoder and build X on train
    X_train, feature_cols, league_encoder = prepare_features(
        parts["train"]["df"], None, None
    )
    # Apply same encoder + feature order to val and test
    X_val, _, _ = prepare_features(
        parts["val"]["df"], feature_cols, league_encoder
    )
    X_test, _, _ = prepare_features(
        parts["test"]["df"], feature_cols, league_encoder
    )

    # Check for unseen league values in val/test
    for part_name in ["val", "test"]:
        unseen = set(parts[part_name]["df"]["league"]) - set(league_encoder.classes_)
        if unseen:
            print(f"  Warning: unseen leagues in {part_name}: {unseen}")

    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Train X: {X_train.shape}  Val X: {X_val.shape}  Test X: {X_test.shape}")
    print(f"  League mapping: {dict(zip(league_encoder.classes_, range(len(league_encoder.classes_))))}")

    # --- Train classifier ---
    print("\n" + "=" * 70)
    print("3. TRAIN CLASSIFIER (H/D/A)")
    print("=" * 70)
    clf, clf_evals = train_classifier(
        X_train, parts["train"]["y_result"],
        X_val,   parts["val"]["y_result"],
        XGB_PARAMS,
    )
    print(f"  Best iteration: {clf.best_iteration}")
    best_clf_train = clf_evals["validation_0"]["mlogloss"][clf.best_iteration]
    best_clf_val   = clf_evals["validation_1"]["mlogloss"][clf.best_iteration]
    print(f"  Best train mlogloss: {best_clf_train:.4f}")
    print(f"  Best val   mlogloss: {best_clf_val:.4f}")

    # --- Train home-goals regressor ---
    print("\n" + "=" * 70)
    print("4. TRAIN REGRESSOR — HOME GOALS")
    print("=" * 70)
    reg_h, reg_h_evals = train_regressor(
        X_train, parts["train"]["y_home"],
        X_val,   parts["val"]["y_home"],
        XGB_PARAMS, "home_goals",
    )
    print(f"  Best iteration: {reg_h.best_iteration}")
    best_rh_train = reg_h_evals["validation_0"]["rmse"][reg_h.best_iteration]
    best_rh_val   = reg_h_evals["validation_1"]["rmse"][reg_h.best_iteration]
    print(f"  Best train RMSE: {best_rh_train:.4f}")
    print(f"  Best val   RMSE: {best_rh_val:.4f}")

    # --- Train away-goals regressor ---
    print("\n" + "=" * 70)
    print("5. TRAIN REGRESSOR — AWAY GOALS")
    print("=" * 70)
    reg_a, reg_a_evals = train_regressor(
        X_train, parts["train"]["y_away"],
        X_val,   parts["val"]["y_away"],
        XGB_PARAMS, "away_goals",
    )
    print(f"  Best iteration: {reg_a.best_iteration}")
    best_ra_train = reg_a_evals["validation_0"]["rmse"][reg_a.best_iteration]
    best_ra_val   = reg_a_evals["validation_1"]["rmse"][reg_a.best_iteration]
    print(f"  Best train RMSE: {best_ra_train:.4f}")
    print(f"  Best val   RMSE: {best_ra_val:.4f}")

    # --- Evaluate on test set ---
    print("\n" + "=" * 70)
    print("6. TEST SET EVALUATION")
    print("=" * 70)

    clf_metrics = evaluate_classifier(clf, X_test, parts["test"]["y_result"])
    reg_h_metrics = evaluate_regressor(reg_h, X_test, parts["test"]["y_home"], "home_goals")
    reg_a_metrics = evaluate_regressor(reg_a, X_test, parts["test"]["y_away"], "away_goals")

    # Total goals: combine home + away predictions
    y_pred_h = reg_h.predict(X_test)
    y_pred_a = reg_a.predict(X_test)
    y_pred_total = y_pred_h + y_pred_a
    y_true_total = parts["test"]["y_home"] + parts["test"]["y_away"]
    total_rmse = np.sqrt(mean_squared_error(y_true_total, y_pred_total))
    total_mae = mean_absolute_error(y_true_total, y_pred_total)

    print("\n  --- Classifier ---")
    print(f"  Accuracy:   {clf_metrics['accuracy']:.4f}  ({clf_metrics['accuracy']*100:.2f}%)")
    print(f"  Log Loss:   {clf_metrics['log_loss']:.4f}")
    print(f"  Brier:      {clf_metrics['brier_score']:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"              Pred H   Pred D   Pred A")
    labels = ["True H", "True D", "True A"]
    for i, label in enumerate(labels):
        print(f"    {label}  {clf_metrics['confusion_matrix'][i][0]:>7d}  {clf_metrics['confusion_matrix'][i][1]:>7d}  {clf_metrics['confusion_matrix'][i][2]:>7d}")
    print(f"\n  Per-class metrics:")
    for cls in ["H", "D", "A"]:
        m = clf_metrics["per_class"][cls]
        print(f"    {cls}:  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1-score']:.3f}  support={int(m['support'])}")

    print(f"\n  --- Regressor: Home Goals ---")
    print(f"  RMSE:  {reg_h_metrics['rmse']:.4f}")
    print(f"  MAE:   {reg_h_metrics['mae']:.4f}")
    print(f"  Poisson Deviance: {reg_h_metrics['poisson_deviance']:.4f}")
    print(f"  Sum actual:    {reg_h_metrics['sum_actual']:.0f}")
    print(f"  Sum predicted: {reg_h_metrics['sum_predicted']:.0f}")

    print(f"\n  --- Regressor: Away Goals ---")
    print(f"  RMSE:  {reg_a_metrics['rmse']:.4f}")
    print(f"  MAE:   {reg_a_metrics['mae']:.4f}")
    print(f"  Poisson Deviance: {reg_a_metrics['poisson_deviance']:.4f}")
    print(f"  Sum actual:    {reg_a_metrics['sum_actual']:.0f}")
    print(f"  Sum predicted: {reg_a_metrics['sum_predicted']:.0f}")

    print(f"\n  --- Combined: Total Goals ---")
    print(f"  RMSE: {total_rmse:.4f}")
    print(f"  MAE:  {total_mae:.4f}")

    # --- Baseline comparison ---
    print("\n" + "=" * 70)
    print("7. BASELINE COMPARISON (same test set)")
    print("=" * 70)
    b365 = evaluate_b365_baseline(parts["test"]["df"])
    if "error" in b365:
        print(f"  B365 baseline: {b365['error']}")
    else:
        print(f"  B365 accuracy: {b365['accuracy']:.4f} ({b365['accuracy']*100:.2f}%)  on {b365['n_matches']} matches")
        print(f"  B365 log loss: {b365['log_loss']:.4f}")
        print(f"\n  Model vs B365:")
        print(f"    Accuracy:  {clf_metrics['accuracy']:.4f}  vs  {b365['accuracy']:.4f}  (Δ = {clf_metrics['accuracy']-b365['accuracy']:+.4f})")
        print(f"    Log Loss:  {clf_metrics['log_loss']:.4f}  vs  {b365['log_loss']:.4f}  (Δ = {clf_metrics['log_loss']-b365['log_loss']:+.4f})")

        # Also compute model accuracy on the same subset where b365 is available
        b365_valid_mask = ~(parts["test"]["df"]["mkt_b365_p_home"].isna() |
                            parts["test"]["df"]["mkt_b365_p_draw"].isna() |
                            parts["test"]["df"]["mkt_b365_p_away"].isna())
        if b365_valid_mask.sum() < len(parts["test"]["df"]):
            clf_on_b365_subset = evaluate_classifier(
                clf,
                X_test[b365_valid_mask.values],
                parts["test"]["y_result"][b365_valid_mask.values],
            )
            print(f"\n    On b365-available subset ({b365_valid_mask.sum()} matches):")
            print(f"    Model acc:  {clf_on_b365_subset['accuracy']:.4f}")
            print(f"    Model ll:   {clf_on_b365_subset['log_loss']:.4f}")

    # --- Feature importance ---
    print("\n" + "=" * 70)
    print("8. FEATURE IMPORTANCE")
    print("=" * 70)
    print_feature_importance(clf, feature_cols, top_n=15, label="Classifier")
    print_feature_importance(reg_h, feature_cols, top_n=10, label="Regressor — Home Goals")
    print_feature_importance(reg_a, feature_cols, top_n=10, label="Regressor — Away Goals")

    # --- Save ---
    print("\n" + "=" * 70)
    print("9. SAVING ARTIFACTS")
    print("=" * 70)

    metrics = {
        "classifier": clf_metrics,
        "regressor_home_goals": reg_h_metrics,
        "regressor_away_goals": reg_a_metrics,
        "total_goals": {"rmse": float(total_rmse), "mae": float(total_mae)},
        "baseline_b365": b365,
    }
    save_artifacts(clf, reg_h, reg_a, feature_cols, league_encoder, parts, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
