"""
Stage 3.5 — Poisson-derived match probabilities + Ensemble

Uses the two trained Poisson regressors (home/away goals) to derive
match-result probabilities via the independent-Poisson scoreline model,
then blends with the XGBoost classifier via weighted average.

No model retraining required — inference only.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
)

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
REPORT_DIR = "reports"

ID_COLS = {"date", "league", "season", "home_team", "away_team"}
TARGET_COLS = {
    "target_result", "target_home_goals",
    "target_away_goals", "target_total_goals",
}

# ---------------------------------------------------------------------------
# 1. Data loading (identical split logic to train.py)
# ---------------------------------------------------------------------------

def load_and_split(features_path: str) -> dict:
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["season"])

    mask_train = df["season"].isin(TRAIN_SEASONS)
    mask_val = df["season"] == VAL_SEASON
    mask_test = df["season"].isin(TEST_SEASONS)

    parts = {}
    for name, mask in [("val", mask_val), ("test", mask_test)]:
        subset = df[mask].copy()
        parts[name] = {
            "df": subset,
            "y_result": subset["target_result"].map({"H": 0, "D": 1, "A": 2}).values.astype(int),
            "n": len(subset),
            "date_range": f"{subset['date'].min()}  →  {subset['date'].max()}",
        }
    # Also load train for reference (not used in search)
    parts["train"] = {"df": df[mask_train].copy()}

    for k in ["val", "test"]:
        print(f"  {k:5s}: {parts[k]['n']:5d} matches  [{parts[k]['date_range']}]")

    return parts


# ---------------------------------------------------------------------------
# 2. Load models & prepare features
# ---------------------------------------------------------------------------

def load_artifacts():
    """Load all three models + feature config."""
    clf = xgb.Booster()
    clf.load_model(f"{MODEL_DIR}/clf_result.json")

    reg_h = xgb.Booster()
    reg_h.load_model(f"{MODEL_DIR}/reg_home_goals.json")

    reg_a = xgb.Booster()
    reg_a.load_model(f"{MODEL_DIR}/reg_away_goals.json")

    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_cols = json.load(f)

    with open(f"{MODEL_DIR}/label_encoder.json") as f:
        le_data = json.load(f)

    print(f"Loaded 3 models, {len(feature_cols)} feature columns")
    return clf, reg_h, reg_a, feature_cols, le_data


def prepare_features(df, feature_cols, le_data):
    """Build feature matrix matching train.py exactly."""
    league_map = le_data["mapping"]
    league_encoded = df["league"].map(league_map)

    # Check for unseen leagues
    unseen = set(df["league"]) - set(league_map)
    if unseen:
        raise ValueError(f"Unseen leagues: {unseen}")

    drop_cols = [c for c in ID_COLS | TARGET_COLS if c in df.columns]
    X_df = df.drop(columns=drop_cols).copy()
    X_df["league_encoded"] = league_encoded
    X_df = X_df[feature_cols]  # enforce order

    return X_df.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Poisson match-result probabilities
# ---------------------------------------------------------------------------

def poisson_to_result_probs(lambda_h, lambda_a, max_goals=10):
    """
    Derive P(H) / P(D) / P(A) from independent Poisson goal models.

    Parameters
    ----------
    lambda_h, lambda_a : np.ndarray, shape (n_samples,)
        Predicted expected goals for home / away team.
    max_goals : int
        Truncation point for the Poisson scoreline matrix.

    Returns
    -------
    probs : np.ndarray, shape (n_samples, 3)
        Columns: [P(H), P(D), P(A)]  — matches XGBoost class order (H=0, D=1, A=2).
    """
    goals = np.arange(max_goals + 1)  # [0, 1, ..., 10]

    # Probability of each goal count for each sample
    pmf_h = poisson.pmf(goals[None, :], lambda_h[:, None])  # (n, G+1)
    pmf_a = poisson.pmf(goals[None, :], lambda_a[:, None])  # (n, G+1)

    # Scoreline probability cube: P(i, j) for each sample
    score_probs = pmf_h[:, :, None] * pmf_a[:, None, :]  # (n, G+1, G+1)

    # Masks for H / D / A
    i_idx, j_idx = np.meshgrid(goals, goals, indexing="ij")
    home_mask = i_idx > j_idx
    draw_mask = i_idx == j_idx
    away_mask = i_idx < j_idx

    p_home = score_probs[:, home_mask].sum(axis=1)
    p_draw = score_probs[:, draw_mask].sum(axis=1)
    p_away = score_probs[:, away_mask].sum(axis=1)

    probs = np.stack([p_home, p_draw, p_away], axis=1)
    probs /= probs.sum(axis=1, keepdims=True)  # renormalise after truncation
    return probs


# ---------------------------------------------------------------------------
# 4. Inference helpers
# ---------------------------------------------------------------------------

def predict_xgb_proba(model, X):
    """Get XGBoost probability predictions. Returns (n, 3) array."""
    dmat = xgb.DMatrix(X, missing=np.nan)
    return model.predict(dmat)  # shape (n, 3) for multi:softprob


def predict_poisson_proba(reg_h, reg_a, X):
    """Get Poisson-derived H/D/A probabilities."""
    dmat = xgb.DMatrix(X, missing=np.nan)
    lambda_h = reg_h.predict(dmat)
    lambda_a = reg_a.predict(dmat)
    return poisson_to_result_probs(lambda_h, lambda_a), lambda_h, lambda_a


# ---------------------------------------------------------------------------
# 5. Threshold tuning (optional: improve draw prediction)
# ---------------------------------------------------------------------------

def apply_draw_threshold(probs, threshold=1.0):
    """
    Apply draw bias to probability-based predictions.

    Standard: pred = argmax([p_h, p_d, p_a])
    With threshold t: if p_d * t > max(p_h, p_a), predict draw instead.

    Returns integer predictions 0/1/2.
    """
    preds = np.argmax(probs, axis=1).copy()
    # Where draw probability * threshold exceeds max of home/away, force draw
    other_max = np.maximum(probs[:, 0], probs[:, 2])
    force_draw = (probs[:, 1] * threshold) > other_max
    preds[force_draw] = 1
    return preds


def search_draw_threshold(probs, y_true):
    """Grid search for best draw threshold on validation set."""
    best_t, best_acc = 1.0, -1
    results = []
    for t in np.arange(1.0, 5.1, 0.25):
        preds = apply_draw_threshold(probs, t)
        acc = accuracy_score(y_true, preds)
        report = classification_report(y_true, preds, target_names=["H", "D", "A"],
                                        output_dict=True, zero_division=0)
        draw_recall = report["D"]["recall"]
        macro_f1 = report["macro avg"]["f1-score"]
        results.append({
            "threshold": round(t, 2),
            "accuracy": acc,
            "draw_recall": draw_recall,
            "macro_f1": macro_f1,
        })
        if macro_f1 > best_acc:
            best_acc = macro_f1
            best_t = t
    return best_t, best_acc, results


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------

def evaluate(probs, y_true, threshold=None):
    """Full evaluation of predicted probabilities."""
    if threshold is not None:
        preds = apply_draw_threshold(probs, threshold)
    else:
        preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, preds)
    ll = log_loss(y_true, probs)

    y_onehot = np.eye(3)[y_true]
    brier = np.mean([
        brier_score_loss(y_onehot[:, k], probs[:, k]) for k in range(3)
    ])

    cm = confusion_matrix(y_true, preds)
    report = classification_report(
        y_true, preds, target_names=["H", "D", "A"],
        output_dict=True, zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
        "per_class": {k: report[k] for k in ["H", "D", "A"]},
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    features_path = "data/processed/features.parquet"

    # -- Load data --
    print("=" * 70)
    print("1. DATA LOADING")
    print("=" * 70)
    parts = load_and_split(features_path)

    # -- Load models --
    print("\n" + "=" * 70)
    print("2. LOADING MODELS")
    print("=" * 70)
    clf, reg_h, reg_a, feature_cols, le_data = load_artifacts()

    # -- Prepare features --
    print("\n" + "=" * 70)
    print("3. FEATURE PREPARATION")
    print("=" * 70)
    X_val = prepare_features(parts["val"]["df"], feature_cols, le_data)
    X_test = prepare_features(parts["test"]["df"], feature_cols, le_data)
    print(f"  Val X:  {X_val.shape}")
    print(f"  Test X: {X_test.shape}")

    # -- Predict on validation set --
    print("\n" + "=" * 70)
    print("4. VALIDATION SET — WEIGHT SEARCH")
    print("=" * 70)

    p_xgb_val = predict_xgb_proba(clf, X_val)
    p_poisson_val, lam_h_val, lam_a_val = predict_poisson_proba(reg_h, reg_a, X_val)

    # Grid search for best ensemble weight
    w_range = np.arange(0.0, 1.01, 0.05)
    best_w, best_ll = 0.5, float("inf")
    weight_results = []

    for w in w_range:
        p_ens = w * p_xgb_val + (1 - w) * p_poisson_val
        ll = log_loss(parts["val"]["y_result"], p_ens)
        preds = np.argmax(p_ens, axis=1)
        acc = accuracy_score(parts["val"]["y_result"], preds)
        report = classification_report(
            parts["val"]["y_result"], preds,
            target_names=["H", "D", "A"], output_dict=True, zero_division=0,
        )
        weight_results.append({
            "w_xgb": round(float(w), 2),
            "log_loss": float(ll),
            "accuracy": float(acc),
            "draw_recall": float(report["D"]["recall"]),
            "draw_f1": float(report["D"]["f1-score"]),
        })
        if ll < best_ll:
            best_ll = ll
            best_w = w

    w_best = round(float(best_w), 2)
    print(f"  Best w_xgb = {w_best:.2f}  (log_loss = {best_ll:.4f})")

    # Weight search table
    print(f"\n  {'w_xgb':>6s}  {'log_loss':>9s}  {'acc':>7s}  {'draw_recall':>12s}  {'draw_f1':>8s}")
    print("  " + "-" * 55)
    for r in weight_results:
        marker = " ←" if r["w_xgb"] == w_best else ""
        print(f"  {r['w_xgb']:6.2f}  {r['log_loss']:9.4f}  {r['accuracy']:7.4f}  {r['draw_recall']:12.4f}  {r['draw_f1']:8.4f}{marker}")

    # -- Threshold tuning on ensemble probabilities --
    print("\n" + "=" * 70)
    print("5. VALIDATION SET — DRAW THRESHOLD TUNING")
    print("=" * 70)
    p_ens_val = w_best * p_xgb_val + (1 - w_best) * p_poisson_val
    best_t, best_macro_f1, threshold_results = search_draw_threshold(
        p_ens_val, parts["val"]["y_result"]
    )
    print(f"  Best draw threshold = {best_t:.2f}  (macro F1 = {best_macro_f1:.4f})")
    print(f"\n  {'thresh':>6s}  {'accuracy':>9s}  {'draw_recall':>12s}  {'macro_f1':>10s}")
    print("  " + "-" * 45)
    for r in threshold_results:
        m = " ←" if r["threshold"] == best_t else ""
        print(f"  {r['threshold']:6.2f}  {r['accuracy']:9.4f}  {r['draw_recall']:12.4f}  {r['macro_f1']:10.4f}{m}")

    # -- Test set final evaluation --
    print("\n" + "=" * 70)
    print("6. TEST SET — 4-METHOD COMPARISON")
    print("=" * 70)

    p_xgb_test = predict_xgb_proba(clf, X_test)
    p_poisson_test, lam_h_test, lam_a_test = predict_poisson_proba(reg_h, reg_a, X_test)
    p_ens_test = w_best * p_xgb_test + (1 - w_best) * p_poisson_test

    # B365 baseline
    df_test = parts["test"]["df"]
    b365_valid = ~(df_test["mkt_b365_p_home"].isna() |
                   df_test["mkt_b365_p_draw"].isna() |
                   df_test["mkt_b365_p_away"].isna())
    p_b365_test = df_test[["mkt_b365_p_home", "mkt_b365_p_draw",
                            "mkt_b365_p_away"]].values
    y_test_b365 = parts["test"]["y_result"][b365_valid]
    p_b365_test = p_b365_test[b365_valid]

    methods = {
        "XGBoost":   (p_xgb_test, None),
        "Poisson":   (p_poisson_test, None),
        "Ensemble":  (p_ens_test, best_t),      # apply best threshold
        "B365":      (p_b365_test, None),
    }

    y_test_full = parts["test"]["y_result"]

    all_metrics = {}
    print()
    for name, (probs, thresh) in methods.items():
        y_ref = y_test_b365 if name == "B365" else y_test_full
        metrics = evaluate(probs, y_ref, threshold=thresh)
        all_metrics[name] = metrics
        cm = metrics["confusion_matrix"]
        print(f"  ┌─ {name} ──────────────────────────────────────────────")
        print(f"  │ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  │ Log Loss:  {metrics['log_loss']:.4f}")
        print(f"  │ Brier:     {metrics['brier_score']:.4f}")
        print(f"  │ Confusion Matrix (rows=true, cols=pred):")
        print(f"  │            Pred H   Pred D   Pred A")
        for i, label in enumerate(["True H", "True D", "True A"]):
            print(f"  │   {label}  {cm[i][0]:>7d}  {cm[i][1]:>7d}  {cm[i][2]:>7d}")
        print(f"  │ Per-class:")
        for cls in ["H", "D", "A"]:
            m = metrics["per_class"][cls]
            print(f"  │   {cls}:  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}  N={int(m['support'])}")
        print(f"  └{'─' * 58}")

    # -- Key comparison summary --
    print("\n" + "=" * 70)
    print("7. SUMMARY — DRAW RECALL IMPROVEMENT")
    print("=" * 70)
    print(f"\n  {'Method':<12s}  {'Accuracy':>9s}  {'Log Loss':>9s}  {'Draw R':>8s}  {'Draw F1':>8s}")
    print("  " + "-" * 55)
    for name, m in all_metrics.items():
        print(f"  {name:<12s}  {m['accuracy']:9.4f}  {m['log_loss']:9.4f}  "
              f"{m['per_class']['D']['recall']:8.4f}  {m['per_class']['D']['f1-score']:8.4f}")

    # Improvement deltas
    xgb_draw_r = all_metrics["XGBoost"]["per_class"]["D"]["recall"]
    ens_draw_r = all_metrics["Ensemble"]["per_class"]["D"]["recall"]
    print(f"\n  Draw recall improvement: {xgb_draw_r:.4f}  →  {ens_draw_r:.4f}  "
          f"(× {ens_draw_r/xgb_draw_r if xgb_draw_r > 0 else float('inf'):.1f})")

    # -- Save artifacts --
    print("\n" + "=" * 70)
    print("8. SAVING ARTIFACTS")
    print("=" * 70)

    # ensemble_config.json
    os.makedirs(MODEL_DIR, exist_ok=True)
    config = {
        "method": "weighted_average",
        "w_xgboost": w_best,
        "w_poisson": round(1 - w_best, 2),
        "draw_threshold": best_t,
        "max_goals_truncation": 10,
        "validation_log_loss": best_ll,
        "validation_accuracy": float(weight_results[[r["w_xgb"] for r in weight_results].index(w_best)]["accuracy"])
            if False else next(r["accuracy"] for r in weight_results if r["w_xgb"] == w_best),
        "validation_draw_recall": next(r["draw_recall"] for r in weight_results if r["w_xgb"] == w_best),
    }
    with open(f"{MODEL_DIR}/ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {MODEL_DIR}/ensemble_config.json")

    # ensemble_probs.parquet
    df_probs = df_test[["date", "league", "season", "home_team", "away_team"]].copy()
    df_probs["xgb_p_home"] = p_xgb_test[:, 0]
    df_probs["xgb_p_draw"] = p_xgb_test[:, 1]
    df_probs["xgb_p_away"] = p_xgb_test[:, 2]
    df_probs["poisson_p_home"] = p_poisson_test[:, 0]
    df_probs["poisson_p_draw"] = p_poisson_test[:, 1]
    df_probs["poisson_p_away"] = p_poisson_test[:, 2]
    df_probs["ensemble_p_home"] = p_ens_test[:, 0]
    df_probs["ensemble_p_draw"] = p_ens_test[:, 1]
    df_probs["ensemble_p_away"] = p_ens_test[:, 2]
    df_probs["b365_p_home"] = df_test["mkt_b365_p_home"].values
    df_probs["b365_p_draw"] = df_test["mkt_b365_p_draw"].values
    df_probs["b365_p_away"] = df_test["mkt_b365_p_away"].values
    df_probs["lambda_home"] = lam_h_test
    df_probs["lambda_away"] = lam_a_test
    df_probs["target_result"] = df_test["target_result"].values
    df_probs["target_home_goals"] = df_test["target_home_goals"].values
    df_probs["target_away_goals"] = df_test["target_away_goals"].values
    os.makedirs("data/processed", exist_ok=True)
    df_probs.to_parquet("data/processed/ensemble_probs.parquet", index=False)
    print(f"  Saved: data/processed/ensemble_probs.parquet  ({df_probs.shape})")

    # ensemble_evaluation.md
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/ensemble_evaluation.md", "w") as f:
        f.write("# Ensemble Evaluation Report\n\n")
        f.write(f"## Validation Set Weight Search\n\n")
        f.write(f"| w_xgb | Log Loss | Accuracy | Draw Recall | Draw F1 |\n")
        f.write(f"|------:|---------:|---------:|------------:|--------:|\n")
        for r in weight_results:
            flag = " ← best" if r["w_xgb"] == w_best else ""
            f.write(f"| {r['w_xgb']:.2f}{flag} | {r['log_loss']:.4f} | {r['accuracy']:.4f} | {r['draw_recall']:.4f} | {r['draw_f1']:.4f} |\n")

        f.write(f"\n## Draw Threshold Tuning\n\n")
        f.write(f"Best threshold: **{best_t:.2f}** (max macro F1 = {best_macro_f1:.4f})\n\n")
        f.write(f"| Threshold | Accuracy | Draw Recall | Macro F1 |\n")
        f.write(f"|----------:|---------:|------------:|---------:|\n")
        for r in threshold_results:
            flag = " ← best" if r["threshold"] == best_t else ""
            f.write(f"| {r['threshold']:.2f}{flag} | {r['accuracy']:.4f} | {r['draw_recall']:.4f} | {r['macro_f1']:.4f} |\n")

        f.write(f"\n## Test Set — 4-Method Comparison\n\n")
        f.write(f"| Method | Accuracy | Log Loss | Brier | Draw P | Draw R | Draw F1 |\n")
        f.write(f"|--------|---------:|---------:|------:|-------:|-------:|--------:|\n")
        for name in ["XGBoost", "Poisson", "Ensemble", "B365"]:
            m = all_metrics[name]
            d = m["per_class"]["D"]
            f.write(f"| {name} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier_score']:.4f} | "
                    f"{d['precision']:.3f} | {d['recall']:.3f} | {d['f1-score']:.3f} |\n")

        f.write(f"\n## Confusion Matrices\n\n")
        for name in ["XGBoost", "Poisson", "Ensemble", "B365"]:
            cm = all_metrics[name]["confusion_matrix"]
            f.write(f"### {name}\n\n")
            f.write(f"| | Pred H | Pred D | Pred A |\n")
            f.write(f"|---|---|---|---|\n")
            for i, label in enumerate(["True H", "True D", "True A"]):
                f.write(f"| {label} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |\n")
            f.write("\n")

        f.write(f"## Ensemble Configuration\n\n")
        f.write(f"- **Method:** weighted average\n")
        f.write(f"- **w_xgboost:** {w_best:.2f}\n")
        f.write(f"- **w_poisson:** {1-w_best:.2f}\n")
        f.write(f"- **Draw threshold:** {best_t:.2f}\n")
        f.write(f"- **max_goals_truncation:** 10\n\n")

        xgb_dr = all_metrics["XGBoost"]["per_class"]["D"]["recall"]
        ens_dr = all_metrics["Ensemble"]["per_class"]["D"]["recall"]
        f.write(f"## Key Finding\n\n")
        f.write(f"Ensemble improved draw recall from **{xgb_dr:.4f}** to **{ens_dr:.4f}** "
                f"({ens_dr/xgb_dr:.1f}× improvement) ")
        f.write(f"while maintaining competitive accuracy.\n")

    print(f"  Saved: {REPORT_DIR}/ensemble_evaluation.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
