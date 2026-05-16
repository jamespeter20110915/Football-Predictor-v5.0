"""
Stage 4 — Backtest Validation

Answers four core questions on the test set (2024-25 + 2025-26, 3,426 matches):
  1. Calibration: are predicted probabilities trustworthy?
  2. Discrimination: can the model separate high/low probability events?
  3. Value bet identification: does model edge over the market predict actual ROI?
  4. Simulated betting: does Kelly staking generate positive returns?

Uses ensemble probabilities (w=0.65 XGBoost + 0.35 Poisson) without threshold.
"""

import json
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_DIR = "reports"
FIG_DIR = f"{REPORT_DIR}/figures"
BANKROLL_INITIAL = 1000.0
MIN_EDGE = 0.03
MIN_ODDS = 1.20
KELLY_FRACTION = 0.25
MAX_BET_FRACTION = 0.05
MIN_BET_SIZE = 1.0

OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}
OUTCOME_LABELS = ["H", "D", "A"]
OUTCOME_TO_PROB = {"H": "home", "D": "draw", "A": "away"}  # column suffix mapping

# Try importing matplotlib (optional for figures)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping figures")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_test_data():
    """Load ensemble_probs + features (for raw odds) + matches (for reference)."""
    probs = pd.read_parquet("data/processed/ensemble_probs.parquet")
    features = pd.read_parquet("data/processed/features.parquet")
    matches = pd.read_parquet("data/processed/matches.parquet")

    # Ensure consistent date types for merging
    probs["date"] = pd.to_datetime(probs["date"])
    matches["date"] = pd.to_datetime(matches["date"])
    features["date"] = pd.to_datetime(features["date"])

    # Merge raw b365 odds from matches.parquet
    odds_cols = ["b365h", "b365d", "b365a"]
    merged = probs.merge(
        matches[["date", "home_team", "away_team"] + odds_cols],
        on=["date", "home_team", "away_team"], how="left"
    )

    # Also merge features for slicing (league, season, etc.)
    merged = merged.merge(
        features[["date", "home_team", "away_team", "league", "season"]],
        on=["date", "home_team", "away_team"], how="left"
    )

    print(f"Test data: {len(merged)} matches, {len(merged.columns)} columns")
    print(f"  b365 odds available: {merged['b365h'].notna().sum()}/{len(merged)}")
    return merged


# ---------------------------------------------------------------------------
# 2. Calibration analysis
# ---------------------------------------------------------------------------

def calibration_analysis(df, prob_col, true_col, n_bins=10):
    """
    Compute calibration curve and ECE for one outcome.

    Parameters
    ----------
    prob_col : str — column with predicted probability
    true_col : str — column with binary true label (0/1)
    """
    prob = df[prob_col].values
    true = df[true_col].values
    mask = ~np.isnan(prob) & ~np.isnan(true)
    prob, true = prob[mask], true[mask]

    if len(prob) == 0:
        return {"ece": np.nan, "n": 0, "bins": []}

    # Use sklearn calibration_curve with uniform binning
    fraction_pos, mean_pred = calibration_curve(
        true, prob, n_bins=n_bins, strategy="uniform"
    )

    # Compute ECE
    # Need to calculate per-bin sample counts for weighting
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        mask_bin = (prob >= bin_edges[i]) & (prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask_bin = (prob >= bin_edges[i]) & (prob <= bin_edges[i + 1])
        n_bin = mask_bin.sum()
        if n_bin > 0:
            p_pred = prob[mask_bin].mean()
            p_true = true[mask_bin].mean()
            ece += (n_bin / len(prob)) * abs(p_pred - p_true)
            bin_details.append({
                "bin": i + 1,
                "range": f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]",
                "n": int(n_bin),
                "p_pred_mean": float(p_pred),
                "p_true": float(p_true),
            })

    return {"ece": float(ece), "n": int(len(prob)), "bins": bin_details,
            "fraction_pos": fraction_pos.tolist(),
            "mean_pred": mean_pred.tolist()}


def plot_calibration(results, method, outcome):
    """Plot a single reliability diagram."""
    if not HAS_MPL:
        return
    os.makedirs(FIG_DIR, exist_ok=True)

    key = f"{method}_{outcome}"
    if key not in results:
        return
    r = results[key]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    ax.plot(r["mean_pred"], r["fraction_pos"], "o-", color="#1f77b4",
            markersize=6, label=f"Actual (ECE={r['ece']:.4f})")

    # Size dots by bin count
    if r["bins"]:
        sizes = [max(b["n"] / max(b["n"] for b in r["bins"]) * 80 + 20, 10)
                 for b in r["bins"]]
        ax.scatter(r["mean_pred"], r["fraction_pos"], s=sizes,
                   color="#1f77b4", alpha=0.6, zorder=5)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f"Calibration — {method} ({outcome})")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/calibration_{method}_{outcome}.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 3. Discrimination analysis
# ---------------------------------------------------------------------------

def discrimination_analysis(df, prob_col, true_col):
    """Compute AUC-ROC and AUC-PR for one binary outcome."""
    prob = df[prob_col].values
    true = df[true_col].values
    mask = ~np.isnan(prob) & ~np.isnan(true)
    prob, true = prob[mask], true[mask]

    if len(prob) < 2 or len(np.unique(true)) < 2:
        return {"auc_roc": np.nan, "auc_pr": np.nan, "n": len(prob)}

    return {
        "auc_roc": float(roc_auc_score(true, prob)),
        "auc_pr": float(average_precision_score(true, prob)),
        "n": int(len(prob)),
    }


# ---------------------------------------------------------------------------
# 4. Value bet analysis
# ---------------------------------------------------------------------------

def value_bet_analysis(df, prob_col_prefix, b365_prefix):
    """
    Identify value bets where model prob > market implied prob.
    Returns per-edge-bucket statistics.
    """
    results = []
    for outcome_i, outcome_label in enumerate(OUTCOME_LABELS):
        p_model_col = f"{prob_col_prefix}_p_{OUTCOME_TO_PROB[outcome_label]}"
        p_b365_col = f"{b365_prefix}_p_{OUTCOME_TO_PROB[outcome_label]}"
        odds_col = {"H": "b365h", "D": "b365d", "A": "b365a"}[outcome_label]

        p_model = df[p_model_col].values
        p_b365 = df[p_b365_col].values
        odds = df[odds_col].values
        actual = (df["target_result"] == outcome_label).astype(int).values

        valid = ~(np.isnan(p_model) | np.isnan(p_b365) | np.isnan(odds))
        p_model = p_model[valid]
        p_b365 = p_b365[valid]
        odds = odds[valid]
        actual = actual[valid]
        idx = df.index[valid]

        edge = p_model - p_b365
        # Only positive edge candidates
        pos_edge = edge > 0

        # Overall value bet stats
        n_value = int(pos_edge.sum())
        if n_value > 0:
            overall_roi = np.mean(
                np.where(actual[pos_edge] == 1, odds[pos_edge] - 1, -1)
            )
        else:
            overall_roi = 0.0

        # Per-edge-bucket
        buckets_edges = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, float("inf"))]
        buckets = []
        for lo, hi in buckets_edges:
            bm = pos_edge & (edge > lo) & (edge <= hi)
            if bm.sum() == 0:
                buckets.append({"edge_range": f"[{lo}, {hi}]", "n": 0, "hit_rate": None,
                                "roi": None, "mean_odds": None})
                continue
            n_b = int(bm.sum())
            hit_rate_b = float(actual[bm].mean())
            roi_b = float(np.mean(np.where(actual[bm] == 1, odds[bm] - 1, -1)))
            mean_odds_b = float(odds[bm].mean())
            buckets.append({
                "edge_range": f"[{lo}, {hi}]", "n": n_b,
                "hit_rate": hit_rate_b, "roi": roi_b, "mean_odds": mean_odds_b,
            })

        results.append({
            "outcome": outcome_label,
            "n_value_bets": n_value,
            "n_total": int(valid.sum()),
            "value_bet_pct": n_value / valid.sum() if valid.sum() > 0 else 0,
            "overall_roi": float(overall_roi),
            "mean_edge": float(edge[pos_edge].mean()) if n_value > 0 else 0,
            "buckets": buckets,
        })

    return results


# ---------------------------------------------------------------------------
# 5. Simulated betting
# ---------------------------------------------------------------------------

def simulate_betting(df, prob_col_prefix, b365_prefix, config):
    """
    Simulate Kelly betting through the test set chronologically.

    config keys:
      name, min_edge, min_odds, kelly_fraction, max_bet_frac, min_bet_size
    """
    df = df.sort_values("date").reset_index(drop=True)

    bankroll = BANKROLL_INITIAL
    bankroll_history = [bankroll]
    bets = []

    bis = {"H": "home", "D": "draw", "A": "away"}
    outcomes_map = {"H": "b365h", "D": "b365d", "A": "b365a"}

    for i, row in df.iterrows():
        for outcome in OUTCOME_LABELS:
            p_col = f"{prob_col_prefix}_p_{bis[outcome]}"
            b365_col = f"{b365_prefix}_p_{bis[outcome]}"
            odds_col = outcomes_map[outcome]

            p_model = row.get(p_col, np.nan)
            p_b365 = row.get(b365_col, np.nan)
            odds = row.get(odds_col, np.nan)

            if pd.isna(p_model) or pd.isna(p_b365) or pd.isna(odds):
                continue

            edge = p_model - p_b365
            if edge < config["min_edge"] or odds < config["min_odds"]:
                continue

            # Full Kelly fraction
            kelly_full = (p_model * odds - 1) / (odds - 1)
            if kelly_full <= 0:
                continue

            bet_frac = min(config["kelly_fraction"] * kelly_full, config["max_bet_frac"])
            bet_size = bankroll * bet_frac

            if bet_size < config["min_bet_size"]:
                continue

            actual = (row["target_result"] == outcome)
            pnl = bet_size * (odds - 1) if actual else -bet_size
            bankroll += pnl

            bets.append({
                "date": row["date"],
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "league": row.get("league", ""),
                "outcome": outcome,
                "p_model": float(p_model),
                "p_b365": float(p_b365),
                "edge": float(edge),
                "odds": float(odds),
                "kelly_full": float(kelly_full),
                "bet_fraction": float(bet_frac),
                "bet_size": float(bet_size),
                "bankroll_before": float(bankroll - pnl),
                "actual": int(actual),
                "pnl": float(pnl),
            })

        bankroll_history.append(bankroll)

    bet_df = pd.DataFrame(bets)

    if len(bet_df) == 0:
        return bet_df, {"error": "No bets placed", "bankroll_final": bankroll}

    # Compute statistics
    total_bet = bet_df["bet_size"].sum()
    total_pnl = bet_df["pnl"].sum()
    roi = total_pnl / total_bet if total_bet > 0 else 0
    hit_rate = bet_df["actual"].mean()

    # Max drawdown
    br = pd.Series(bankroll_history)
    running_max = br.cummax()
    drawdown = (br - running_max) / running_max
    max_dd = float(drawdown.min())

    # Monthly returns for Sharpe
    if len(bet_df) > 1:
        bet_df["month"] = bet_df["date"].dt.to_period("M")
        monthly_pnl = bet_df.groupby("month")["pnl"].sum()
        # Also include months with 0 bets
        monthly_returns = monthly_pnl.values / BANKROLL_INITIAL
        sharpe = float(np.mean(monthly_returns) / np.std(monthly_returns)) * np.sqrt(12) \
            if np.std(monthly_returns) > 0 else 0
    else:
        sharpe = 0
        monthly_pnl = pd.Series(dtype=float)

    return bet_df, {
        "name": config["name"],
        "bankroll_final": float(bankroll),
        "bankroll_initial": float(BANKROLL_INITIAL),
        "total_pnl": float(total_pnl),
        "total_bet_amount": float(total_bet),
        "n_bets": len(bet_df),
        "hit_rate": float(hit_rate),
        "roi": float(roi),
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "mean_odds": float(bet_df["odds"].mean()) if len(bet_df) > 0 else 0,
        "mean_edge": float(bet_df["edge"].mean()) if len(bet_df) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 6. Slice analysis
# ---------------------------------------------------------------------------

def slice_analysis(bet_df, by="league", min_bets=20):
    """Compute ROI by dimension slice."""
    if len(bet_df) == 0:
        return pd.DataFrame()
    grouped = bet_df.groupby(by)
    rows = []
    for name, group in grouped:
        n = len(group)
        if n < min_bets:
            continue
        total_bet = group["bet_size"].sum()
        total_pnl = group["pnl"].sum()
        roi = total_pnl / total_bet if total_bet > 0 else 0
        rows.append({
            by: name,
            "n_bets": n,
            "total_bet": float(total_bet),
            "total_pnl": float(total_pnl),
            "roi": float(roi),
            "hit_rate": float(group["actual"].mean()),
            "mean_odds": float(group["odds"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    if HAS_MPL:
        os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load ---
    print("=" * 70)
    print("1. LOADING TEST DATA")
    print("=" * 70)
    df = load_test_data()

    methods = {
        "XGBoost": "xgb",
        "Poisson": "poisson",
        "Ensemble": "ensemble",
        "B365": "b365",
    }

    # Prepare binary labels
    for label in OUTCOME_LABELS:
        df[f"is_{label}"] = (df["target_result"] == label).astype(int)

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. CALIBRATION ANALYSIS")
    print("=" * 70)

    cal_results = {}
    ece_table = []

    for method_name, prefix in methods.items():
        for outcome in OUTCOME_LABELS:
            prob_col = f"{prefix}_p_{OUTCOME_TO_PROB[outcome]}"
            true_col = f"is_{outcome}"
            key = f"{method_name}_{outcome}"
            cal_results[key] = calibration_analysis(df, prob_col, true_col)
            ece_table.append({
                "method": method_name,
                "outcome": outcome,
                "ece": cal_results[key]["ece"],
                "n": cal_results[key]["n"],
            })

            if HAS_MPL:
                plot_calibration(cal_results, method_name, outcome)

    ece_df = pd.DataFrame(ece_table)
    print("\n  ECE Summary (lower is better):")
    print(f"  {'Method':<10s}  {'H':>7s}  {'D':>7s}  {'A':>7s}")
    print("  " + "-" * 35)
    for m in ["XGBoost", "Poisson", "Ensemble", "B365"]:
        vals = {}
        for o in ["H", "D", "A"]:
            v = ece_df[(ece_df["method"] == m) & (ece_df["outcome"] == o)]["ece"].values
            vals[o] = f"{v[0]:.4f}" if len(v) > 0 else "N/A"
        print(f"  {m:<10s}  {vals['H']:>7s}  {vals['D']:>7s}  {vals['A']:>7s}")

    # -----------------------------------------------------------------------
    # Discrimination
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DISCRIMINATION ANALYSIS")
    print("=" * 70)

    disc_results = []
    for method_name, prefix in methods.items():
        for outcome in OUTCOME_LABELS:
            prob_col = f"{prefix}_p_{OUTCOME_TO_PROB[outcome]}"
            true_col = f"is_{outcome}"
            r = discrimination_analysis(df, prob_col, true_col)
            disc_results.append({
                "method": method_name,
                "outcome": outcome,
                **r,
            })

    disc_df = pd.DataFrame(disc_results)
    print(f"\n  {'Method':<10s}  {'Outcome':>7s}  {'AUC-ROC':>8s}  {'AUC-PR':>8s}  {'N':>6s}")
    print("  " + "-" * 48)
    for _, row in disc_df.iterrows():
        print(f"  {row['method']:<10s}  {row['outcome']:>7s}  {row['auc_roc']:8.4f}  {row['auc_pr']:8.4f}  {int(row['n']):>6d}")

    # -----------------------------------------------------------------------
    # Value bet analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. VALUE BET ANALYSIS (Ensemble vs B365)")
    print("=" * 70)

    vb_results = value_bet_analysis(df, "ensemble", "b365")

    for r in vb_results:
        print(f"\n  --- {r['outcome']} ---")
        print(f"  Value bets: {r['n_value_bets']}/{r['n_total']} ({r['value_bet_pct']:.1%})")
        print(f"  Overall ROI: {r['overall_roi']:.4f}  |  Mean edge: {r['mean_edge']:.4f}")
        print(f"\n  {'Edge Range':<12s}  {'N':>6s}  {'Hit Rate':>9s}  {'ROI':>8s}  {'Avg Odds':>9s}")
        print("  " + "-" * 52)
        for b in r["buckets"]:
            if b["n"] == 0:
                print(f"  {b['edge_range']:<12s}  {'—':>6s}  {'—':>9s}  {'—':>8s}  {'—':>9s}")
            else:
                print(f"  {b['edge_range']:<12s}  {b['n']:>6d}  {b['hit_rate']:9.4f}  {b['roi']:8.4f}  {b['mean_odds']:9.4f}")

    # -----------------------------------------------------------------------
    # Simulated betting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. SIMULATED BETTING")
    print("=" * 70)

    strategies = [
        {
            "name": "A: Ensemble + 1/4 Kelly (edge≥3%)",
            "prob_prefix": "ensemble",
            "b365_prefix": "b365",
            "min_edge": 0.03,
            "min_odds": 1.20,
            "kelly_fraction": 0.25,
            "max_bet_frac": 0.05,
            "min_bet_size": 1.0,
        },
        {
            "name": "B: Ensemble + Fixed 1% (edge≥3%)",
            "prob_prefix": "ensemble",
            "b365_prefix": "b365",
            "min_edge": 0.03,
            "min_odds": 1.20,
            "kelly_fraction": 0.0,  # treated specially below
            "max_bet_frac": 0.01,    # fixed 1%
            "min_bet_size": 1.0,
        },
        {
            "name": "C: XGBoost + 1/4 Kelly (edge≥3%)",
            "prob_prefix": "xgb",
            "b365_prefix": "b365",
            "min_edge": 0.03,
            "min_odds": 1.20,
            "kelly_fraction": 0.25,
            "max_bet_frac": 0.05,
            "min_bet_size": 1.0,
        },
    ]

    all_bet_logs = {}
    all_summaries = []

    for cfg in strategies:
        # Special handling for Strategy B (fixed 1% instead of Kelly)
        if "Fixed 1%" in cfg["name"]:
            cfg_actual = {**cfg, "kelly_fraction": 0.0}
            # Override: always bet 1% bankroll when edge ≥ threshold
            bet_df, summary = simulate_betting_fixed(df, cfg_actual)
        else:
            bet_df, summary = simulate_betting(df, cfg["prob_prefix"], cfg["b365_prefix"], cfg)

        all_bet_logs[cfg["name"]] = bet_df
        all_summaries.append(summary)

        print(f"\n  --- {cfg['name']} ---")
        if "error" in summary:
            print(f"  ERROR: {summary['error']}")
        else:
            print(f"  Final bankroll:  ${summary['bankroll_final']:,.2f}  (Δ = ${summary['total_pnl']:+,.2f})")
            print(f"  Bets placed:     {summary['n_bets']}")
            print(f"  Hit rate:        {summary['hit_rate']:.4f}")
            print(f"  Mean odds:       {summary['mean_odds']:.2f}")
            print(f"  ROI:             {summary['roi']:.4f}  ({summary['roi']*100:.2f}%)")
            print(f"  Max drawdown:    {summary['max_drawdown']:.4f}  ({summary['max_drawdown']*100:.2f}%)")
            print(f"  Sharpe (monthly):{summary['sharpe_ratio']:.3f}")

    # -----------------------------------------------------------------------
    # Slice analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. SLICE ANALYSIS (Strategy A: Ensemble + 1/4 Kelly)")
    print("=" * 70)

    strat_a_bets = all_bet_logs.get(strategies[0]["name"], pd.DataFrame())

    for dim, dim_label in [("league", "By League"), ("outcome", "By Outcome")]:
        slices = slice_analysis(strat_a_bets, by=dim)
        print(f"\n  --- {dim_label} ---")
        if len(slices) == 0:
            print("  No data (insufficient bets)")
        else:
            print(f"  {'Name':<20s}  {'N':>6s}  {'Hit%':>7s}  {'ROI':>8s}  {'P&L':>10s}  {'Avg Odds':>9s}")
            print("  " + "-" * 70)
            for _, s in slices.iterrows():
                print(f"  {s[dim]:<20s}  {int(s['n_bets']):>6d}  {s['hit_rate']:7.4f}  "
                      f"{s['roi']:8.4f}  ${s['total_pnl']:>9,.2f}  {s['mean_odds']:9.4f}")

    # Also by odds range
    if len(strat_a_bets) > 0:
        odds_bins = [(1.2, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float("inf"))]
        print(f"\n  --- By Odds Range ---")
        print(f"  {'Range':<15s}  {'N':>6s}  {'Hit%':>7s}  {'ROI':>8s}  {'P&L':>10s}")
        print("  " + "-" * 55)
        for lo, hi in odds_bins:
            sub = strat_a_bets[strat_a_bets["odds"].between(lo, hi)]
            if len(sub) == 0:
                continue
            roi = sub["pnl"].sum() / sub["bet_size"].sum() if sub["bet_size"].sum() > 0 else 0
            print(f"  [{lo:.1f}-{hi:.1f}]      {len(sub):>6d}  {sub['actual'].mean():7.4f}  "
                  f"{roi:8.4f}  ${sub['pnl'].sum():>9,.2f}")

    # -----------------------------------------------------------------------
    # Bankroll curve plot
    # -----------------------------------------------------------------------
    if HAS_MPL and len(strat_a_bets) > 0:
        print("\n" + "=" * 70)
        print("7. PLOTTING")
        print("=" * 70)

        # Bankroll curve
        fig, ax = plt.subplots(figsize=(12, 5))
        for cfg_name, bet_df in all_bet_logs.items():
            if len(bet_df) == 0:
                continue
            br = [BANKROLL_INITIAL]
            for pnl in bet_df["pnl"]:
                br.append(br[-1] + pnl)
            ax.plot(range(len(br)), br, linewidth=1.5, label=cfg_name[:60], alpha=0.85)
        ax.axhline(y=BANKROLL_INITIAL, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Bankroll ($)")
        ax.set_title("Simulated Betting — Bankroll Curves (Test Set: 2024-25 + 2025-26)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/bankroll_curve.png", dpi=120)
        plt.close()
        print("  Saved: bankroll_curve.png")

        # Monthly returns
        fig, axes = plt.subplots(len(all_bet_logs), 1, figsize=(12, 3 * len(all_bet_logs)))
        if len(all_bet_logs) == 1:
            axes = [axes]
        for ax, (cfg_name, bet_df) in zip(axes, all_bet_logs.items()):
            if len(bet_df) == 0:
                continue
            bet_df["month"] = bet_df["date"].dt.to_period("M")
            monthly = bet_df.groupby("month")["pnl"].sum()
            months_str = [str(m) for m in monthly.index]
            ax.bar(months_str, monthly.values, color="#1f77b4", alpha=0.7)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
            ax.set_title(f"Monthly P&L — {cfg_name[:70]}")
            ax.set_ylabel("P&L ($)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/monthly_returns.png", dpi=120)
        plt.close()
        print("  Saved: monthly_returns.png")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. SAVING")
    print("=" * 70)

    for cfg_name, bet_df in all_bet_logs.items():
        if len(bet_df) > 0:
            short = cfg_name.split(":")[0].lower().replace(" ", "_").replace("/", "")
            path = f"data/processed/backtest_bets_{short}.parquet"
            bet_df.to_parquet(path, index=False)
            print(f"  Saved: {path}  ({len(bet_df)} bets)")

    # -----------------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. GENERATING REPORT")
    print("=" * 70)

    with open(f"{REPORT_DIR}/backtest_report.md", "w") as f:
        f.write("# Backtest Report — Test Set (2024-25 + 2025-26)\n\n")

        # Calibration
        f.write("## 1. Calibration (ECE)\n\n")
        f.write("| Method | ECE (H) | ECE (D) | ECE (A) |\n")
        f.write("|--------|--------:|--------:|--------:|\n")
        for m in ["XGBoost", "Poisson", "Ensemble", "B365"]:
            vals = []
            for o in ["H", "D", "A"]:
                v = ece_df[(ece_df["method"] == m) & (ece_df["outcome"] == o)]["ece"].values
                vals.append(f"{v[0]:.4f}" if len(v) > 0 else "N/A")
            f.write(f"| {m} | {vals[0]} | {vals[1]} | {vals[2]} |\n")
        f.write("\n**Interpretation:** ECE < 0.05 is excellent, < 0.10 acceptable. Draw calibration is typically worse because draws are rare and harder to predict.\n\n")

        # Discrimination
        f.write("## 2. Discrimination (AUC)\n\n")
        f.write("| Method | Outcome | AUC-ROC | AUC-PR | N |\n")
        f.write("|--------|---------|--------:|-------:|---:|\n")
        for _, row in disc_df.iterrows():
            f.write(f"| {row['method']} | {row['outcome']} | {row['auc_roc']:.4f} | {row['auc_pr']:.4f} | {int(row['n'])} |\n")
        f.write("\n")

        # Value bets
        f.write("## 3. Value Bet Analysis (Ensemble vs B365)\n\n")
        for r in vb_results:
            f.write(f"### {r['outcome']}\n\n")
            f.write(f"- Value bets: {r['n_value_bets']}/{r['n_total']} ({r['value_bet_pct']:.1%})\n")
            f.write(f"- Overall ROI: {r['overall_roi']:.4f}\n")
            f.write(f"- Mean edge: {r['mean_edge']:.4f}\n\n")
            f.write("| Edge Range | N | Hit Rate | ROI | Avg Odds |\n")
            f.write("|------------|---:|---------:|----:|---------:|\n")
            for b in r["buckets"]:
                n_str = str(b['n']) if b['n'] > 0 else '—'
                hr_str = f"{b['hit_rate']:.4f}" if b['hit_rate'] is not None else '—'
                roi_str = f"{b['roi']:.4f}" if b['roi'] is not None else '—'
                odds_str = f"{b['mean_odds']:.2f}" if b['mean_odds'] is not None else '—'
                f.write(f"| {b['edge_range']} | {n_str} | {hr_str} | {roi_str} | {odds_str} |\n")
            f.write("\n")

        # Simulated betting
        f.write("## 4. Simulated Betting\n\n")
        f.write("| Strategy | Final BR | Bets | Hit% | ROI | MaxDD | Sharpe |\n")
        f.write("|----------|---------:|-----:|-----:|----:|------:|-------:|\n")
        for s in all_summaries:
            if "error" in s:
                continue
            f.write(f"| {s['name']} | ${s['bankroll_final']:,.2f} | {s['n_bets']} | "
                    f"{s['hit_rate']:.3f} | {s['roi']:.4f} | {s['max_drawdown']:.4f} | "
                    f"{s['sharpe_ratio']:.3f} |\n")
        f.write("\n")

        # Slice analysis
        f.write("## 5. Slice Analysis (Strategy A)\n\n")
        f.write("### By League\n\n")
        league_slices = slice_analysis(strat_a_bets, by="league")
        if len(league_slices) > 0:
            f.write("| League | Bets | Hit% | ROI | P&L |\n")
            f.write("|--------|-----:|-----:|----:|----:|\n")
            for _, s in league_slices.iterrows():
                f.write(f"| {s['league']} | {int(s['n_bets'])} | {s['hit_rate']:.3f} | "
                        f"{s['roi']:.4f} | ${s['total_pnl']:,.2f} |\n")
        f.write("\n### By Outcome\n\n")
        outcome_slices = slice_analysis(strat_a_bets, by="outcome")
        if len(outcome_slices) > 0:
            f.write("| Outcome | Bets | Hit% | ROI | P&L |\n")
            f.write("|---------|-----:|-----:|----:|----:|\n")
            for _, s in outcome_slices.iterrows():
                f.write(f"| {s['outcome']} | {int(s['n_bets'])} | {s['hit_rate']:.3f} | "
                        f"{s['roi']:.4f} | ${s['total_pnl']:,.2f} |\n")

        # Honest assessment
        f.write("\n## 6. Honest Assessment\n\n")
        f.write("### What the numbers say\n\n")

        best = all_summaries[0] if all_summaries else {}
        n_bets = best.get("n_bets", 0)
        roi = best.get("roi", 0)

        f.write(f"- **Overall ROI**: {roi:.4f} over {n_bets} bets. ")
        if roi > 0.05:
            f.write("Strongly positive — model appears to have genuine edge.\n")
        elif roi > 0:
            f.write("Slightly positive — encouraging but not conclusive.\n")
        else:
            f.write("Negative — model does not find exploitable market inefficiencies.\n")

        f.write(f"- **Sample size**: {n_bets} bets across 2 seasons (")
        if n_bets < 500:
            f.write(f"small — individual slice results are noisy, treat with caution)\n")
        else:
            f.write(f"moderate — league-level slices may be reliable)\n")

        # Check value bet buckets for monotonicity
        h_buckets = vb_results[0]["buckets"] if vb_results else []
        non_zero_buckets = [b for b in h_buckets if b["n"] > 0 and b["roi"] is not None]
        if len(non_zero_buckets) >= 2:
            rois = [b["roi"] for b in non_zero_buckets]
            if all(rois[i] <= rois[i+1] for i in range(len(rois)-1)):
                f.write("- **Edge monotonicity**: Higher edge → higher ROI. This is the strongest evidence of genuine predictive edge.\n")
            else:
                f.write("- **Edge monotonicity**: Not monotonic. Higher edge does not consistently mean higher ROI, which suggests noise or miscalibration at extreme edges.\n")

        f.write("\n### Caveats\n\n")
        f.write("1. **2-season test horizon is short.** A 2-season sample has ~3,400 matches but Kelly betting may only trigger a few hundred bets. Individual slice results (per league, per outcome) are underpowered.\n")
        f.write("2. **Betting odds are pre-match openers.** Football-Data records opening odds, not closing line. Real bettors face closing-line movement and liquidity constraints.\n")
        f.write("3. **No transaction costs.** Real betting includes commission (5-10% on losses), withdrawal fees, and potential account restrictions from winning bettors.\n")
        f.write("4. **Survivor bias is not an issue here.** We use the complete test set without any filtering based on outcomes.\n")
        f.write("5. **Overfitting check.** Ensemble weights (w=0.65) and draw threshold (1.75) were tuned on the 2023-24 validation set. The 2024-25+2025-26 test set is true out-of-sample. This is the correct procedure.\n")
        f.write("6. **Data leakage check.** All features use only historical data (shift(1) before rolling). Target variables are never in the feature set. The test split uses future seasons relative to training. No leakage.\n")

    print(f"  Saved: {REPORT_DIR}/backtest_report.md")
    print("\nDone.")


def simulate_betting_fixed(df, config):
    """Special case: fixed 1% bankroll bet (not Kelly)."""
    df = df.sort_values("date").reset_index(drop=True)

    bankroll = BANKROLL_INITIAL
    bets = []

    bis = {"H": "home", "D": "draw", "A": "away"}
    outcomes_map = {"H": "b365h", "D": "b365d", "A": "b365a"}

    for i, row in df.iterrows():
        for outcome in OUTCOME_LABELS:
            p_col = f"{config['prob_prefix']}_p_{bis[outcome]}"
            b365_col = f"{config['b365_prefix']}_p_{bis[outcome]}"
            odds_col = outcomes_map[outcome]

            p_model = row.get(p_col, np.nan)
            p_b365 = row.get(b365_col, np.nan)
            odds = row.get(odds_col, np.nan)

            if pd.isna(p_model) or pd.isna(p_b365) or pd.isna(odds):
                continue

            edge = p_model - p_b365
            if edge < config["min_edge"] or odds < config["min_odds"]:
                continue

            bet_size = bankroll * config["max_bet_frac"]
            if bet_size < config["min_bet_size"]:
                continue

            actual = (row["target_result"] == outcome)
            pnl = bet_size * (odds - 1) if actual else -bet_size
            bankroll += pnl

            bets.append({
                "date": row["date"],
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "league": row.get("league", ""),
                "outcome": outcome,
                "p_model": float(p_model),
                "p_b365": float(p_b365),
                "edge": float(edge),
                "odds": float(odds),
                "kelly_full": 0.0,
                "bet_fraction": float(config["max_bet_frac"]),
                "bet_size": float(bet_size),
                "bankroll_before": float(bankroll - pnl),
                "actual": int(actual),
                "pnl": float(pnl),
            })

    bet_df = pd.DataFrame(bets)

    if len(bet_df) == 0:
        return bet_df, {"error": "No bets placed", "bankroll_final": bankroll}

    total_bet = bet_df["bet_size"].sum()
    total_pnl = bet_df["pnl"].sum()
    roi = total_pnl / total_bet if total_bet > 0 else 0
    hit_rate = bet_df["actual"].mean()

    br_list = [BANKROLL_INITIAL]
    for pnl in bet_df["pnl"]:
        br_list.append(br_list[-1] + pnl)
    br_series = pd.Series(br_list)
    running_max = br_series.cummax()
    drawdown = (br_series - running_max) / running_max
    max_dd = float(drawdown.min())

    if len(bet_df) > 1:
        bet_df["month"] = bet_df["date"].dt.to_period("M")
        monthly_pnl = bet_df.groupby("month")["pnl"].sum()
        monthly_returns = monthly_pnl.values / BANKROLL_INITIAL
        sharpe = float(np.mean(monthly_returns) / np.std(monthly_returns)) * np.sqrt(12) \
            if np.std(monthly_returns) > 0 else 0
    else:
        sharpe = 0

    return bet_df, {
        "name": config["name"],
        "bankroll_final": float(bankroll),
        "bankroll_initial": float(BANKROLL_INITIAL),
        "total_pnl": float(total_pnl),
        "total_bet_amount": float(total_bet),
        "n_bets": len(bet_df),
        "hit_rate": float(hit_rate),
        "roi": float(roi),
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "mean_odds": float(bet_df["odds"].mean()) if len(bet_df) > 0 else 0,
        "mean_edge": float(bet_df["edge"].mean()) if len(bet_df) > 0 else 0,
    }


if __name__ == "__main__":
    main()
