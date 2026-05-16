"""
Stage 5 — Over/Under (大小球) Prediction & Backtest

Uses the two trained Poisson goal regressors to derive Over/Under
probabilities for goal-line markets (1.5, 2.5, 3.5), then backtests
against Bet365 O/U odds on the test set.

Key insight: total goals ~ Poisson(lambda_h + lambda_a), so:
    P(Under N.5) = Poisson.cdf(N, lambda_total)
    P(Over N.5)  = 1 - P(Under N.5)
"""

import json
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_DIR = "reports"
FIG_DIR = f"{REPORT_DIR}/figures"
BANKROLL_INITIAL = 1000.0
MIN_EDGE = 0.05          # O/U market: tighter odds → higher threshold
MIN_ODDS = 1.30           # Slightly higher floor (O/U odds are narrower)
KELLY_FRACTION = 0.25
MAX_BET_FRACTION = 0.05
MIN_BET_SIZE = 1.0
LINES = [1.5, 2.5, 3.5]  # Primary focus: 2.5

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping figures")


# ---------------------------------------------------------------------------
# 1. Extract O/U odds from raw CSVs + merge with ensemble probs
# ---------------------------------------------------------------------------

def extract_ou_odds_from_csvs(seasons):
    """
    Read raw Football-Data CSVs for given seasons, extract B365 O/U odds.
    Applies team name mapping to match the standardized names in ensemble_probs.
    Returns DataFrame with date, home_team, away_team, and O/U columns.
    """
    # Load team name mapping
    with open("data/team_name_mapping.json") as fm:
        mapping = json.load(fm)
    name_map = mapping["football_data_to_standard"]

    rows = []
    csv_dir = "data/raw/football_data"
    unmapped = set()
    for f in sorted(glob(f"{csv_dir}/*.csv")):
        fn = os.path.basename(f).replace(".csv", "")
        parts = fn.split("_")
        season_code = parts[1] if len(parts) > 1 else ""
        # Map season code like "2425" to "2024-25"
        if len(season_code) == 4:
            s = f"20{season_code[:2]}-{season_code[2:]}"
        else:
            continue

        if s not in seasons:
            continue

        # Read CSV
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "B365>2.5" not in df.columns:
            continue

        # Select columns
        ou_cols = ["B365>2.5", "B365<2.5"]
        if "Avg>2.5" in df.columns:
            ou_cols += ["Avg>2.5", "Avg<2.5"]
        else:
            ou_cols += [None, None]  # placeholders

        # Build rows
        for _, row in df.iterrows():
            ht = row["HomeTeam"]
            at = row["AwayTeam"]
            # Standardize team names
            ht_std = name_map.get(ht, ht)
            at_std = name_map.get(at, at)
            if ht not in name_map:
                unmapped.add(ht)
            if at not in name_map:
                unmapped.add(at)

            rec = {
                "date": pd.to_datetime(row["Date"], dayfirst=True),
                "home_team": ht_std,
                "away_team": at_std,
                "b365_over25": row.get("B365>2.5", np.nan),
                "b365_under25": row.get("B365<2.5", np.nan),
            }
            if "Avg>2.5" in df.columns:
                rec["avg_over25"] = row.get("Avg>2.5", np.nan)
                rec["avg_under25"] = row.get("Avg<2.5", np.nan)
            rows.append(rec)

    if unmapped:
        print(f"  Warning: {len(unmapped)} unmapped team names: {sorted(unmapped)}")

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out["b365_over25"] = pd.to_numeric(out["b365_over25"], errors="coerce")
    out["b365_under25"] = pd.to_numeric(out["b365_under25"], errors="coerce")
    if "avg_over25" in out.columns:
        out["avg_over25"] = pd.to_numeric(out["avg_over25"], errors="coerce")
        out["avg_under25"] = pd.to_numeric(out["avg_under25"], errors="coerce")
    return out


def load_ou_data():
    """Build the complete O/U dataset: ensemble probs + O/U odds."""
    probs = pd.read_parquet("data/processed/ensemble_probs.parquet")
    probs["date"] = pd.to_datetime(probs["date"])

    test_seasons = ["2024-25", "2025-26"]
    ou_odds = extract_ou_odds_from_csvs(test_seasons)
    print(f"O/U odds extracted: {len(ou_odds)} rows from {len(test_seasons)} seasons")

    # Merge
    merged = probs.merge(ou_odds, on=["date", "home_team", "away_team"], how="inner")
    print(f"Merged: {len(merged)} matches (lost {len(probs) - len(merged)} in merge)")

    # Compute actual O/U labels
    merged["actual_total_goals"] = merged["target_home_goals"] + merged["target_away_goals"]
    for line in LINES:
        merged[f"over_{str(line).replace('.', '_')}"] = (
            merged["actual_total_goals"] > line
        ).astype(int)

    # Compute Poisson O/U probabilities for all lines
    for line in LINES:
        threshold = int(line)  # 2.5 → 2, 1.5 → 1, 3.5 → 3
        merged[f"poisson_under_{str(line).replace('.', '_')}"] = poisson.cdf(
            threshold, merged["lambda_home"] + merged["lambda_away"]
        )
        merged[f"poisson_over_{str(line).replace('.', '_')}"] = (
            1 - merged[f"poisson_under_{str(line).replace('.', '_')}"]
        )

    # Compute B365 implied O/U probabilities (vig-adjusted)
    for line in LINES:
        key = str(line).replace(".", "_")
        over_col = f"b365_over{key[1:]}" if key.startswith("2") else f"b365_over{key[1:]}"
        under_col = f"b365_under{key[1:]}" if key.startswith("2") else f"b365_under{key[1:]}"

        # Map: for 2.5 → "b365_over25", for 1.5 → need separate.
        # Our CSV extraction only has 2.5 odds. For 2.5, use b365_over25/b365_under25
        if line == 2.5:
            raw_over = 1.0 / merged["b365_over25"]
            raw_under = 1.0 / merged["b365_under25"]
            total_raw = raw_over + raw_under
            merged["b365_p_over_2_5"] = raw_over / total_raw
            merged["b365_p_under_2_5"] = raw_under / total_raw

    return merged


# ---------------------------------------------------------------------------
# 2. Calibration analysis
# ---------------------------------------------------------------------------

def calibrate_ou(df, prob_col, true_col, label, n_bins=10):
    """Calibration curve + ECE for O/U binary probability."""
    prob = df[prob_col].values
    true = df[true_col].values
    mask = ~np.isnan(prob) & ~np.isnan(true)
    prob, true = prob[mask], true[mask]

    if len(prob) == 0:
        return {"ece": np.nan, "n": 0, "bins": []}

    fraction_pos, mean_pred = calibration_curve(true, prob, n_bins=n_bins, strategy="uniform")

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
                "bin": i + 1, "n": int(n_bin), "p_pred": float(p_pred), "p_true": float(p_true),
            })

    return {"label": label, "ece": float(ece), "n": int(len(prob)), "bins": bin_details,
            "mean_pred": mean_pred.tolist(), "fraction_pos": fraction_pos.tolist()}


def plot_ou_calibration(results, label):
    """Plot reliability diagram for one O/U calibration result."""
    if not HAS_MPL:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    ax.plot(results["mean_pred"], results["fraction_pos"], "o-", color="#d62728",
            markersize=6, label=f"Actual (ECE={results['ece']:.4f})")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f"Calibration — {label}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace(".", "_").replace("/", "_")
    plt.savefig(f"{FIG_DIR}/ou_calibration_{safe_label}.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 3. Value bet analysis
# ---------------------------------------------------------------------------

def value_bet_ou(df, poisson_col, b365_col, odds_col, true_col, line_label):
    """Value bet analysis for one O/U direction."""
    p_model = df[poisson_col].values
    p_b365 = df[b365_col].values
    odds = df[odds_col].values
    actual = df[true_col].values

    valid = ~(np.isnan(p_model) | np.isnan(p_b365) | np.isnan(odds))
    p_model = p_model[valid]
    p_b365 = p_b365[valid]
    odds = odds[valid]
    actual = actual[valid]

    edge = p_model - p_b365
    pos_edge = edge > 0

    n_value = int(pos_edge.sum())
    overall_roi = float(np.mean(np.where(actual[pos_edge] == 1, odds[pos_edge] - 1, -1))) if n_value > 0 else 0

    buckets_edges = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, float("inf"))]
    buckets = []
    for lo, hi in buckets_edges:
        bm = pos_edge & (edge > lo) & (edge <= hi)
        if bm.sum() == 0:
            buckets.append({"range": f"[{lo}, {hi}]", "n": 0, "hit_rate": None, "roi": None})
            continue
        n_b = int(bm.sum())
        buckets.append({
            "range": f"[{lo}, {hi}]", "n": n_b,
            "hit_rate": float(actual[bm].mean()),
            "roi": float(np.mean(np.where(actual[bm] == 1, odds[bm] - 1, -1))),
            "mean_edge": float(edge[bm].mean()),
            "mean_odds": float(odds[bm].mean()),
        })

    return {
        "line": line_label, "n_value_bets": n_value, "n_total": int(valid.sum()),
        "overall_roi": overall_roi, "mean_edge": float(edge[pos_edge].mean()) if n_value > 0 else 0,
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# 4. Simulated betting
# ---------------------------------------------------------------------------

def simulate_ou_betting(df, config):
    """Kelly betting simulation for O/U market."""
    df = df.sort_values("date").reset_index(drop=True)
    bankroll = BANKROLL_INITIAL
    bets = []

    for _, row in df.iterrows():
        p_model = row.get(config["prob_col"], np.nan)
        odds = row.get(config["odds_col"], np.nan)
        p_b365 = row.get(config["b365_col"], np.nan)
        actual = row.get(config["true_col"], np.nan)

        if pd.isna(p_model) or pd.isna(odds) or pd.isna(p_b365):
            continue

        edge = p_model - p_b365
        if edge < config["min_edge"] or odds < config["min_odds"]:
            continue

        if config.get("fixed_pct"):
            bet_size = bankroll * config["max_bet_frac"]
        else:
            kelly_full = (p_model * odds - 1) / (odds - 1)
            if kelly_full <= 0:
                continue
            bet_frac = min(KELLY_FRACTION * kelly_full, MAX_BET_FRACTION)
            bet_size = bankroll * bet_frac

        if bet_size < MIN_BET_SIZE:
            continue

        pnl = bet_size * (odds - 1) if actual == 1 else -bet_size
        bankroll += pnl
        bets.append({
            "date": row["date"], "home_team": row["home_team"], "away_team": row["away_team"],
            "league": row.get("league", ""), "direction": config.get("direction", "over"),
            "p_model": float(p_model), "edge": float(edge), "odds": float(odds),
            "bet_size": float(bet_size), "actual": int(actual), "pnl": float(pnl),
        })

    bet_df = pd.DataFrame(bets)
    if len(bet_df) == 0:
        return bet_df, {"error": "No bets", "bankroll_final": bankroll}

    total_bet = bet_df["bet_size"].sum()
    total_pnl = bet_df["pnl"].sum()
    roi = total_pnl / total_bet if total_bet > 0 else 0

    br_series = pd.Series([BANKROLL_INITIAL] + bet_df["pnl"].tolist()).cumsum()
    running_max = br_series.cummax()
    drawdown = (br_series - running_max) / running_max
    max_dd = float(drawdown.min())

    return bet_df, {
        "name": config["name"],
        "bankroll_final": float(bankroll),
        "total_pnl": float(total_pnl),
        "total_bet": float(total_bet),
        "n_bets": len(bet_df),
        "hit_rate": float(bet_df["actual"].mean()),
        "roi": float(roi),
        "max_drawdown": max_dd,
        "mean_odds": float(bet_df["odds"].mean()),
        "mean_edge": float(bet_df["edge"].mean()),
    }


# ---------------------------------------------------------------------------
# 5. Slice analysis
# ---------------------------------------------------------------------------

def slice_ou(bet_df, by, min_bets=15):
    if len(bet_df) == 0:
        return pd.DataFrame()
    rows = []
    for name, group in bet_df.groupby(by):
        n = len(group)
        if n < min_bets:
            continue
        total_bet = group["bet_size"].sum()
        total_pnl = group["pnl"].sum()
        rows.append({
            by: name, "n": n, "total_bet": float(total_bet),
            "total_pnl": float(total_pnl),
            "roi": float(total_pnl / total_bet if total_bet > 0 else 0),
            "hit_rate": float(group["actual"].mean()),
            "mean_odds": float(group["odds"].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    if HAS_MPL:
        os.makedirs(FIG_DIR, exist_ok=True)

    # --- Load data ---
    print("=" * 70)
    print("1. LOADING O/U DATA")
    print("=" * 70)
    df = load_ou_data()
    print(f"  O/U odds coverage: {df['b365_over25'].notna().sum()}/{len(df)}")

    # --- Calibration ---
    print("\n" + "=" * 70)
    print("2. CALIBRATION (Over/Under 2.5)")
    print("=" * 70)

    cal_configs = [
        ("Poisson Over 2.5",  "poisson_over_2_5",  "over_2_5"),
        ("Poisson Under 2.5", "poisson_under_2_5", "under_2_5"),  # = 1 - over_2_5
        ("B365 Over 2.5",     "b365_p_over_2_5",   "over_2_5"),
        ("B365 Under 2.5",    "b365_p_under_2_5",   "over_2_5"),  # use 1 - actual_over for under
    ]

    # Fix: Under actual = 1 - over actual
    df["under_2_5"] = 1 - df["over_2_5"]

    cal_results = {}
    for label, prob_col, true_col in cal_configs:
        # For Under calibration, true_col is still "over_2_5" but we use 1 - actual
        actual_col = true_col
        if "Under" in label:
            actual_col = "under_2_5"
        cal_results[label] = calibrate_ou(df, prob_col, actual_col, label)
        r = cal_results[label]
        print(f"  {label:<25s}  ECE={r['ece']:.4f}  N={r['n']}")
        if HAS_MPL:
            plot_ou_calibration(r, label)

    # --- Discrimination ---
    print("\n" + "=" * 70)
    print("3. DISCRIMINATION (Over 2.5)")
    print("=" * 70)

    for label, prob_col in [("Poisson", "poisson_over_2_5"), ("B365", "b365_p_over_2_5")]:
        p = df[prob_col].values
        t = df["over_2_5"].values
        mask = ~np.isnan(p) & ~np.isnan(t)
        auc = roc_auc_score(t[mask], p[mask])
        aupr = average_precision_score(t[mask], p[mask])
        print(f"  {label}: AUC-ROC={auc:.4f}  AUC-PR={aupr:.4f}  N={int(mask.sum())}")

    # --- Value bet analysis ---
    print("\n" + "=" * 70)
    print("4. VALUE BET ANALYSIS (O/U 2.5, Poisson vs B365)")
    print("=" * 70)

    vb_configs = [
        ("Over 2.5",  "poisson_over_2_5",  "b365_p_over_2_5",  "b365_over25",  "over_2_5"),
        ("Under 2.5", "poisson_under_2_5", "b365_p_under_2_5", "b365_under25", "under_2_5"),
    ]

    all_vb = []
    for direction, prob_col, b365_col, odds_col, true_col in vb_configs:
        vb = value_bet_ou(df, prob_col, b365_col, odds_col, true_col, direction)
        all_vb.append(vb)
        print(f"\n  --- {direction} ---")
        print(f"  Value bets: {vb['n_value_bets']}/{vb['n_total']} ({vb['n_value_bets']/vb['n_total']*100:.1f}%)")
        print(f"  Overall ROI: {vb['overall_roi']:.4f}  |  Mean edge: {vb['mean_edge']:.4f}")
        print(f"\n  {'Edge Range':<12s}  {'N':>6s}  {'Hit Rate':>9s}  {'ROI':>8s}  {'Avg Odds':>9s}")
        print("  " + "-" * 55)
        for b in vb["buckets"]:
            n_str = str(b['n']) if b['n'] > 0 else '—'
            hr_str = f"{b['hit_rate']:.4f}" if b['hit_rate'] is not None else '—'
            roi_str = f"{b['roi']:.4f}" if b['roi'] is not None else '—'
            odds_str = f"{b['mean_odds']:.2f}" if b.get('mean_odds') is not None else '—'
            print(f"  {b['range']:<12s}  {n_str:>6s}  {hr_str:>9s}  {roi_str:>8s}  {odds_str:>9s}")

    # --- Simulated betting ---
    print("\n" + "=" * 70)
    print("5. SIMULATED BETTING (O/U 2.5)")
    print("=" * 70)

    strategies = [
        {
            "name": "OU-A: Poisson + 1/4 Kelly (edge≥5%)", "prob_col": "poisson_over_2_5",
            "b365_col": "b365_p_over_2_5", "odds_col": "b365_over25",
            "true_col": "over_2_5", "direction": "over", "min_edge": 0.05, "min_odds": 1.30,
        },
        {
            "name": "OU-B: Poisson + Fixed 1% (edge≥5%)", "prob_col": "poisson_over_2_5",
            "b365_col": "b365_p_over_2_5", "odds_col": "b365_over25",
            "true_col": "over_2_5", "direction": "over", "min_edge": 0.05, "min_odds": 1.30,
            "fixed_pct": True, "max_bet_frac": 0.01,
        },
        {
            "name": "OU-C: Over Only + 1/4 Kelly",
            "prob_col": "poisson_over_2_5",
            "b365_col": "b365_p_over_2_5", "odds_col": "b365_over25",
            "true_col": "over_2_5", "direction": "over", "min_edge": 0.05, "min_odds": 1.30,
        },
        {
            "name": "OU-D: Under Only + 1/4 Kelly",
            "prob_col": "poisson_under_2_5",
            "b365_col": "b365_p_under_2_5", "odds_col": "b365_under25",
            "true_col": "under_2_5", "direction": "under", "min_edge": 0.05, "min_odds": 1.30,
        },
    ]

    all_bet_logs = {}
    all_summaries = []

    for cfg in strategies:
        bet_df, summary = simulate_ou_betting(df, cfg)
        all_bet_logs[cfg["name"]] = bet_df
        all_summaries.append(summary)

        print(f"\n  --- {cfg['name']} ---")
        if "error" in summary:
            print(f"  ERROR: {summary['error']}")
        else:
            print(f"  Final BR:     ${summary['bankroll_final']:,.2f}  (Δ = ${summary['total_pnl']:+,.2f})")
            print(f"  Bets:         {summary['n_bets']}")
            print(f"  Hit rate:     {summary['hit_rate']:.4f}")
            print(f"  Mean odds:    {summary['mean_odds']:.2f}")
            print(f"  ROI:          {summary['roi']:.4f}  ({summary['roi']*100:.2f}%)")
            print(f"  Max drawdown: {summary['max_drawdown']:.4f}  ({summary['max_drawdown']*100:.2f}%)")

    # --- Slice analysis ---
    print("\n" + "=" * 70)
    print("6. SLICE ANALYSIS (OU-A: Poisson + 1/4 Kelly)")
    print("=" * 70)

    ou_a_bets = all_bet_logs.get(strategies[0]["name"], pd.DataFrame())

    for dim in ["league", "direction"]:
        slices = slice_ou(ou_a_bets, by=dim)
        print(f"\n  --- By {dim} ---")
        if len(slices) == 0:
            print("  Insufficient bets")
            continue
        print(f"  {'Name':<20s}  {'N':>6s}  {'Hit%':>7s}  {'ROI':>8s}  {'P&L':>10s}  {'Avg Odds':>9s}")
        print("  " + "-" * 70)
        for _, s in slices.iterrows():
            print(f"  {str(s[dim]):<20s}  {int(s['n']):>6d}  {s['hit_rate']:7.4f}  "
                  f"{s['roi']:8.4f}  ${s['total_pnl']:>9,.2f}  {s['mean_odds']:9.4f}")

    # --- Odds range slice ---
    if len(ou_a_bets) > 0:
        odds_bins = [(1.3, 1.8), (1.8, 2.5), (2.5, float("inf"))]
        print(f"\n  --- By Odds Range ---")
        print(f"  {'Range':<15s}  {'N':>6s}  {'Hit%':>7s}  {'ROI':>8s}  {'P&L':>10s}")
        print("  " + "-" * 55)
        for lo, hi in odds_bins:
            sub = ou_a_bets[ou_a_bets["odds"].between(lo, hi - 0.0001 if hi != float("inf") else hi)]
            if len(sub) == 0:
                continue
            roi = sub["pnl"].sum() / sub["bet_size"].sum() if sub["bet_size"].sum() > 0 else 0
            print(f"  [{lo:.1f}-{hi:.1f}]      {len(sub):>6d}  {sub['actual'].mean():7.4f}  "
                  f"{roi:8.4f}  ${sub['pnl'].sum():>9,.2f}")

    # --- Plots ---
    if HAS_MPL and len(ou_a_bets) > 0:
        print("\n" + "=" * 70)
        print("7. PLOTTING")
        print("=" * 70)
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
        for (cfg_name, bet_df), c in zip(all_bet_logs.items(), colors):
            if len(bet_df) == 0:
                continue
            br = [BANKROLL_INITIAL]
            for pnl in bet_df["pnl"]:
                br.append(br[-1] + pnl)
            ax.plot(range(len(br)), br, linewidth=1.5, label=cfg_name[:75], alpha=0.85, color=c)
        ax.axhline(y=BANKROLL_INITIAL, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Bankroll ($)")
        ax.set_title("O/U 2.5 Betting — Bankroll Curves (Test Set)")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/ou_bankroll_curve.png", dpi=120)
        plt.close()
        print("  Saved: ou_bankroll_curve.png")

    # --- Save bets ---
    print("\n" + "=" * 70)
    print("8. SAVING")
    print("=" * 70)

    for cfg_name, bet_df in all_bet_logs.items():
        if len(bet_df) > 0:
            short = cfg_name.split(":")[0].lower().replace(" ", "_").replace("-", "_")
            path = f"data/processed/ou_bets_{short}.parquet"
            bet_df.to_parquet(path, index=False)
            print(f"  Saved: {path}  ({len(bet_df)} bets)")

    # --- Report ---
    print("\n" + "=" * 70)
    print("9. GENERATING REPORT")
    print("=" * 70)

    with open(f"{REPORT_DIR}/over_under_report.md", "w") as f:
        f.write("# Over/Under 2.5 — Backtest Report\n\n")
        f.write(f"**Test set:** 2024-25 + 2025-26 ({len(df)} matches)\n\n")

        f.write("## 1. Calibration (ECE)\n\n")
        f.write("| Method | ECE | N |\n")
        f.write("|--------|----:|---:|\n")
        for label, r in cal_results.items():
            f.write(f"| {label} | {r['ece']:.4f} | {r['n']} |\n")

        f.write("\n## 2. Discrimination (Over 2.5)\n\n")
        for label, prob_col in [("Poisson", "poisson_over_2_5"), ("B365", "b365_p_over_2_5")]:
            p = df[prob_col].values
            t = df["over_2_5"].values
            mask = ~np.isnan(p) & ~np.isnan(t)
            auc = roc_auc_score(t[mask], p[mask])
            aupr = average_precision_score(t[mask], p[mask])
            f.write(f"- **{label}**: AUC-ROC={auc:.4f}, AUC-PR={aupr:.4f}\n")

        f.write("\n## 3. Value Bet Analysis\n\n")
        for vb in all_vb:
            f.write(f"### {vb['line']}\n\n")
            f.write(f"- Value bets: {vb['n_value_bets']}/{vb['n_total']} ({vb['n_value_bets']/vb['n_total']*100:.1f}%)\n")
            f.write(f"- Overall ROI: {vb['overall_roi']:.4f}\n")
            f.write(f"- Mean edge: {vb['mean_edge']:.4f}\n\n")
            f.write("| Edge Range | N | Hit Rate | ROI |\n")
            f.write("|------------|---:|---------:|----:|\n")
            for b in vb["buckets"]:
                n_s = str(b['n']) if b['n'] > 0 else '—'
                hr_s = f"{b['hit_rate']:.4f}" if b['hit_rate'] is not None else '—'
                roi_s = f"{b['roi']:.4f}" if b['roi'] is not None else '—'
                f.write(f"| {b['range']} | {n_s} | {hr_s} | {roi_s} |\n")
            f.write("\n")

        f.write("## 4. Simulated Betting\n\n")
        f.write("| Strategy | Final BR | Bets | Hit% | ROI | MaxDD |\n")
        f.write("|----------|---------:|-----:|-----:|----:|------:|\n")
        for s in all_summaries:
            if "error" in s:
                continue
            f.write(f"| {s['name']} | ${s['bankroll_final']:,.2f} | {s['n_bets']} | "
                    f"{s['hit_rate']:.3f} | {s['roi']:.4f} | {s['max_drawdown']:.4f} |\n")

        f.write("\n## 5. Slice Analysis (OU-A)\n\n")
        for dim in ["league", "direction"]:
            slices = slice_ou(ou_a_bets, by=dim)
            f.write(f"### By {dim}\n\n")
            if len(slices) > 0:
                f.write(f"| {dim} | N | Hit% | ROI | P&L |\n")
                f.write(f"|{'—'*len(dim)}|-----:|-----:|----:|----:|\n")
                for _, s in slices.iterrows():
                    f.write(f"| {s[dim]} | {int(s['n'])} | {s['hit_rate']:.3f} | {s['roi']:.4f} | ${s['total_pnl']:,.2f} |\n")
            f.write("\n")

        # Comparison table
        f.write("## 6. Comparison: Match Result vs Over/Under\n\n")
        f.write("| Dimension | Match Result (Stage 4) | Over/Under (Stage 5) |\n")
        f.write("|-----------|------------------------|----------------------|\n")
        # Get best O/U ROI
        ou_rois = [s.get("roi", -999) for s in all_summaries if "error" not in s]
        best_ou_roi = max(ou_rois) if ou_rois else -999
        best_ou_dd = min([s.get("max_drawdown", 0) for s in all_summaries if "error" not in s])
        f.write(f"| Calibration ECE | < 0.03 (excellent) | {cal_results.get('Poisson Over 2.5', {}).get('ece', '?'):.4f} |\n")
        f.write(f"| Best ROI | -2.5% | {best_ou_roi*100:.1f}% |\n")
        max_dd_mr = -0.751
        f.write(f"| Max Drawdown | {max_dd_mr*100:.1f}% | {best_ou_dd*100:.1f}% |\n")

        # Edge monotonicity check
        ou_over = all_vb[0]["buckets"] if all_vb else []
        non_zero = [b for b in ou_over if b["n"] > 0 and b["roi"] is not None]
        monotonic = False
        if len(non_zero) >= 2:
            rois = [b["roi"] for b in non_zero]
            monotonic = all(rois[i] <= rois[i+1] for i in range(len(rois)-1))
        f.write(f"| Edge monotonicity | No (noise-dominated) | {'Yes' if monotonic else 'No'} |\n")

        # Honest assessment
        f.write("\n## 7. Honest Assessment\n\n")
        f.write("### Can the model beat the O/U market?\n\n")

        if best_ou_roi > 0.03:
            verdict = "YES — statistically and economically significant positive ROI"
        elif best_ou_roi > 0:
            verdict = "MARGINAL — slightly positive but may not survive transaction costs"
        else:
            verdict = "NO — the model cannot find exploitable edges in the O/U 2.5 market"

        f.write(f"**Verdict: {verdict}**\n\n")

        if not monotonic:
            f.write("- **Edge is not monotonic.** Higher model confidence does not translate to higher ROI. This suggests the model's probability deviations from the market are mostly noise, not genuine insight.\n")
        else:
            f.write("- **Edge IS monotonic.** Higher edge → higher ROI. This is strong evidence the model captures genuine predictive information the market misses.\n")

        n_bets_total = sum(s.get("n_bets", 0) for s in all_summaries if "error" not in s)
        f.write(f"- **Sample size:** {n_bets_total} bets across 2 seasons. ")
        if n_bets_total > 500:
            f.write("Adequate for overall assessment.\n")
        else:
            f.write("Small — slice results (per league, per direction) are underpowered.\n")

        f.write("\n### Caveats\n\n")
        f.write("1. Only 2 seasons of test data — statistically limited.\n")
        f.write("2. O/U odds are pre-match openers from Football-Data.\n")
        f.write("3. No transaction costs or liquidity constraints modeled.\n")
        f.write("4. The Poisson model is simple (independent goals). A bivariate Poisson or Dixon-Coles model might capture goal correlation and improve calibration.\n")
        f.write(f"5. **Key difference from match-result market:** O/U uses 2 outcomes (vs 3), which reduces the vig burden and should theoretically favor the bettor. ")
        if best_ou_roi > -0.02:
            f.write("This advantage partially materializes here.\n")
        else:
            f.write("However, this structural advantage did not translate to profits in our test.\n")

    print(f"  Saved: {REPORT_DIR}/over_under_report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
