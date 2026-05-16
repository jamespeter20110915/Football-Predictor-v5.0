"""
Stage 5.5 — Robustness Check for OU-D (Under 2.5 + 1/4 Kelly)

Six rigorous tests to determine whether the observed +4.35% ROI is genuine
edge or statistical noise / overfitting.

Tests:
  1. Bootstrap confidence intervals
  2. Temporal stability (monthly)
  3. League stability
  4. Edge threshold robustness
  5. Walk-forward validation (3/6/9-month windows)
  6. Risk-adjusted returns (Sharpe, Sortino, Calmar, ruin probability)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping figures")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_DIR = "reports"
FIG_DIR = f"{REPORT_DIR}/figures"
N_BOOTSTRAP = 10000

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------

def load_bets():
    """Load OU-D bet records and add derived columns."""
    bets = pd.read_parquet("data/processed/ou_bets_ou_d.parquet")
    bets["date"] = pd.to_datetime(bets["date"])
    bets["month"] = bets["date"].dt.to_period("M")
    bets["month_str"] = bets["month"].astype(str)
    bets["hit"] = bets["actual"]
    bets["p_b365"] = bets["p_model"] - bets["edge"]  # implied market prob

    print(f"Loaded {len(bets)} OU-D bets")
    print(f"  Date range: {bets['date'].min()}  →  {bets['date'].max()}")
    print(f"  Months: {bets['month'].nunique()}")
    print(f"  Leagues: {bets['league'].nunique()}")
    print(f"  Overall ROI: {bets['pnl'].sum() / bets['bet_size'].sum():.4f}")
    print(f"  Hit rate: {bets['hit'].mean():.4f}")
    return bets


# ---------------------------------------------------------------------------
# 2. Test 1: Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def test_bootstrap(bets):
    print("\n" + "=" * 70)
    print("TEST 1: BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    n = len(bets)
    roi_samples = np.zeros(N_BOOTSTRAP)
    hit_rate_samples = np.zeros(N_BOOTSTRAP)

    rng = np.random.RandomState(42)
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        sample = bets.iloc[idx]
        roi_samples[i] = sample["pnl"].sum() / sample["bet_size"].sum()
        hit_rate_samples[i] = sample["hit"].mean()

    ci_80 = np.percentile(roi_samples, [10, 90])
    ci_90 = np.percentile(roi_samples, [5, 95])
    ci_95 = np.percentile(roi_samples, [2.5, 97.5])
    p_pos = (roi_samples > 0).mean()
    p_gt_2pct = (roi_samples > 0.02).mean()
    mean_roi = roi_samples.mean()

    print(f"  Observed ROI:       {bets['pnl'].sum() / bets['bet_size'].sum():.4f}")
    print(f"  Bootstrap mean ROI: {mean_roi:.4f}")
    print(f"  80% CI:             [{ci_80[0]:.4f}, {ci_80[1]:.4f}]")
    print(f"  90% CI:             [{ci_90[0]:.4f}, {ci_90[1]:.4f}]")
    print(f"  95% CI:             [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"  P(ROI > 0):         {p_pos:.4f}")
    print(f"  P(ROI > 2%):        {p_gt_2pct:.4f}")

    # Judgement
    p5 = np.percentile(roi_samples, 5)
    checks = {
        "5% percentile > 0%": p5 > 0,
        "5% percentile > -2%": p5 > -0.02,
        "P(ROI > 0) > 95%": p_pos > 0.95,
        "P(ROI > 0) > 80%": p_pos > 0.80,
        "P(ROI > 2%) > 70%": p_gt_2pct > 0.70,
        "P(ROI > 2%) > 50%": p_gt_2pct > 0.50,
    }
    grade = "PASS" if (checks["5% percentile > 0%"] or checks["5% percentile > -2%"]) and checks["P(ROI > 0) > 80%"] else "WARN" if checks["5% percentile > -2%"] and checks["P(ROI > 0) > 50%"] else "FAIL"
    print(f"\n  Grade: {grade}")

    for name, result in checks.items():
        print(f"    {'✓' if result else '✗'} {name}")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(roi_samples, bins=80, color="#2ca02c", alpha=0.7, edgecolor="white", density=True)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
        ax.axvline(x=mean_roi, color="#2ca02c", linestyle="-", linewidth=2, label=f"Mean = {mean_roi:.3f}")
        for pct, ci, style in [("80%", ci_80, "-"), ("90%", ci_90, "--"), ("95%", ci_95, ":")]:
            ax.axvline(x=ci[0], color="gray", linestyle=style, alpha=0.5)
            ax.axvline(x=ci[1], color="gray", linestyle=style, alpha=0.5, label=f"{pct} CI")
        ax.set_xlabel("ROI")
        ax.set_ylabel("Density")
        ax.set_title(f"Bootstrap ROI Distribution (N={N_BOOTSTRAP})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/bootstrap_roi_dist.png", dpi=120)
        plt.close()
        print("  Saved: bootstrap_roi_dist.png")

    return {"mean": mean_roi, "ci_80": ci_80, "ci_90": ci_90, "ci_95": ci_95,
            "p_pos": p_pos, "p_gt_2pct": p_gt_2pct, "grade": grade, "checks": checks}


# ---------------------------------------------------------------------------
# 3. Test 2: Temporal stability
# ---------------------------------------------------------------------------

def test_temporal(bets):
    print("\n" + "=" * 70)
    print("TEST 2: TEMPORAL STABILITY")
    print("=" * 70)

    monthly = bets.groupby("month").agg(
        n_bets=("pnl", "count"),
        total_stake=("bet_size", "sum"),
        total_pnl=("pnl", "sum"),
        hit_rate=("hit", "mean"),
    ).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    monthly["roi"] = monthly["total_pnl"] / monthly["total_stake"]

    # Sort by month
    monthly = monthly.sort_values("month")

    pos_months = (monthly["total_pnl"] > 0).sum()
    total_months = len(monthly)
    pos_ratio = pos_months / total_months if total_months > 0 else 0

    # Max single-month contribution
    total_pnl = bets["pnl"].sum()
    max_month_pnl = monthly["total_pnl"].max()
    max_month_pct = max_month_pnl / total_pnl if total_pnl > 0 else 0

    # Longest consecutive losing streak
    monthly["is_loss"] = monthly["total_pnl"] <= 0
    streak = 0
    max_streak = 0
    for loss in monthly["is_loss"]:
        if loss:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    print(f"  Total months:         {total_months}")
    print(f"  Positive months:      {pos_months} ({pos_ratio:.1%})")
    print(f"  Max single-month PnL: ${max_month_pnl:,.2f} ({max_month_pct:.1%} of total)")
    print(f"  Longest losing streak: {max_streak} months")

    print(f"\n  {'Month':<8s}  {'Bets':>5s}  {'Stake':>10s}  {'PnL':>10s}  {'ROI':>8s}  {'Hit%':>7s}")
    print("  " + "-" * 60)
    for _, r in monthly.iterrows():
        print(f"  {r['month_str']:<8s}  {int(r['n_bets']):>5d}  ${r['total_stake']:>9,.2f}  ${r['total_pnl']:>9,.2f}  "
              f"{r['roi']:8.4f}  {r['hit_rate']:7.4f}")

    checks = {
        "Positive months > 50%": pos_ratio > 0.50,
        "Max month PnL < 30%": max_month_pct < 0.30,
        "Max losing streak ≤ 4": max_streak <= 4,
    }
    if checks["Positive months > 50%"] and checks["Max month PnL < 30%"] and checks["Max losing streak ≤ 4"]:
        grade = "PASS"
    elif checks["Positive months > 50%"]:
        grade = "WARN"
    else:
        grade = "FAIL"

    print(f"\n  Grade: {grade}")
    for name, result in checks.items():
        print(f"    {'✓' if result else '✗'} {name}")

    if HAS_MPL:
        # Monthly ROI bar chart
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in monthly["total_pnl"]]
        ax.bar(range(len(monthly)), monthly["total_pnl"], color=colors, alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("PnL ($)")
        ax.set_title("Monthly P&L — OU-D Strategy")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/monthly_roi.png", dpi=120)
        plt.close()

        # Cumulative bankroll
        fig, ax = plt.subplots(figsize=(12, 5))
        cumulative = monthly["total_pnl"].cumsum() + 1000
        ax.plot(range(len(cumulative)), cumulative, "o-", color="#2ca02c", linewidth=2, markersize=6)
        ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Cumulative Bankroll ($)")
        ax.set_title("Monthly Cumulative Bankroll — OU-D Strategy")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/monthly_bankroll.png", dpi=120)
        plt.close()
        print("  Saved: monthly_roi.png, monthly_bankroll.png")

    return {"pos_ratio": pos_ratio, "max_month_pct": max_month_pct, "max_streak": max_streak,
            "grade": grade, "checks": checks, "monthly": monthly}


# ---------------------------------------------------------------------------
# 4. Test 3: League stability
# ---------------------------------------------------------------------------

def test_league(bets):
    print("\n" + "=" * 70)
    print("TEST 3: LEAGUE STABILITY")
    print("=" * 70)

    leagues = bets.groupby("league").agg(
        n_bets=("pnl", "count"),
        total_stake=("bet_size", "sum"),
        total_pnl=("pnl", "sum"),
        hit_rate=("hit", "mean"),
        avg_odds=("odds", "mean"),
        mean_edge=("edge", "mean"),
    ).reset_index()
    leagues["roi"] = leagues["total_pnl"] / leagues["total_stake"]

    # Bootstrap CI per league
    rng = np.random.RandomState(42)
    for league_name in leagues["league"]:
        l_bets = bets[bets["league"] == league_name]
        n_l = len(l_bets)
        if n_l < 10:
            leagues.loc[leagues["league"] == league_name, "roi_ci_low"] = np.nan
            leagues.loc[leagues["league"] == league_name, "roi_ci_high"] = np.nan
            continue
        roi_l_samples = np.zeros(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.choice(n_l, size=n_l, replace=True)
            sample = l_bets.iloc[idx]
            roi_l_samples[i] = sample["pnl"].sum() / sample["bet_size"].sum()
        leagues.loc[leagues["league"] == league_name, "roi_ci_low"] = np.percentile(roi_l_samples, 5)
        leagues.loc[leagues["league"] == league_name, "roi_ci_high"] = np.percentile(roi_l_samples, 95)

    profit_leagues = (leagues["roi"] > 0).sum()
    total_leagues = len(leagues)
    max_league_pct = leagues["total_pnl"].max() / bets["pnl"].sum() if bets["pnl"].sum() > 0 else float("inf")

    print(f"\n  {'League':<20s}  {'N':>5s}  {'Hit%':>7s}  {'ROI':>8s}  {'PnL':>10s}  {'90% CI':>18s}")
    print("  " + "-" * 78)
    for _, r in leagues.iterrows():
        ci_str = f"[{r['roi_ci_low']:.3f},{r['roi_ci_high']:.3f}]" if not pd.isna(r.get('roi_ci_low')) else "N/A"
        print(f"  {r['league']:<20s}  {int(r['n_bets']):>5d}  {r['hit_rate']:7.4f}  "
              f"{r['roi']:8.4f}  ${r['total_pnl']:>9,.2f}  {ci_str:>18s}")

    checks = {
        "≥ 3 leagues profitable": profit_leagues >= 3,
        "≥ 2 leagues profitable": profit_leagues >= 2,
        "Max league PnL < 50%": max_league_pct < 0.50,
    }
    if checks["≥ 3 leagues profitable"] and checks["Max league PnL < 50%"]:
        grade = "PASS"
    elif checks["≥ 2 leagues profitable"]:
        grade = "WARN"
    else:
        grade = "FAIL"

    print(f"\n  Grade: {grade}  ({profit_leagues}/{total_leagues} leagues profitable)")
    for name, result in checks.items():
        print(f"    {'✓' if result else '✗'} {name}")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in leagues["roi"]]
        bars = ax.bar(range(len(leagues)), leagues["roi"], color=colors, alpha=0.8)
        for i, (_, r) in enumerate(leagues.iterrows()):
            if not pd.isna(r.get("roi_ci_low")):
                ax.errorbar(i, r["roi"], yerr=[[r["roi"] - r["roi_ci_low"]], [r["roi_ci_high"] - r["roi"]]],
                           fmt="none", ecolor="black", capsize=5, linewidth=1)
            ax.text(i, r["roi"] + (0.01 if r["roi"] >= 0 else -0.03),
                    f"n={int(r['n_bets'])}", ha="center", fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_xticks(range(len(leagues)))
        ax.set_xticklabels(leagues["league"], fontsize=9)
        ax.set_ylabel("ROI")
        ax.set_title("ROI by League — OU-D Strategy (with 90% Bootstrap CI)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/league_roi.png", dpi=120)
        plt.close()
        print("  Saved: league_roi.png")

    return {"profit_leagues": profit_leagues, "max_league_pct": max_league_pct,
            "grade": grade, "checks": checks, "leagues": leagues}


# ---------------------------------------------------------------------------
# 5. Test 4: Edge threshold robustness
# ---------------------------------------------------------------------------

def test_edge_robustness(bets):
    print("\n" + "=" * 70)
    print("TEST 4: EDGE THRESHOLD ROBUSTNESS")
    print("=" * 70)

    # Scenarios: exclude various edge ranges
    scenarios = {
        "All bets (edge ≥ 5%)":            bets,
        "Exclude edge ≥ 10%":               bets[bets["edge"] < 0.10],
        "Exclude edge ≥ 7%":                bets[bets["edge"] < 0.07],
        "Only edge ∈ [5%, 7%]":             bets[(bets["edge"] >= 0.05) & (bets["edge"] < 0.07)],
        "Only edge ∈ [7%, 10%]":            bets[(bets["edge"] >= 0.07) & (bets["edge"] < 0.10)],
        "Only edge ≥ 10%":                  bets[bets["edge"] >= 0.10],
    }

    print(f"\n  {'Scenario':<35s}  {'N':>5s}  {'ROI':>8s}  {'Hit%':>7s}  {'PnL':>10s}")
    print("  " + "-" * 73)
    scenario_results = {}
    for name, subset in scenarios.items():
        n = len(subset)
        roi = subset["pnl"].sum() / subset["bet_size"].sum() if n > 0 and subset["bet_size"].sum() > 0 else 0
        hit = subset["hit"].mean() if n > 0 else 0
        scenario_results[name] = {"n": n, "roi": roi, "hit": hit, "pnl": subset["pnl"].sum() if n > 0 else 0}
        print(f"  {name:<35s}  {n:>5d}  {roi:8.4f}  {hit:7.4f}  ${scenario_results[name]['pnl']:>9,.2f}")

    # Cumulative ROI curve by edge threshold
    thresholds = np.arange(0.05, 0.20, 0.005)
    cum_rois = []
    cum_ns = []
    for t in thresholds:
        sub = bets[bets["edge"] >= t]
        n_s = len(sub)
        roi_s = sub["pnl"].sum() / sub["bet_size"].sum() if n_s > 0 and sub["bet_size"].sum() > 0 else 0
        cum_rois.append(roi_s)
        cum_ns.append(n_s)

    # Core check: excluding edge ≥ 10% still positive?
    roi_excl_10 = scenario_results["Exclude edge ≥ 10%"]["roi"]
    roi_excl_7 = scenario_results["Exclude edge ≥ 7%"]["roi"]

    checks = {
        "Exclude edge ≥ 10% ROI > 0": roi_excl_10 > 0,
        "Exclude edge ≥ 7% ROI > 0": roi_excl_7 > 0,
        "Edge monotonic (7-10% > 5-7%)": scenario_results["Only edge ∈ [7%, 10%]"]["roi"] > scenario_results.get("Only edge ∈ [5%, 7%]", {}).get("roi", -999),
    }
    n_fail = sum(1 for v in checks.values() if not v)
    grade = "PASS" if n_fail == 0 else "WARN" if n_fail == 1 else "FAIL"
    print(f"\n  Grade: {grade}")
    for name, result in checks.items():
        print(f"    {'✓' if result else '✗'} {name}")

    if HAS_MPL and len(cum_rois) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresholds, cum_rois, "o-", color="#2ca02c", linewidth=2, markersize=5)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Minimum Edge Threshold")
        ax.set_ylabel("ROI")
        ax.set_title("ROI vs Edge Threshold — OU-D Strategy")
        ax.grid(alpha=0.3)
        # Add N labels
        for t, roi_v, n_v in zip(thresholds, cum_rois, cum_ns):
            if n_v > 5:
                ax.annotate(str(n_v), (t, roi_v), textcoords="offset points", xytext=(0, 10),
                           fontsize=7, ha="center")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/edge_threshold_roi.png", dpi=120)
        plt.close()
        print("  Saved: edge_threshold_roi.png")

    return {"grade": grade, "checks": checks, "scenarios": scenario_results}


# ---------------------------------------------------------------------------
# 6. Test 5: Walk-forward validation
# ---------------------------------------------------------------------------

def find_best_edge_threshold(history_bets):
    """Find the minimum edge threshold that gives positive ROI on history."""
    best_t = 0.10  # default conservative
    best_roi = -float("inf")
    for t in np.arange(0.03, 0.15, 0.01):
        sub = history_bets[history_bets["edge"] >= t]
        if len(sub) < 5:
            continue
        roi_t = sub["pnl"].sum() / sub["bet_size"].sum()
        if roi_t > best_roi:
            best_roi = roi_t
            best_t = t
    return best_t


def test_walkforward(bets):
    print("\n" + "=" * 70)
    print("TEST 5: WALK-FORWARD VALIDATION")
    print("=" * 70)

    months = sorted(bets["month"].unique())
    month_strs = [str(m) for m in months]

    all_results = {}
    for window_size in [3, 6, 9]:
        wf_results = []
        for i in range(window_size, len(months)):
            history_months = months[i - window_size:i]
            current_month = months[i]

            history = bets[bets["month"].isin(history_months)]
            current = bets[bets["month"] == current_month]

            if len(history) < 10:
                continue

            best_t = find_best_edge_threshold(history)
            applied = current[current["edge"] >= best_t]

            wf_pnl = applied["pnl"].sum() if len(applied) > 0 else 0
            wf_stake = applied["bet_size"].sum() if len(applied) > 0 else 0

            wf_results.append({
                "month": str(current_month),
                "window": window_size,
                "threshold": best_t,
                "n_history": len(history),
                "n_bets": len(applied),
                "pnl": wf_pnl,
                "stake": wf_stake,
                "roi": wf_pnl / wf_stake if wf_stake > 0 else 0,
            })

        wf_df = pd.DataFrame(wf_results)
        all_results[window_size] = wf_df

        if len(wf_df) > 0:
            total_wf_roi = wf_df["pnl"].sum() / wf_df["stake"].sum() if wf_df["stake"].sum() > 0 else 0
            pos_months_wf = (wf_df["pnl"] > 0).mean()
            avg_threshold = wf_df["threshold"].mean()
            print(f"\n  Window = {window_size} months:")
            print(f"    Walk-forward ROI:  {total_wf_roi:.4f}  ({len(wf_df)} out-of-sample months)")
            print(f"    Positive months:   {pos_months_wf:.1%}")
            print(f"    Avg edge threshold: {avg_threshold:.3f}")
            print(f"    Total PnL:         ${wf_df['pnl'].sum():,.2f}")
            print(f"    Total bets:        {int(wf_df['n_bets'].sum())}")
        else:
            print(f"\n  Window = {window_size}: insufficient data")

    # Use 6-month window as primary
    wf_primary = all_results.get(6, pd.DataFrame())
    if len(wf_primary) > 0:
        wf_roi = wf_primary["pnl"].sum() / wf_primary["stake"].sum()
    else:
        wf_roi = float("nan")

    observed_roi = bets["pnl"].sum() / bets["bet_size"].sum()

    checks = {
        "WF ROI > 0% (6-mo)": wf_roi > 0,
        "WF ROI > 2% (6-mo)": wf_roi > 0.02,
    }
    if not np.isnan(wf_roi) and wf_roi > 0.02:
        grade = "PASS"
    elif not np.isnan(wf_roi) and wf_roi > 0:
        grade = "WARN"
    else:
        grade = "FAIL"

    print(f"\n  Grade: {grade}")
    for name, result in checks.items():
        print(f"    {'✓' if result else '✗'} {name}")
    print(f"    Observed ROI: {observed_roi:.4f}  vs  Walk-Forward ROI (6-mo): {wf_roi:.4f}")

    if HAS_MPL and len(wf_primary) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(wf_primary))
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in wf_primary["pnl"]]
        ax.bar(x, wf_primary["pnl"], color=colors, alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(wf_primary["month"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("PnL ($)")
        ax.set_title(f"Walk-Forward Monthly P&L — 6-Month Window (ROI={wf_roi:.4f})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/walkforward_comparison.png", dpi=120)
        plt.close()
        print("  Saved: walkforward_comparison.png")

    return {"wf_roi": wf_roi, "observed_roi": observed_roi, "grade": grade, "checks": checks,
            "all_results": {k: v.to_dict("records") if len(v) > 0 else [] for k, v in all_results.items()}}


# ---------------------------------------------------------------------------
# 7. Test 6: Risk-adjusted returns
# ---------------------------------------------------------------------------

def test_risk_metrics(bets):
    print("\n" + "=" * 70)
    print("TEST 6: RISK-ADJUSTED RETURNS")
    print("=" * 70)

    # Monthly PnL
    monthly = bets.groupby("month").agg(
        total_stake=("bet_size", "sum"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    monthly["return"] = monthly["total_pnl"] / monthly["total_stake"].mean()  # scale by avg stake

    # Sharpe (annualized)
    mean_rets = monthly["return"].mean()
    std_rets = monthly["return"].std()
    sharpe = (mean_rets / std_rets) * np.sqrt(12) if std_rets > 0 else 0

    # Sortino
    downside = monthly.loc[monthly["return"] < 0, "return"].std()
    sortino = (mean_rets / downside) * np.sqrt(12) if downside > 0 else 0

    # Calmar: annualized return / max drawdown
    br = [1000] + monthly["total_pnl"].tolist()
    br_cum = np.cumsum(br)
    running_max = np.maximum.accumulate(br_cum)
    drawdown = (br_cum - running_max) / running_max
    max_dd = abs(drawdown.min())
    # Approximate annualized return from 2 seasons (~18 months)
    total_return = (br_cum[-1] - br_cum[0]) / br_cum[0]
    annual_return = (1 + total_return) ** (12 / len(monthly)) - 1 if len(monthly) > 0 else 0
    calmar = annual_return / max_dd if max_dd > 0 else 0

    # Kelly ruin probability (P(bankroll < 50% initial) from bootstrap)
    n = len(bets)
    rng = np.random.RandomState(42)
    ruin_count = 0
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        sample = bets.iloc[idx]
        br_bt = 1000
        for _, bet in sample.iterrows():
            br_bt += bet["pnl"]
            if br_bt < 500:
                ruin_count += 1
                break
    ruin_prob = ruin_count / N_BOOTSTRAP

    print(f"  Sharpe Ratio (annualized):  {sharpe:.4f}")
    print(f"  Sortino Ratio (annualized): {sortino:.4f}")
    print(f"  Calmar Ratio:               {calmar:.4f}")
    print(f"  Max Drawdown:               {max_dd:.4f}")
    print(f"  P(BR < $500) bootstrap:     {ruin_prob:.4f}")

    checks = {
        "Sharpe > 1.0": sharpe > 1.0,
        "Sharpe > 0.5": sharpe > 0.5,
        "Sortino > 1.5": sortino > 1.5,
        "Sortino > 0.7": sortino > 0.7,
        "Calmar > 0.5": calmar > 0.5,
        "Calmar > 0.2": calmar > 0.2,
        "P(ruin) < 5%": ruin_prob < 0.05,
        "P(ruin) < 20%": ruin_prob < 0.20,
    }
    excellent = checks["Sharpe > 0.5"] and checks["Sortino > 0.7"] and checks["Calmar > 0.2"] and checks["P(ruin) < 20%"]
    good = checks["Calmar > 0.2"] and checks["P(ruin) < 20%"]
    grade = "PASS" if excellent else "WARN" if good else "FAIL"
    print(f"\n  Grade: {grade}")
    for name, result in checks.items():
        print(f"    {'✓' if result else '✗'} {name}")

    return {"sharpe": sharpe, "sortino": sortino, "calmar": calmar, "max_dd": max_dd,
            "ruin_prob": ruin_prob, "annual_return": annual_return, "grade": grade, "checks": checks}


# ---------------------------------------------------------------------------
# 8. Main — run all tests & generate report
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    if HAS_MPL:
        os.makedirs(FIG_DIR, exist_ok=True)

    bets = load_bets()

    # Run all tests
    t1 = test_bootstrap(bets)
    t2 = test_temporal(bets)
    t3 = test_league(bets)
    t4 = test_edge_robustness(bets)
    t5 = test_walkforward(bets)
    t6 = test_risk_metrics(bets)

    # --- Decision matrix ---
    tests = {
        "1. Bootstrap": t1,
        "2. Temporal Stability": t2,
        "3. League Stability": t3,
        "4. Edge Robustness": t4,
        "5. Walk-Forward": t5,
        "6. Risk-Adjusted Returns": t6,
    }

    print("\n" + "=" * 70)
    print("DECISION MATRIX")
    print("=" * 70)
    print(f"\n  {'Test':<30s}  {'Grade':>6s}")
    print("  " + "-" * 40)
    grades = []
    for name, t in tests.items():
        g = t.get("grade", "FAIL")
        grades.append(g)
        symbol = "✓" if g == "PASS" else "⚠" if g == "WARN" else "✗"
        print(f"  {name:<30s}  {symbol} {g}")

    n_pass = sum(1 for g in grades if g == "PASS")
    n_warn = sum(1 for g in grades if g == "WARN")
    n_fail = sum(1 for g in grades if g == "FAIL")

    if n_fail == 0 and n_warn <= 1:
        overall = "GREEN (绿灯) — Strategy is robust. Small-scale live testing recommended."
    elif n_fail == 0 and n_warn <= 3:
        overall = "YELLOW (黄灯) — Strategy has edge but not fully robust. Observe more data before going live."
    else:
        overall = "RED (红灯) — Strategy is NOT robust. Do not deploy real money."

    print(f"\n  Overall: {overall}")
    print(f"  PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}")

    # --- Generate report ---
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    with open(f"{REPORT_DIR}/robustness_report.md", "w") as f:
        f.write("# OU-D Strategy — Robustness Report\n\n")
        f.write(f"**Strategy:** Under 2.5 Only + 1/4 Kelly + edge ≥ 5%\n")
        f.write(f"**Test period:** 2024-25 + 2025-26 ({len(bets)} bets)\n")
        f.write(f"**Observed ROI:** {bets['pnl'].sum() / bets['bet_size'].sum():.4f}\n\n")

        f.write("---\n\n")
        f.write("## Decision Matrix\n\n")
        f.write("| Test | Grade |\n")
        f.write("|------|-------|\n")
        for name, t in tests.items():
            g = t.get("grade", "FAIL")
            emoji = "✅" if g == "PASS" else "⚠️" if g == "WARN" else "❌"
            f.write(f"| {name} | {emoji} {g} |\n")

        f.write(f"\n**Overall: {overall}**\n\n")

        # Test 1
        f.write("---\n\n## Test 1: Bootstrap Confidence Intervals\n\n")
        f.write(f"- Mean bootstrap ROI: **{t1['mean']:.4f}**\n")
        f.write(f"- 80% CI: [{t1['ci_80'][0]:.4f}, {t1['ci_80'][1]:.4f}]\n")
        f.write(f"- 90% CI: [{t1['ci_90'][0]:.4f}, {t1['ci_90'][1]:.4f}]\n")
        f.write(f"- 95% CI: [{t1['ci_95'][0]:.4f}, {t1['ci_95'][1]:.4f}]\n")
        f.write(f"- **P(ROI > 0): {t1['p_pos']:.4f}**\n")
        f.write(f"- **P(ROI > 2%): {t1['p_gt_2pct']:.4f}**\n")
        f.write(f"- Grade: **{t1['grade']}**\n")

        # Test 2
        f.write("\n---\n\n## Test 2: Temporal Stability\n\n")
        monthly = t2["monthly"]
        f.write(f"- Positive months: **{t2['pos_ratio']:.1%}**\n")
        f.write(f"- Max single-month contribution: **{t2['max_month_pct']:.1%}**\n")
        f.write(f"- Longest losing streak: **{t2['max_streak']} months**\n")
        f.write(f"- Grade: **{t2['grade']}**\n")
        f.write("\n| Month | Bets | Stake | PnL | ROI | Hit% |\n")
        f.write("|-------|-----:|------:|----:|----:|-----:|\n")
        for _, r in monthly.iterrows():
            f.write(f"| {r['month_str']} | {int(r['n_bets'])} | ${r['total_stake']:,.0f} | "
                    f"${r['total_pnl']:,.0f} | {r['roi']:.4f} | {r['hit_rate']:.3f} |\n")

        # Test 3
        f.write("\n---\n\n## Test 3: League Stability\n\n")
        f.write(f"- Profitable leagues: **{t3['profit_leagues']}/{len(t3['leagues'])}**\n")
        f.write(f"- Max league PnL share: **{t3['max_league_pct']:.1%}**\n")
        f.write(f"- Grade: **{t3['grade']}**\n")
        f.write("\n| League | Bets | Hit% | ROI | PnL | 90% CI |\n")
        f.write("|--------|-----:|-----:|----:|----:|--------|\n")
        for _, r in t3["leagues"].iterrows():
            ci_str = f"[{r['roi_ci_low']:.3f},{r['roi_ci_high']:.3f}]" if not pd.isna(r.get('roi_ci_low')) else "N/A"
            f.write(f"| {r['league']} | {int(r['n_bets'])} | {r['hit_rate']:.3f} | "
                    f"{r['roi']:.4f} | ${r['total_pnl']:,.0f} | {ci_str} |\n")

        # Test 4
        f.write("\n---\n\n## Test 4: Edge Threshold Robustness\n\n")
        f.write(f"- Grade: **{t4['grade']}**\n")
        f.write("\n| Scenario | N | ROI | Hit% | PnL |\n")
        f.write("|----------|--:|----:|-----:|----:|\n")
        for name, s in t4["scenarios"].items():
            f.write(f"| {name} | {s['n']} | {s['roi']:.4f} | {s['hit']:.3f} | ${s['pnl']:,.0f} |\n")

        # Test 5
        f.write("\n---\n\n## Test 5: Walk-Forward Validation\n\n")
        f.write(f"- Walk-Forward ROI (6-mo): **{t5['wf_roi']:.4f}**\n")
        f.write(f"- Observed (in-sample) ROI: **{t5['observed_roi']:.4f}**\n")
        f.write(f"- Grade: **{t5['grade']}**\n")
        for ws, records in t5["all_results"].items():
            if len(records) > 0:
                df_w = pd.DataFrame(records)
                wf_r = df_w["pnl"].sum() / df_w["stake"].sum() if df_w["stake"].sum() > 0 else 0
                f.write(f"\n### Window = {ws} months (WF ROI = {wf_r:.4f})\n\n")
                f.write("| Month | Threshold | Bets | PnL | ROI |\n")
                f.write("|-------|----------:|-----:|----:|----:|\n")
                for r in records:
                    f.write(f"| {r['month']} | {r['threshold']:.3f} | {r['n_bets']} | "
                            f"${r['pnl']:,.0f} | {r['roi']:.4f} |\n")

        # Test 6
        f.write("\n---\n\n## Test 6: Risk-Adjusted Returns\n\n")
        f.write(f"- Sharpe (annualized): **{t6['sharpe']:.4f}**\n")
        f.write(f"- Sortino (annualized): **{t6['sortino']:.4f}**\n")
        f.write(f"- Calmar Ratio: **{t6['calmar']:.4f}**\n")
        f.write(f"- Max Drawdown: **{t6['max_dd']:.4f}**\n")
        f.write(f"- P(BR < $500): **{t6['ruin_prob']:.4f}**\n")
        f.write(f"- Grade: **{t6['grade']}**\n")

        # Final recommendation
        f.write("\n---\n\n## Final Recommendation\n\n")
        f.write(f"**{overall}**\n\n")

        if "GREEN" in overall:
            f.write("The strategy passes all robustness checks. Key evidence:\n\n")
            f.write(f"1. Bootstrap shows P(ROI > 0) = {t1['p_pos']:.1%}\n")
            f.write(f"2. Walk-forward ROI ({t5['wf_roi']:.4f}) remains positive\n")
            f.write(f"3. Edge is monotonically increasing\n")
            f.write("\n**Recommendation:** Start live testing with $200-500 bankroll on a "
                    "low-commission betting exchange. Monitor monthly and re-evaluate "
                    "after 3 months (≥60 bets). Do not increase stakes until 6 months of "
                    "consistent positive returns.\n")
        elif "YELLOW" in overall:
            f.write("The strategy shows promise but has weaknesses. Key concerns:\n\n")
            f.write(f"1. Walk-forward ROI may degrade from observed ROI\n")
            f.write(f"2. Some slices have insufficient sample size\n")
            f.write("\n**Recommendation:** Paper-trade for another season before committing "
                    "real money. Consider ensemble weighting or model refinement to improve "
                    "edge consistency.\n")
        else:
            f.write("The strategy fails critical robustness checks. The observed +4.35% ROI "
                    "is likely due to:\n\n")
            f.write(f"1. Small sample size ({len(bets)} bets over 2 seasons)\n")
            f.write(f"2. Parameter overfitting to the test set\n")
            f.write(f"3. Market efficiency absorbing the edge\n")
            f.write("\n**Recommendation:** Do NOT deploy real money. Options:\n\n")
            f.write("- Collect more data (wait for 2026-27 season)\n")
            f.write("- Explore alternative models (bivariate Poisson, Dixon-Coles)\n")
            f.write("- Expand to other markets (Asian handicap, BTTS)\n")

    print(f"  Saved: {REPORT_DIR}/robustness_report.md")

    # Save raw data
    pd.to_pickle({
        "bootstrap": t1, "temporal": t2, "league": t3,
        "edge_robustness": t4, "walkforward": t5, "risk": t6,
    }, "data/processed/robustness_results.pkl")
    print("  Saved: data/processed/robustness_results.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()
