"""
Feature engineering for football match prediction model.

All features are computed chronologically: for match i, only data from matches
[0, i-1] is used. This is enforced via sort-by-date + groupby shift(1) before any
rolling aggregation.

Cross-season strategy: rolling windows (5/10 games) span across season boundaries
without reset. The small window size means old-season data is flushed naturally
within a few games of a new season. For newly promoted teams with no top-flight
history, all rolling features will be NaN until they accumulate enough matches.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Target variables (pure row-level, no leakage possible)
# ---------------------------------------------------------------------------

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Match-level target variables. No temporal leakage — each row stands alone."""
    out = pd.DataFrame(index=df.index)
    out["target_result"] = df["ftr"]
    out["target_home_goals"] = df["fthg"]
    out["target_away_goals"] = df["ftag"]
    out["target_total_goals"] = df["fthg"] + df["ftag"]
    return out


# ---------------------------------------------------------------------------
# 2. Market features (implied probabilities from betting odds)
# ---------------------------------------------------------------------------

def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bookmaker odds into vig-adjusted implied probabilities.

    Method:
      raw_implied = 1 / odds
      overround   = sum(raw_implied) - 1          (the bookmaker's margin)
      adjusted    = raw_implied / sum(raw_implied) (normalises to sum=1)
    """
    out = pd.DataFrame(index=df.index)

    for book, (h_col, d_col, a_col) in [
        ("b365", ("b365h", "b365d", "b365a")),
        ("avg",  ("avgh",  "avgd",  "avga")),
    ]:
        raw_h = 1.0 / df[h_col]
        raw_d = 1.0 / df[d_col]
        raw_a = 1.0 / df[a_col]
        total = raw_h + raw_d + raw_a
        out[f"mkt_{book}_p_home"] = raw_h / total
        out[f"mkt_{book}_p_draw"] = raw_d / total
        out[f"mkt_{book}_p_away"] = raw_a / total

    return out


# ---------------------------------------------------------------------------
# 3. Long-format conversion (one row per team per match)
# ---------------------------------------------------------------------------

def build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot each match into two rows — one from each team's perspective.

    Each row describes the match from one team's point of view:
      - goals_for / goals_against: what the team scored / conceded
      - xg_for / xg_against:      xG for the team / xG conceded (xGA = opponent's xG)
      - result: 'H' = win, 'D' = draw, 'A' = loss (from this team's perspective)
      - points: 3 for win, 1 for draw, 0 for loss
    """
    home = pd.DataFrame({
        "match_id":   df.index,
        "date":       df["date"],
        "league":     df["league"],
        "season":     df["season"],
        "team":       df["home_team"],
        "opponent":   df["away_team"],
        "venue":      "H",
        "goals_for":  df["fthg"],
        "goals_against": df["ftag"],
        "xg_for":     df["xg_h"],
        "xg_against": df["xg_a"],
        "shots_for":  df["hs"],
        "shots_against": df["away_s"],
        "sot_for":    df["hst"],
        "sot_against": df["ast"],
        "result":     df["ftr"],                              # 'H' = home team won
        "points":     df["ftr"].map({"H": 3, "D": 1, "A": 0}),
        "is_win":     (df["ftr"] == "H").astype(int),
    })

    away = pd.DataFrame({
        "match_id":   df.index,
        "date":       df["date"],
        "league":     df["league"],
        "season":     df["season"],
        "team":       df["away_team"],
        "opponent":   df["home_team"],
        "venue":      "A",
        "goals_for":  df["ftag"],
        "goals_against": df["fthg"],
        "xg_for":     df["xg_a"],
        "xg_against": df["xg_h"],
        "shots_for":  df["away_s"],
        "shots_against": df["hs"],
        "sot_for":    df["ast"],
        "sot_against": df["hst"],
        "result":     df["ftr"].map({"A": "H", "D": "D", "H": "A"}),
        "points":     df["ftr"].map({"A": 3, "D": 1, "H": 0}),
        "is_win":     (df["ftr"] == "A").astype(int),
    })

    long = pd.concat([home, away], ignore_index=True)
    # Sort so each (team, league) block is contiguous — required for rolling(index)
    long = long.sort_values(["team", "league", "date"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# 4. Rolling performance features (5 and 10 match windows)
# ---------------------------------------------------------------------------

METRICS = ["goals_for", "goals_against", "xg_for", "xg_against",
           "shots_for", "sot_for", "is_win", "points"]


def compute_rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window aggregates for each team.

    For each metric, shift(1) within the (team, league) group to exclude the
    current match, then apply rolling(window).mean() over past matches.

    Windows: 5 (min_periods=3) and 10 (min_periods=5).

    Prerequisite: long_df must have contiguous RangeIndex with each (team, league)
    block in a single contiguous segment.  build_long_format() ensures this by
    sorting on [team, league, date] then reset_index(drop=True).
    """
    long = long_df.copy()

    for window, min_periods in [(5, 3), (10, 5)]:
        for col in METRICS:
            shifted = long.groupby(["team", "league"], sort=False)[col].shift(1)
            # Groupby on the *columns* from long to split by team/league,
            # then rolling(window) uses index values for the window.  Because
            # each (team, league) block has contiguous indices, the window
            # correctly captures consecutive matches in date order.
            rolled = (
                shifted.groupby([long["team"], long["league"]], sort=False)
                .rolling(window=window, min_periods=min_periods)
                .mean()
            )
            long[f"roll{window}_{col}"] = rolled.reset_index(level=[0, 1], drop=True)

        long = long.rename(columns={
            f"roll{window}_is_win": f"roll{window}_win_rate",
            f"roll{window}_points": f"roll{window}_points_rate",
        })

    return long


def _pivot_rolling_to_match(long_df: pd.DataFrame) -> tuple:
    """
    Extract home and away rolling features from long format.
    Returns (home_df, away_df) indexed by match_id.
    """
    roll_cols = [c for c in long_df.columns if c.startswith("roll")]

    home_feats = long_df[long_df["venue"] == "H"].set_index("match_id")[roll_cols]
    away_feats = long_df[long_df["venue"] == "A"].set_index("match_id")[roll_cols]

    home_feats = home_feats.add_prefix("h_")
    away_feats = away_feats.add_prefix("a_")

    return home_feats, away_feats


# ---------------------------------------------------------------------------
# 5. Venue-specific features (home team's last 5 home games, etc.)
# ---------------------------------------------------------------------------

def compute_venue_features(long_df: pd.DataFrame) -> tuple:
    """
    Venue-specific rolling features (window=5, min_periods=3).

    - Home team features:  stats from the team's last 5 HOME games only.
    - Away team features:  stats from the team's last 5 AWAY games only.

    We filter to one venue, sort so each (team, league) block is contiguous,
    reset_index for clean rolling, then shift+roll within each group.
    """
    venue_metrics = ["goals_for", "goals_against", "xg_for", "xg_against", "is_win"]

    results = {}

    for venue, prefix in [("H", "h_venue5"), ("A", "a_venue5")]:
        sub = long_df[long_df["venue"] == venue].copy()
        sub = sub.sort_values(["team", "league", "date"]).reset_index(drop=True)

        for col in venue_metrics:
            shifted = sub.groupby(["team", "league"], sort=False)[col].shift(1)
            rolled = (
                shifted.groupby([sub["team"], sub["league"]], sort=False)
                .rolling(window=5, min_periods=3)
                .mean()
            )
            sub[f"{prefix}_{col}"] = rolled.reset_index(level=[0, 1], drop=True)

        sub = sub.rename(columns={f"{prefix}_is_win": f"{prefix}_win_rate"})

        feat_cols = [c for c in sub.columns if c.startswith(prefix)]
        results[venue] = sub.set_index("match_id")[feat_cols]

    return results["H"], results["A"]


# ---------------------------------------------------------------------------
# 6. Time-based features (rest days, matchday)
# ---------------------------------------------------------------------------

def compute_time_features(df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """
    rest_days: days since the team's previous match (any venue). NaN for the
               first recorded match of each team.
    matchday:  cumulative count of matches played by the team within the current
               season (1-indexed: first match of season = matchday 1).
    """
    long = long_df.sort_values(["team", "league", "date"]).copy()

    long["rest_days"] = long.groupby(["team", "league"])["date"].diff().dt.days
    long["matchday"] = long.groupby(["team", "league", "season"]).cumcount() + 1

    home_time = long[long["venue"] == "H"].set_index("match_id")[["rest_days", "matchday"]]
    away_time = long[long["venue"] == "A"].set_index("match_id")[["rest_days", "matchday"]]

    out = pd.DataFrame(index=df.index)
    out["rest_days_home"] = home_time["rest_days"]
    out["rest_days_away"] = away_time["rest_days"]
    out["matchday_home"] = home_time["matchday"]
    out["matchday_away"] = away_time["matchday"]
    return out


# ---------------------------------------------------------------------------
# 7. Head-to-head features (last 5 meetings between the two teams)
# ---------------------------------------------------------------------------

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Head-to-head features from previous meetings of the same two teams in the
    same league. Only matches strictly before the current date are used.

    Features (from current home team's perspective):
      - h2h5_win_rate:     fraction of past H2H games won by the current home team
      - h2h5_avg_goal_diff: mean (home_goals - away_goals) in past H2H games
      - h2h5_avg_xg_diff:   mean (home_xG - away_xG) in past H2H games

    Minimum 3 past meetings required; otherwise NaN.
    """
    df = df.reset_index()  # make match_id a regular column
    teams_arr = np.sort(df[["home_team", "away_team"]].values, axis=1)
    df["_h2h_pair"] = pd.Series(
        teams_arr[:, 0] + "||" + teams_arr[:, 1] + "||" + df["league"].values
    )

    results = []
    for _, group in tqdm(df.groupby("_h2h_pair"), desc="H2H features"):
        group = group.sort_values("date")
        rows = group.to_dict("records")

        for i, row in enumerate(rows):
            past = rows[:i]
            if len(past) < 3:
                continue

            past = past[-5:]

            home_wins = 0
            goal_diffs = []
            xg_diffs = []

            for p in past:
                if p["home_team"] == row["home_team"]:
                    if p["ftr"] == "H":
                        home_wins += 1
                    goal_diffs.append(p["fthg"] - p["ftag"])
                    xg_diffs.append(p["xg_h"] - p["xg_a"])
                else:
                    if p["ftr"] == "A":
                        home_wins += 1
                    goal_diffs.append(p["ftag"] - p["fthg"])
                    xg_diffs.append(p["xg_a"] - p["xg_h"])

            n = len(past)
            results.append({
                "match_id": row["match_id"],
                "h2h5_win_rate": home_wins / n,
                "h2h5_avg_goal_diff": np.mean(goal_diffs),
                "h2h5_avg_xg_diff": np.mean(xg_diffs),
            })

    if not results:
        return pd.DataFrame(index=df["match_id"],
                            columns=["h2h5_win_rate", "h2h5_avg_goal_diff",
                                     "h2h5_avg_xg_diff"])

    h2h_df = pd.DataFrame(results).set_index("match_id")
    h2h_df = h2h_df.reindex(df["match_id"])
    return h2h_df


# ---------------------------------------------------------------------------
# 8. Main orchestration
# ---------------------------------------------------------------------------

def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all model features from raw match data.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Raw match data from data/processed/matches.parquet.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by match_id (chronological order), with all
        rolling, venue, time, H2H, market, and target columns.
    """
    df = matches_df.copy()

    # Drop footer / garbage rows (null team names from CSV artifacts)
    df = df.dropna(subset=["home_team", "away_team"])

    # Strict chronological ordering with deterministic tiebreak
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    df.index.name = "match_id"

    print("Computing targets ...")
    targets = compute_targets(df)

    print("Computing market features ...")
    market = compute_market_features(df)

    print("Building long format ...")
    long_df = build_long_format(df)

    print("Computing rolling features (5 & 10 match windows) ...")
    long_rolled = compute_rolling_features(long_df)
    h_rolling, a_rolling = _pivot_rolling_to_match(long_rolled)
    print(f"  Home rolling: {len(h_rolling.columns)} cols")
    print(f"  Away rolling: {len(a_rolling.columns)} cols")

    print("Computing venue-specific features ...")
    h_venue, a_venue = compute_venue_features(long_df)
    print(f"  Home venue: {len(h_venue.columns)} cols")
    print(f"  Away venue: {len(a_venue.columns)} cols")

    print("Computing time features ...")
    time_feats = compute_time_features(df, long_df)

    print("Computing H2H features ...")
    h2h = compute_h2h_features(df)
    print(f"  H2H: {len(h2h.columns)} cols")

    print("Assembling final feature matrix ...")
    id_cols = df[["date", "league", "season", "home_team", "away_team"]].copy()

    features = pd.concat([
        id_cols,
        h_rolling,
        a_rolling,
        h_venue,
        a_venue,
        time_feats,
        h2h,
        market,
        targets,
    ], axis=1)

    features.index.name = "match_id"
    return features


# ---------------------------------------------------------------------------
# 9. CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    input_path = "data/processed/matches.parquet"
    output_path = "data/processed/features.parquet"

    print(f"Loading {input_path} ...")
    matches = pd.read_parquet(input_path)
    print(f"  Shape: {matches.shape}")

    features = build_features(matches)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path, index=True)
    print(f"\nSaved features to {output_path}")
    print(f"  Shape: {features.shape}")

    # --- Validation output ---
    print("\n" + "=" * 70)
    print("VALIDATION OUTPUT")
    print("=" * 70)

    print(f"\n1. Shape: {features.shape}")

    print(f"\n2. All columns ({len(features.columns)}):")
    for i, col in enumerate(features.columns, 1):
        print(f"   {i:3d}. {col}")

    print(f"\n3. Missing ratio per feature:")
    missing = features.isnull().mean().sort_values(ascending=False)
    for col in missing.index:
        ratio = missing[col]
        bar = "█" * int(ratio * 50) if ratio > 0 else "·"
        print(f"   {ratio:.4f}  {bar}  {col}")

    print(f"\n4. First 3 rows (transposed for readability):")
    sample = features.head(3).T
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.max_rows", 200)
    print(sample.to_string())
