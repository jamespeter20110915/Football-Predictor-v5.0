"""Merge Football-Data and Understat datasets into a unified parquet file."""

import json
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scrapers.football_data_scraper import load_all as load_fd
from scrapers.understat_scraper import load_all as load_us

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = PROJECT_ROOT / "data" / "team_name_mapping.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches.parquet"


def _load_mapping() -> tuple[dict[str, str], dict[str, str]]:
    """Load both team-name mapping dictionaries."""
    with open(MAPPING_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["football_data_to_standard"], data["understat_to_standard"]


def apply_mapping(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Apply team-name standardisation to home_team / away_team columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``home_team`` and ``away_team`` columns (or ``HomeTeam`` /
        ``AwayTeam`` for the Football-Data source).
    source : str
        Either ``"football_data"`` or ``"understat"``.
    """
    fd_map, us_map = _load_mapping()
    mapping = fd_map if source == "football_data" else us_map

    # Normalise Football-Data column names first
    if "HomeTeam" in df.columns:
        df = df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})

    for col in ("home_team", "away_team"):
        if col not in df.columns:
            continue
        original = set(df[col].dropna().unique())
        df[col] = df[col].map(lambda x: mapping.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
        mapped = set(df[col].dropna().unique())

        # Warn about unmapped names
        unmapped = original - set(mapping.keys())
        if unmapped:
            tqdm.write(f"  [WARN] {len(unmapped)} unmapped {source} team name(s): {sorted(unmapped)}")

    return df


def merge_and_save() -> pd.DataFrame:
    """Run the full merge pipeline and write ``matches.parquet``."""
    # ── 1. Load both sources ──────────────────────────────────────────
    print("\nLoading Football-Data CSVs ...")
    fd = load_fd()
    print(f"  → {len(fd):,} rows")

    print("\nLoading Understat JSON ...")
    us = load_us()
    print(f"  → {len(us):,} rows")

    if fd.empty:
        warnings.warn("Football-Data DataFrame is empty — nothing to merge.")
        return pd.DataFrame()

    # ── 2. Apply team-name mapping ────────────────────────────────────
    print("\nApplying team-name mapping (Football-Data) ...")
    fd = apply_mapping(fd, source="football_data")

    print("Applying team-name mapping (Understat) ...")
    us = apply_mapping(us, source="understat") if not us.empty else us

    # ── 3. Normalise dates ────────────────────────────────────────────
    fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

    # ── 4. Join ───────────────────────────────────────────────────────
    if not us.empty:
        us["date"] = pd.to_datetime(us["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        print("\nJoining datasets (left join: Football-Data ← Understat) ...")
        merged = fd.merge(
            us[["date", "home_team", "away_team", "xg_h", "xg_a"]],
            on=["date", "home_team", "away_team"],
            how="left",
            indicator=False,
        )
    else:
        print("\nSkipping join (no Understat data). Adding empty xG columns ...")
        merged = fd.copy()
        merged["xg_h"] = pd.NA
        merged["xg_a"] = pd.NA

    total = len(merged)
    xg_coverage = merged["xg_h"].notna().mean()

    print(f"  → Merged {total:,} matches")
    print(f"  → xG coverage: {xg_coverage:.2%}")

    if 0 < xg_coverage < 0.90:
        print("\n  ⚠  xG coverage below 90% — sample of unmatched rows:")
        unmatched = merged[merged["xg_h"].isna()]
        print(unmatched[["date", "home_team", "away_team", "league"]].head(10).to_string(index=False))

    # ── 5. Build output columns ───────────────────────────────────────
    col_map = {
        "Div": "league_raw",
        "league": "league",
        "season": "season",
        "Time": "time",
        "FTHG": "fthg",
        "FTAG": "ftag",
        "FTR": "ftr",
        "HTHG": "hthg",
        "HTAG": "htag",
        "HTR": "htr",
        "HS": "hs",
        "AS": "away_s",
        "HST": "hst",
        "AST": "ast",
        "HF": "hf",
        "AF": "af",
        "HC": "hc",
        "AC": "ac",
        "HY": "hy",
        "AY": "ay",
        "HR": "hr",
        "AR": "ar",
        "B365H": "b365h",
        "B365D": "b365d",
        "B365A": "b365a",
        "AvgH": "avgh",
        "AvgD": "avgd",
        "AvgA": "avga",
    }

    out_cols = {}
    for old, new in col_map.items():
        if old in merged.columns:
            out_cols[old] = new

    merged = merged.rename(columns=out_cols)

    final_cols = [
        "date", "league", "season", "home_team", "away_team", "time",
        "fthg", "ftag", "ftr", "hthg", "htag", "htr",
        "hs", "away_s", "hst", "ast",
        "hf", "af", "hc", "ac", "hy", "ay", "hr", "ar",
        "xg_h", "xg_a",
        "b365h", "b365d", "b365a", "avgh", "avgd", "avga",
    ]
    available = [c for c in final_cols if c in merged.columns]
    merged = merged[available]

    # ── 6. Sort ───────────────────────────────────────────────────────
    merged = merged.sort_values(["date", "league", "home_team"]).reset_index(drop=True)

    # ── 7. Save ───────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(merged):,} rows to {OUTPUT_PATH}")

    # ── 8. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total matches:          {len(merged):,}")
    print(f"  xG coverage:            {xg_coverage:.2%}")

    print("\n  By league:")
    for league, cnt in merged["league"].value_counts().items():
        print(f"    {league:<30s} {cnt:>6,}")

    print("\n  By season:")
    for season, cnt in sorted(merged["season"].value_counts().items()):
        print(f"    {season:<10s} {cnt:>6,}")

    print("\n  Missing-value percentages:")
    missing = merged.isna().mean().sort_values(ascending=False)
    for col, pct in missing.items():
        if pct > 0:
            print(f"    {col:<20s} {pct:>7.2%}")

    return merged
