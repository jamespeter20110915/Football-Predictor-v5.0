"""Download and load football match data from https://www.football-data.co.uk."""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "football_data"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

SEASONS = [
    "1415", "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]

BASE_URL = "https://www.football-data.co.uk/mmz4281"

CORE_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
    "AvgH", "AvgD", "AvgA",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
})


def _download_one(league_code: str, season: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """Download a single CSV, save to disk. Returns the file path or None on failure."""
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    filepath = RAW_DIR / f"{league_code}_{season}.csv"

    if filepath.exists():
        return str(filepath)

    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            return str(filepath)
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                tqdm.write(f"  [FAIL] {url} — {e}")
                return None


def download_all() -> None:
    """Download all league/season CSVs with a progress bar and retries."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(league, season) for league in LEAGUES for season in SEASONS]
    success = 0
    failed = 0

    for league, season in tqdm(tasks, desc="Downloading Football-Data CSVs", unit="file"):
        result = _download_one(league, season)
        if result:
            success += 1
        else:
            failed += 1

    tqdm.write(f"Football-Data: {success} downloaded, {failed} failed, {success + failed} total")


def load_all() -> pd.DataFrame:
    """Read all downloaded CSVs, merge into one DataFrame with league and season columns."""
    frames: list[pd.DataFrame] = []
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    for fp in tqdm(csv_files, desc="Loading Football-Data CSVs", unit="file"):
        filename = fp.stem  # e.g. "E0_1415"
        try:
            league_code, season = filename.split("_", 1)
        except ValueError:
            tqdm.write(f"  [SKIP] Cannot parse filename: {fp.name}")
            continue

        league_name = LEAGUES.get(league_code, league_code)

        try:
            df = pd.read_csv(fp, encoding="utf-8-sig", dtype=str)
        except Exception:
            df = pd.read_csv(fp, encoding="latin-1", dtype=str)

        # Normalise column names: strip BOM leftovers, whitespace
        df.columns = [c.strip().lstrip("﻿").strip() for c in df.columns]

        # Keep only columns that exist in this file
        available = [c for c in CORE_COLUMNS if c in df.columns]
        df = df[available].copy()

        # Add missing columns as NaN
        for c in CORE_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA

        df["league"] = league_name
        df["season"] = f"20{season[:2]}-{season[2:]}"  # "1415" → "2014-15"

        frames.append(df)

    if not frames:
        tqdm.write("  [WARN] No CSV files found to load.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Parse dates — handle dd/mm/yy and dd/mm/yyyy
    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "FTHG", "FTAG", "HTHG", "HTAG",
        "HS", "AS", "HST", "AST",
        "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A",
        "AvgH", "AvgD", "AvgA",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined
