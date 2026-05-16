"""Scrape expected-goals (xG) data from https://understat.com."""

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "understat"

# Understat uses slightly different league slugs in the API URLs
LEAGUES: dict[str, str] = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}

# Understat API uses spaces, not underscores, for multi-word league names
LEAGUE_API_NAME: dict[str, str] = {
    "EPL": "EPL",
    "La_liga": "La liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
}

YEARS = list(range(2014, 2026))  # 2014 → 2014-15 season, ..., 2025 → 2025-26

API_URL = "https://understat.com/getLeagueData"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
})


def _scrape_one(league_slug: str, year: int, retries: int = 3, delay: float = 2.0) -> list[dict]:
    """Fetch match data from the Understat API. Returns list of match dicts or empty list on failure."""
    api_league = LEAGUE_API_NAME[league_slug]
    url = f"{API_URL}/{api_league}/{year}"
    filepath = RAW_DIR / f"{league_slug}_{year}.json"

    if filepath.exists():
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            filepath.unlink(missing_ok=True)

    headers = {"Referer": f"https://understat.com/league/{api_league}/{year}"}

    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=30, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            matches = payload.get("dates", [])
            if matches:
                filepath.write_text(json.dumps(matches, ensure_ascii=False), encoding="utf-8")
                return matches
            else:
                tqdm.write(f"  [WARN] No dates data at {url}")
                return []
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                tqdm.write(f"  [FAIL] {url} — {e}")
                return []


def download_all() -> None:
    """Download all league/year data from Understat API with rate limiting."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(league, year) for league in LEAGUES for year in YEARS]
    success = 0
    empty = 0
    failed = 0

    for league, year in tqdm(tasks, desc="Scraping Understat data", unit="page"):
        result = _scrape_one(league, year)
        if result:
            success += 1
        elif result is not None:
            empty += 1
        else:
            failed += 1

        # Rate limiting
        time.sleep(1.5)

    tqdm.write(
        f"Understat: {success} downloaded, {empty} empty, {failed} failed, "
        f"{success + empty + failed} total"
    )


def load_all() -> pd.DataFrame:
    """Read all saved Understat JSON files and merge into one DataFrame."""
    rows: list[dict] = []
    json_files = sorted(RAW_DIR.glob("*.json"))

    for fp in tqdm(json_files, desc="Loading Understat JSON", unit="file"):
        filename = fp.stem  # e.g. "EPL_2014"
        try:
            league_slug, year_str = filename.rsplit("_", 1)
            year = int(year_str)
        except ValueError:
            tqdm.write(f"  [SKIP] Cannot parse filename: {fp.name}")
            continue

        league_name = LEAGUES.get(league_slug, league_slug)
        season_label = f"{year}-{str(year + 1)[-2:]}"  # "2014-15"

        try:
            matches = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            tqdm.write(f"  [SKIP] Corrupt JSON: {fp.name}")
            continue

        for m in matches:
            rows.append({
                "date": (m.get("datetime") or "")[:10],
                "home_team": (m.get("h") or {}).get("title", ""),
                "away_team": (m.get("a") or {}).get("title", ""),
                "goals_h": (m.get("goals") or {}).get("h", pd.NA),
                "goals_a": (m.get("goals") or {}).get("a", pd.NA),
                "xg_h": (m.get("xG") or {}).get("h", pd.NA),
                "xg_a": (m.get("xG") or {}).get("a", pd.NA),
                "league": league_name,
                "season": season_label,
            })

    if not rows:
        tqdm.write("  [WARN] No Understat JSON files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["goals_h", "goals_a", "xg_h", "xg_a"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
