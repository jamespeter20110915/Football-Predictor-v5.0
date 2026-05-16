# Feature Documentation — Football Predictor v5

## Overview

- **Rows:** 21,511 matches
- **Columns:** 64 (5 ID + 42 features + 13 rolling meta + 4 targets)
- **Temporal ordering:** strict chronological, sorted by `(date, league, home_team)`
- **Leakage prevention:** `groupby(team, league).shift(1)` before every rolling aggregation
- **Cross-season:** rolling windows span seasons without reset (small windows flush naturally)
- **Newly promoted teams:** all rolling features NaN until enough history accumulates

---

## ID Columns (5)

| Column | Description |
|---|---|
| `date` | Match date (datetime) |
| `league` | League name |
| `season` | Season label, e.g. "2014-15" |
| `home_team` | Home team (standardised name) |
| `away_team` | Away team (standardised name) |

---

## Rolling Performance Features (32)

### Home team, 5-match window — prefix `h_roll5_`

| Column | Meaning | Calculation | Range |
|---|---|---|---|
| `h_roll5_goals_for` | Goals scored per game | Rolling avg of team's last ≤5 goals_for | [0, ~6] |
| `h_roll5_goals_against` | Goals conceded per game | Rolling avg of team's last ≤5 goals_against | [0, ~6] |
| `h_roll5_xg_for` | Expected goals for per game | Rolling avg of team's last ≤5 xg_for | [0, ~4] |
| `h_roll5_xg_against` | Expected goals against per game (xGA) | Rolling avg of team's last ≤5 xg_against | [0, ~4] |
| `h_roll5_shots_for` | Shots taken per game | Rolling avg of team's last ≤5 shots_for | [0, ~40] |
| `h_roll5_sot_for` | Shots on target per game | Rolling avg of team's last ≤5 sot_for | [0, ~15] |
| `h_roll5_win_rate` | Win rate | Fraction of last ≤5 games won | [0, 1] |
| `h_roll5_points_rate` | Points per game | Avg points (W=3, D=1, L=0) over last ≤5 | [0, 3] |

- Window: 5 matches, min_periods = 3 (NaN if <3 prior matches available)
- All stats are from the team's perspective (home + away games)

### Home team, 10-match window — prefix `h_roll10_`

Same 8 metrics as above, window=10, min_periods=5.

### Away team, 5-match window — prefix `a_roll5_`

Same 8 metrics, computed for the away team.

### Away team, 10-match window — prefix `a_roll10_`

Same 8 metrics, window=10, min_periods=5, for the away team.

---

## Venue-Specific Features (10)

### Home team's recent HOME form — prefix `h_venue5_`

| Column | Meaning | Calculation | Range |
|---|---|---|---|
| `h_venue5_goals_for` | Goals scored in home games | Rolling avg of team's last ≤5 HOME goals_for | [0, ~6] |
| `h_venue5_goals_against` | Goals conceded in home games | Rolling avg of team's last ≤5 HOME goals_against | [0, ~6] |
| `h_venue5_xg_for` | xG in home games | Rolling avg of team's last ≤5 HOME xg_for | [0, ~4] |
| `h_venue5_xg_against` | xGA in home games | Rolling avg of team's last ≤5 HOME xg_against | [0, ~4] |
| `h_venue5_win_rate` | Win rate at home | Fraction of last ≤5 HOME games won | [0, 1] |

- Window: 5, min_periods = 3
- Only the team's home fixtures are used in the window

### Away team's recent AWAY form — prefix `a_venue5_`

Same 5 metrics, computed from the away team's away fixtures only.

---

## Time Features (4)

| Column | Meaning | Calculation | Range |
|---|---|---|---|
| `rest_days_home` | Days since home team's last match | `date - previous_match_date` for the team | [1, ~30] |
| `rest_days_away` | Days since away team's last match | Same for away team | [1, ~30] |
| `matchday_home` | Home team's match count this season | Cumulative count within (team, league, season), 1-indexed | [1, ~38] |
| `matchday_away` | Away team's match count this season | Same for away team | [1, ~38] |

- `rest_days` is NaN for the first recorded match of each team (no prior date to diff against)

---

## Head-to-Head Features (3)

| Column | Meaning | Calculation | Range |
|---|---|---|---|
| `h2h5_win_rate` | Home team's win rate in past H2H | Fraction of last ≤5 H2H games won by the current home team | [0, 1] |
| `h2h5_avg_goal_diff` | Avg goal difference in past H2H | Mean of (home_goals − away_goals) across last ≤5 H2H games, always from current home team's perspective | [-5, +5] |
| `h2h5_avg_xg_diff` | Avg xG difference in past H2H | Mean of (home_xG − away_xG) across last ≤5 H2H games, same perspective | [-3, +3] |

- Window: 5, min_periods = 3
- Only matches between the same two teams **in the same league** are included
- Home/away orientation is normalised: if a past meeting had flipped venues, the result, goals, and xG are re-oriented to the current home team's perspective
- Cross-league meetings (e.g. different divisions) are excluded

---

## Market Features (6)

| Column | Meaning | Calculation | Range |
|---|---|---|---|
| `mkt_b365_p_home` | Bet365 implied home win probability | `(1/b365h) / sum(1/b365h + 1/b365d + 1/b365a)` | [0, 1] |
| `mkt_b365_p_draw` | Bet365 implied draw probability | Same normalisation | [0, 1] |
| `mkt_b365_p_away` | Bet365 implied away win probability | Same normalisation | [0, 1] |
| `mkt_avg_p_home` | Market average implied home win probability | Same using avgh/avgd/avga | [0, 1] |
| `mkt_avg_p_draw` | Market average implied draw probability | Same | [0, 1] |
| `mkt_avg_p_away` | Market average implied away win probability | Same | [0, 1] |

- Raw implied probabilities = 1/odds
- Overround (vig) removed by normalising to sum=1: `p_i = raw_i / Σ raw`
- mkt_avg missing rate ~42% (older seasons lack market average odds)

---

## Target Variables (4)

| Column | Meaning | Values |
|---|---|---|
| `target_result` | Match result | `H` (home win), `D` (draw), `A` (away win) |
| `target_home_goals` | Full-time home goals | Integer ≥ 0 |
| `target_away_goals` | Full-time away goals | Integer ≥ 0 |
| `target_total_goals` | Total goals in match | Integer ≥ 0 |

---

## Missing Data Summary

| Feature group | Typical missing % | Reason |
|---|---|---|
| Targets | 0% | Full coverage |
| ID columns | 0% | Full coverage |
| Market (b365) | 0.03% | Rare odds gaps |
| Market (avg) | 42.5% | Avg odds not available in older CSV files |
| H2H | 29.6–32.0% | Most team pairs have <3 prior meetings |
| Venue-specific | 2.3–2.5% | Teams lacking 3 home/away games (newly promoted, early season) |
| Rolling 5 | 1.1–1.4% | Teams with <3 prior matches |
| Rolling 10 | 1.9–2.0% | Teams with <5 prior matches |
| rest_days | 0.34–0.43% | First recorded match per team |
| matchday | 0% | Always computable |

---

## Usage

```python
import pandas as pd
features = pd.read_parquet("data/processed/features.parquet")

# Features only (no targets, no IDs)
feature_cols = [c for c in features.columns
                if not c.startswith("target_")
                and c not in ("date", "league", "season", "home_team", "away_team")]
X = features[feature_cols]
y = features[["target_result", "target_home_goals", "target_away_goals", "target_total_goals"]]
```
