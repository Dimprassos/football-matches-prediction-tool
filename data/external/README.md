# External Data

Place optional external football datasets here.

## Understat xG

Save the Understat/Kaggle match or game stats CSV as:

```text
data/external/understat_matches.csv
```

Or refresh it directly from Understat:

```bash
python src/update_understat.py
```

Supported schemas:

- Match-level rows with columns like `date`, `league`, `team_h`, `team_a`, `h_xg`, `a_xg`
- Team-per-match rows with columns like `date`, `league`, `club_name`, `home_away`, `xG`

CSV and Parquet files in this directory are ignored by Git.

## Pre-Match Match Context

Save optional lineups, injuries, suspensions, manager changes, and weather as:

```text
data/external/match_context.csv
```

The pipeline joins rows by `date`, `league`, `home_team`, and `away_team`. The data must be information available before kickoff; do not add post-match stats.

Supported columns:

- Required keys: `date`, `league`, `home_team`, `away_team`
- Lineups: `lineup_available`, `home_lineup_strength`, `away_lineup_strength`
- Team news: `team_news_available`, `home_absence_count`, `away_absence_count`, `home_injury_count`, `away_injury_count`, `home_suspension_count`, `away_suspension_count`, `home_key_absence_count`, `away_key_absence_count`
- Manager changes: `home_manager_change_recent`, `away_manager_change_recent`, or `home_manager_change_days`, `away_manager_change_days`
- Weather: `weather_available`, `temperature_c`, `wind_kph`, `precipitation_mm`

Example:

```csv
date,league,home_team,away_team,lineup_available,home_lineup_strength,away_lineup_strength,team_news_available,home_injury_count,away_injury_count,home_suspension_count,away_suspension_count,home_key_absence_count,away_key_absence_count,home_manager_change_recent,away_manager_change_recent,weather_available,temperature_c,wind_kph,precipitation_mm
2024-08-16,england,Man United,Fulham,1,0.94,0.88,1,2,4,0,1,1,2,0,0,1,18.5,12.0,0.4
```

### Weather Refresh

Weather can be filled from Open-Meteo historical hourly weather. First create:

```text
data/external/team_locations.csv
```

with one row per home team:

```csv
league,team,latitude,longitude
england,Arsenal,51.5549,-0.1084
england,Man United,53.4631,-2.2913
```

You can generate the exact team skeleton from the local raw match files:

```bash
python src/update_weather_context.py --write-location-template
```

Then run:

```bash
python src/update_weather_context.py --start-season 2023 --end-season 2024
```

The script writes weather fields into `match_context.csv` and reports any missing teams in `data/external/missing_team_locations.csv`.

### API-Football Lineups and Injuries

API-Football can fill the team-news columns from the free plan, but the free quota is small. The updater caches every API response in:

```text
data/external/api_football_cache/
```

Set your key in the shell:

```bash
set API_FOOTBALL_KEY=your_key_here
```

PowerShell:

```powershell
$env:API_FOOTBALL_KEY="your_key_here"
```

Then fetch the next local fixture window:

```bash
python scripts/update_team_news.py
```

This simple command fetches injuries/suspensions for the next 7 days and writes `match_context.csv`. Useful quota-safe options:

- `--lineups`: also try confirmed lineups near kickoff.
- `--max 20`: cap the number of fixture-detail calls in one run.
- `--pause 6.5`: seconds between API calls; the simple script defaults to this to avoid free-plan rate limits.
- `--cache-only`: rebuild `match_context.csv` from cached API responses without spending calls.
- `--dry-run`: print the rows without writing `match_context.csv`.

For slow historical backfill, use a narrow date range and include played matches:

```bash
python scripts/update_team_news.py --from-date 2024-08-01 --to-date 2024-08-31 --played --max 40
```

For the API-Football free plan, the current 2025-2026 season may be blocked. Use the built-in free historical training mode instead:

```bash
python scripts/update_team_news.py --free-training
```

This targets API seasons 2022-2024 and skips rows that are already present in `match_context.csv`, so you can rerun it later without repeating the same fixture-detail calls.

For a specific leakage-safe backtest, prefer the targeted mode. Example: backtest `--season 2023` needs validation 2022-2023 and test 2023-2024 context:

```bash
python scripts/update_team_news.py --backtest-season 2023 --max 10
```

This targets the test period by default. To fill the validation period too:

```bash
python scripts/update_team_news.py --backtest-season 2023 --period validation --max 10
```

If API-Football returns HTTP 429, wait and continue with a smaller batch:

```bash
python scripts/update_team_news.py --free-training --max 10
```

The importer writes `lineup_available`, lineup completeness, injury counts, suspension counts, absence counts, and trace columns such as `api_football_fixture_id`. It does not invent player quality ratings; `home_lineup_strength` and `away_lineup_strength` are lineup completeness scores unless a richer licensed source is added later.

## Player-Aware Context Without EAFC/FIFA

The project does not use EAFC/FIFA/SoFIFA ratings as the player-strength source.
Player strength should come from real match participation and optional manual
lineup/absence data. See:

```text
docs/PLAYER_CONTEXT_SCHEMA.md
```

Planned offline CSVs:

```text
data/external/player_registry.csv
data/external/player_match_stats.csv
data/external/match_lineups.csv
data/external/match_absences.csv
```

The intended flow is:

```text
player_registry.csv
player_match_stats.csv
match_lineups.csv
match_absences.csv
  -> rolling player strength, using only matches before the target fixture
  -> enriched match_context.csv
  -> optional runtime/manual player context in the Streamlit prediction UI
  -> player-aware features for a separate experiment variant
```

Leakage rule: confirmed lineups and absences can only be used for historical
training if their availability before kickoff is known or explicitly assumed and
documented. Otherwise they are allowed only as current manual prediction inputs.

`player_match_stats.csv` can optionally include `team_goals`, `opponent_goals`,
or `team_goal_diff`. These fields let the rolling player-strength helper estimate
a simple on/off team goal-difference component. If they are absent, that component
falls back to neutral instead of inventing impact.

### Generating `player_match_stats.csv` from Understat

`player_match_stats.csv` (and a derived `player_registry.csv`) can be produced for
the five supported leagues directly from Understat, which is already the project's xG
source:

```bash
python src/update_understat.py --players-only --start-season 2014 --end-season 2024
```

- `--players` builds the player datasets *in addition* to the team-level xG export;
  `--players-only` skips the team-level export.
- One request is made per match, so each match's JSON is cached under
  `data/external/understat_match_cache/` (git-ignored). Reruns are resumable and only
  fetch missing matches; add `--no-cache` to force a refetch.
- `--max-matches N` caps the run for a quick test; `--no-registry` skips the registry.
- Mapping: `started` from the Understat line-up (a `Sub` position is an off-the-bench
  appearance), `minutes` from `time`, and `goals/assists/shots/key_passes/xg/xa`
  directly. Team names are translated from Understat's long names ("Manchester City")
  to the pipeline's names ("Man City") so the rolling-strength join lines up.

Leakage note: Understat rosters are recorded *after* kickoff (who actually played),
not pre-kickoff confirmed line-ups. Using them as historical participation for rolling
player strength is reasonable (starters ≈ the confirmed XI), but treat the implied
availability as an assumption — Understat does not provide injury/suspension reasons or
pre-kickoff timestamps. `importance_tier` in the generated registry is left as
`unknown` (a manual field), never inferred from minutes.

The implemented conversion path is:

```python
from src.player_context import (
    build_player_match_context,
    load_player_context_tables,
    merge_player_match_context,
)

tables = load_player_context_tables("data/external")
player_context = build_player_match_context(
    tables.registry,
    tables.match_stats,
    tables.lineups,
    tables.absences,
)
```

`merge_player_match_context(existing, player_context)` can combine these generated
player fields with an existing `match_context.csv` while preserving unrelated
weather/API columns.

At runtime, `app.py` also reads `player_registry.csv` and `player_match_stats.csv`
for the optional Player context expander. If the registry has active rows for the
selected teams, the UI shows player choices; if not, it accepts manual `player_id`
values and falls back to neutral strength when no stats/tier exist.
