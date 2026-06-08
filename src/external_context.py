"""Merge optional external match context (lineups, injuries, weather) into fixtures.

Reads ``data/external/match_context.csv`` (produced by the API-Football / weather
updaters) and joins its lineup-strength, absence/injury/suspension counts, and
weather columns onto each match. When the file is absent every context column is
filled with neutral defaults via :func:`_empty_context`, so the pipeline runs fully
offline — this is the seam the "no API access" limitation lives behind.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.team_names import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_CONTEXT_FILE = PROJECT_ROOT / "data" / "external" / "match_context.csv"

EXTERNAL_CONTEXT_VALUE_COLUMNS = [
    "lineup_available",
    "home_lineup_strength",
    "away_lineup_strength",
    "team_news_available",
    "home_absence_count",
    "away_absence_count",
    "home_injury_count",
    "away_injury_count",
    "home_suspension_count",
    "away_suspension_count",
    "home_key_absence_count",
    "away_key_absence_count",
    "home_manager_change_recent",
    "away_manager_change_recent",
    "weather_available",
    "temperature_c",
    "wind_kph",
    "precipitation_mm",
]


LEAGUE_ALIASES = {
    "england": {"england", "epl", "premier league", "premier_league", "e0"},
    "spain": {"spain", "la liga", "laliga", "la_liga", "sp1"},
    "italy": {"italy", "serie a", "serie_a", "i1"},
    "germany": {"germany", "bundesliga", "d1"},
    "france": {"france", "ligue 1", "ligue_1", "f1"},
}


def _empty_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with neutral context columns added (0 for *_available flags, NaN else)."""
    out = df.copy()
    for col in EXTERNAL_CONTEXT_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col.endswith("_available") else np.nan
    return out


def _column_lookup(columns) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def _pick_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = _column_lookup(df.columns)
    for alias in aliases:
        found = lookup.get(alias.lower())
        if found is not None:
            return found
    return None


def _numeric_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    col = _pick_col(df, aliases)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _flag_col(df: pd.DataFrame, aliases: tuple[str, ...], fallback: pd.Series) -> pd.Series:
    col = _pick_col(df, aliases)
    if col is None:
        return fallback.fillna(False).astype(float)

    raw = df[col]
    if raw.dtype == object:
        mapped = raw.astype(str).str.strip().str.lower().map({
            "1": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "available": 1.0,
            "confirmed": 1.0,
            "0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "n": 0.0,
            "missing": 0.0,
            "unavailable": 0.0,
        })
        return pd.to_numeric(mapped, errors="coerce").fillna(0.0)
    return pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def _league_mask(raw: pd.DataFrame, league_name: str) -> pd.Series:
    league_col = _pick_col(raw, ("league", "competition", "division", "country"))
    if league_col is None:
        return pd.Series(True, index=raw.index)

    allowed = LEAGUE_ALIASES.get(league_name, {league_name})
    normalized = raw[league_col].astype(str).str.strip().str.lower()
    return normalized.isin(allowed)


def _parse_match_dates(values: pd.Series) -> pd.Series:
    iso = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    missing = iso.isna()
    if missing.any():
        iso.loc[missing] = pd.to_datetime(values.loc[missing], dayfirst=True, errors="coerce")
    return iso.dt.normalize()


def _normalize_match_context(raw: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """Harmonise a raw match-context export into the canonical (date, teams, context) schema."""
    date_col = _pick_col(raw, ("date", "match_date", "utc_date"))
    home_col = _pick_col(raw, ("home_team", "hometeam", "team_h", "home"))
    away_col = _pick_col(raw, ("away_team", "awayteam", "team_a", "away"))
    if date_col is None or home_col is None or away_col is None:
        raise ValueError(
            "match_context.csv must include date, home_team, and away_team columns "
            "(aliases like match_date, team_h, team_a are also accepted)."
        )

    filtered = raw[_league_mask(raw, league_name)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team", *EXTERNAL_CONTEXT_VALUE_COLUMNS])

    out = pd.DataFrame({
        "date": _parse_match_dates(filtered[date_col]),
        "home_team": filtered[home_col].map(lambda value: normalize_team_name(value, league_name)),
        "away_team": filtered[away_col].map(lambda value: normalize_team_name(value, league_name)),
    })

    out["home_lineup_strength"] = _numeric_col(filtered, ("home_lineup_strength", "home_lineup_rating", "lineup_strength_home"))
    out["away_lineup_strength"] = _numeric_col(filtered, ("away_lineup_strength", "away_lineup_rating", "lineup_strength_away"))

    out["home_injury_count"] = _numeric_col(filtered, ("home_injury_count", "home_injuries", "injuries_home"))
    out["away_injury_count"] = _numeric_col(filtered, ("away_injury_count", "away_injuries", "injuries_away"))
    out["home_suspension_count"] = _numeric_col(filtered, ("home_suspension_count", "home_suspensions", "suspensions_home"))
    out["away_suspension_count"] = _numeric_col(filtered, ("away_suspension_count", "away_suspensions", "suspensions_away"))
    out["home_absence_count"] = _numeric_col(filtered, ("home_absence_count", "home_absences", "absences_home"))
    out["away_absence_count"] = _numeric_col(filtered, ("away_absence_count", "away_absences", "absences_away"))
    out["home_key_absence_count"] = _numeric_col(filtered, ("home_key_absence_count", "home_key_absences", "key_absences_home"))
    out["away_key_absence_count"] = _numeric_col(filtered, ("away_key_absence_count", "away_key_absences", "key_absences_away"))

    for side in ("home", "away"):
        missing_absence = out[f"{side}_absence_count"].isna()
        summed = out[f"{side}_injury_count"].fillna(0.0) + out[f"{side}_suspension_count"].fillna(0.0)
        has_parts = out[f"{side}_injury_count"].notna() | out[f"{side}_suspension_count"].notna()
        out.loc[missing_absence & has_parts, f"{side}_absence_count"] = summed[missing_absence & has_parts]

        days = _numeric_col(filtered, (f"{side}_manager_change_days", f"manager_change_days_{side}"))
        recent = _numeric_col(filtered, (f"{side}_manager_change_recent", f"manager_change_recent_{side}"))
        out[f"{side}_manager_change_recent"] = recent
        out.loc[out[f"{side}_manager_change_recent"].isna() & days.notna(), f"{side}_manager_change_recent"] = (days <= 30).astype(float)

    out["temperature_c"] = _numeric_col(filtered, ("temperature_c", "temp_c", "temperature"))
    out["wind_kph"] = _numeric_col(filtered, ("wind_kph", "wind_speed_kph", "wind"))
    out["precipitation_mm"] = _numeric_col(filtered, ("precipitation_mm", "precip_mm", "rain_mm", "rainfall_mm"))

    lineup_signal = out[["home_lineup_strength", "away_lineup_strength"]].notna().any(axis=1)
    team_news_signal = out[[
        "home_absence_count",
        "away_absence_count",
        "home_injury_count",
        "away_injury_count",
        "home_suspension_count",
        "away_suspension_count",
        "home_key_absence_count",
        "away_key_absence_count",
        "home_manager_change_recent",
        "away_manager_change_recent",
    ]].notna().any(axis=1)
    weather_signal = out[["temperature_c", "wind_kph", "precipitation_mm"]].notna().any(axis=1)

    out["lineup_available"] = _flag_col(filtered, ("lineup_available", "confirmed_lineups_available"), lineup_signal)
    out["team_news_available"] = _flag_col(filtered, ("team_news_available", "injuries_available", "absences_available"), team_news_signal)
    out["weather_available"] = _flag_col(filtered, ("weather_available",), weather_signal)

    out = out.dropna(subset=["date", "home_team", "away_team"])
    out = out.drop_duplicates(["date", "home_team", "away_team"], keep="last")
    return out[["date", "home_team", "away_team", *EXTERNAL_CONTEXT_VALUE_COLUMNS]]


def add_external_match_context(
    df: pd.DataFrame,
    league_name: str,
    path: Path = MATCH_CONTEXT_FILE,
) -> pd.DataFrame:
    """Left-join lineup/injury/weather context onto fixtures (neutral defaults if file absent)."""
    out = _empty_context(df)
    if not path.exists():
        return out

    raw = pd.read_csv(path)
    context = _normalize_match_context(raw, league_name)
    if context.empty:
        return out

    merged = out.merge(
        context,
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_external"),
    )
    for col in EXTERNAL_CONTEXT_VALUE_COLUMNS:
        external_col = f"{col}_external"
        if external_col in merged.columns:
            merged[col] = merged[external_col].combine_first(merged[col])
            merged = merged.drop(columns=[external_col])

    return merged
