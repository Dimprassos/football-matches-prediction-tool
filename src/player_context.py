"""Offline player-context schema loaders, validators, and rolling strength helpers.

Missing files return empty canonical DataFrames, while present files are validated
strictly so bad player/lineup/absence context cannot silently enter training or
prediction. The rolling strength helpers use only rows before the target fixture
date; this preserves the no-leakage rule before the values are later merged into
``match_context.csv``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from src.team_names import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

PLAYER_REGISTRY_FILE = EXTERNAL_DATA_DIR / "player_registry.csv"
PLAYER_MATCH_STATS_FILE = EXTERNAL_DATA_DIR / "player_match_stats.csv"
MATCH_LINEUPS_FILE = EXTERNAL_DATA_DIR / "match_lineups.csv"
MATCH_ABSENCES_FILE = EXTERNAL_DATA_DIR / "match_absences.csv"

VALID_IMPORTANCE_TIERS = {"key", "rotation", "bench", "unknown"}
VALID_ABSENCE_TYPES = {"injury", "suspension", "rest", "other"}
VALID_ABSENCE_STATUSES = {"out", "doubtful", "available"}
IMPORTANCE_TIER_STRENGTH = {
    "key": 0.85,
    "rotation": 0.55,
    "bench": 0.25,
    "unknown": 0.0,
}

PLAYER_REGISTRY_COLUMNS = [
    "player_id",
    "player_name",
    "normalized_player_name",
    "team",
    "league",
    "season",
    "position",
    "importance_tier",
    "date_from",
    "date_to",
    "source",
]

PLAYER_MATCH_STATS_COLUMNS = [
    "date",
    "league",
    "team",
    "opponent",
    "home_away",
    "player_id",
    "started",
    "minutes",
    "goals",
    "assists",
    "shots",
    "key_passes",
    "xg",
    "xa",
    "team_goals",
    "opponent_goals",
    "team_goal_diff",
    "source",
]

MATCH_LINEUPS_COLUMNS = [
    "date",
    "kickoff_at",
    "league",
    "home_team",
    "away_team",
    "team",
    "player_id",
    "is_starter",
    "is_sub",
    "source",
    "available_at",
]

MATCH_ABSENCES_COLUMNS = [
    "date",
    "kickoff_at",
    "league",
    "home_team",
    "away_team",
    "team",
    "player_id",
    "absence_type",
    "status",
    "source",
    "available_at",
]

PLAYER_MATCH_CONTEXT_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
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
    "home_absence_strength_loss",
    "away_absence_strength_loss",
    "absence_strength_loss_diff",
    "home_player_context_available",
    "away_player_context_available",
]


LEAGUE_ALIASES = {
    "england": {"england", "epl", "premier league", "premier_league", "e0"},
    "spain": {"spain", "la liga", "laliga", "la_liga", "sp1"},
    "italy": {"italy", "serie a", "serie_a", "i1"},
    "germany": {"germany", "bundesliga", "d1"},
    "france": {"france", "ligue 1", "ligue_1", "f1"},
}
LEAGUE_LOOKUP = {
    alias: league
    for league, aliases in LEAGUE_ALIASES.items()
    for alias in aliases
}


@dataclass(frozen=True)
class PlayerContextTables:
    """All player-context CSVs after schema normalization and validation."""

    registry: pd.DataFrame
    match_stats: pd.DataFrame
    lineups: pd.DataFrame
    absences: pd.DataFrame


@dataclass(frozen=True)
class PlayerStrength:
    """Leakage-safe player-strength score and its component diagnostics."""

    player_id: str
    player_strength: float
    source: str
    context_available: bool
    history_matches: int
    minutes_share_last_10: float
    starter_rate_last_10: float
    team_goal_diff_on_minus_off: float
    on_off_component: float
    contribution_score: float


RUNTIME_PLAYER_DIAGNOSTIC_COLUMNS = [
    "side",
    "role",
    "team",
    "player_id",
    "player_name",
    "absence_type",
    "status",
    "player_strength",
    "source",
    "context_available",
    "history_matches",
]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return _empty_frame(columns)
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _ensure_columns(df: pd.DataFrame, columns: list[str], defaults: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    defaults = defaults or {}
    for col in columns:
        if col not in out.columns:
            out[col] = defaults.get(col, np.nan)
    return out[columns]


def _normalize_string_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _normalize_league(value) -> str:
    raw = str(value).strip().lower()
    return LEAGUE_LOOKUP.get(raw, raw)


def _normalize_leagues(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["league"] = out["league"].map(_normalize_league)
    return out


def _normalize_player_name(value) -> str:
    """Lowercase, whitespace-normalise, and strip accents for stable manual joins."""
    if pd.isna(value):
        return ""
    text = " ".join(str(value).strip().lower().split())
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _normalize_team_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out.apply(
                lambda row: normalize_team_name(row[col], row["league"]) if pd.notna(row[col]) else row[col],
                axis=1,
            )
    return out


def _normalize_team_columns_fast(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Vectorized equivalent of :func:`_normalize_team_columns` for large frames.

    ``normalize_team_name`` depends only on ``(name, league)``, so each unique pair is
    resolved once and mapped, instead of an O(rows) row-wise ``apply``. Assumes the
    ``league`` column is already normalized. Produces identical output to the row-wise
    version but runs in ~1s on hundreds of thousands of rows.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        leagues = out["league"]
        names = out[col]
        pairs = {(lg, nm) for lg, nm in zip(leagues, names) if pd.notna(nm)}
        mapping = {(lg, nm): normalize_team_name(nm, lg) for lg, nm in pairs}
        out[col] = [
            mapping[(lg, nm)] if pd.notna(nm) else nm
            for lg, nm in zip(leagues, names)
        ]
    return out


def _parse_dates(values: pd.Series, dataset_name: str, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], dayfirst=True, errors="coerce")
    if parsed.isna().any():
        bad = values.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"{dataset_name}.{column} contains invalid dates: {bad}")
    return parsed.dt.normalize()


def _parse_optional_dates(values: pd.Series, dataset_name: str, column: str) -> pd.Series:
    blank = values.isna() | (values.astype(str).str.strip() == "")
    parsed = pd.to_datetime(values.where(~blank), format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna() & ~blank
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], dayfirst=True, errors="coerce")
    invalid = parsed.isna() & ~blank
    if invalid.any():
        bad = values.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{dataset_name}.{column} contains invalid dates: {bad}")
    return parsed.dt.normalize()


def _parse_timestamps_utc(values: pd.Series, dataset_name: str, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        bad = values.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"{dataset_name}.{column} contains invalid timestamps: {bad}")
    return parsed


def _validate_nonempty(df: pd.DataFrame, columns: tuple[str, ...], dataset_name: str) -> None:
    for col in columns:
        empty = _normalize_string_series(df[col]) == ""
        if empty.any():
            raise ValueError(f"{dataset_name}.{col} contains empty values")


def _validate_allowed_values(df: pd.DataFrame, column: str, allowed: set[str], dataset_name: str) -> pd.Series:
    values = _normalize_string_series(df[column]).str.lower()
    invalid = ~values.isin(allowed)
    if invalid.any():
        bad = sorted(values.loc[invalid].unique().tolist())
        raise ValueError(f"{dataset_name}.{column} contains invalid values: {bad}; allowed={sorted(allowed)}")
    return values


def _validate_binary_columns(df: pd.DataFrame, columns: tuple[str, ...], dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce")
        invalid = values.isna() | ~values.isin([0, 1])
        if invalid.any():
            bad = out.loc[invalid, col].astype(str).head(5).tolist()
            raise ValueError(f"{dataset_name}.{col} must contain only 0/1 values; bad={bad}")
        out[col] = values.astype(float)
    return out


def _validate_nonnegative_numeric(df: pd.DataFrame, columns: tuple[str, ...], dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce")
        invalid = values.isna() | (values < 0)
        if invalid.any():
            bad = out.loc[invalid, col].astype(str).head(5).tolist()
            raise ValueError(f"{dataset_name}.{col} must contain non-negative numeric values; bad={bad}")
        out[col] = values.astype(float)
    return out


def _coerce_optional_numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _validate_no_duplicates(df: pd.DataFrame, subset: list[str], dataset_name: str) -> None:
    duplicated = df.duplicated(subset=subset, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, subset].head(5).to_dict("records")
        raise ValueError(f"{dataset_name} contains duplicate keys for {subset}: {sample}")


def _validate_availability_window(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["kickoff_at"] = _parse_timestamps_utc(out["kickoff_at"], dataset_name, "kickoff_at")
    out["available_at"] = _parse_timestamps_utc(out["available_at"], dataset_name, "available_at")
    invalid = out["available_at"] > out["kickoff_at"]
    if invalid.any():
        sample = out.loc[invalid, ["date", "league", "home_team", "away_team", "available_at", "kickoff_at"]].head(5).to_dict("records")
        raise ValueError(f"{dataset_name} has rows where available_at is after kickoff_at: {sample}")
    return out


def load_player_registry(path: Path = PLAYER_REGISTRY_FILE) -> pd.DataFrame:
    """Load ``player_registry.csv`` or return an empty canonical registry."""
    raw = _read_optional_csv(path, PLAYER_REGISTRY_COLUMNS)
    if raw.empty:
        return _empty_frame(PLAYER_REGISTRY_COLUMNS)

    dataset = "player_registry.csv"
    required = {"player_id", "player_name", "team", "league", "season"}
    _require_columns(raw, required, dataset)
    out = _ensure_columns(
        raw,
        PLAYER_REGISTRY_COLUMNS,
        defaults={
            "normalized_player_name": "",
            "position": "",
            "importance_tier": "unknown",
            "date_from": "",
            "date_to": "",
            "source": "unknown",
        },
    )
    out = _normalize_leagues(out)
    out = _normalize_team_columns(out, ("team",))
    out["player_id"] = _normalize_string_series(out["player_id"])
    out["player_name"] = _normalize_string_series(out["player_name"])
    out["normalized_player_name"] = out["normalized_player_name"].where(
        _normalize_string_series(out["normalized_player_name"]) != "",
        out["player_name"],
    ).map(_normalize_player_name)
    out["importance_tier"] = _validate_allowed_values(out, "importance_tier", VALID_IMPORTANCE_TIERS, dataset)
    out["date_from"] = _parse_optional_dates(out["date_from"], dataset, "date_from")
    out["date_to"] = _parse_optional_dates(out["date_to"], dataset, "date_to")
    _validate_nonempty(out, ("player_id", "player_name", "team", "league", "season"), dataset)
    _validate_no_duplicates(out, ["player_id", "league", "season", "team", "date_from"], dataset)
    return out[PLAYER_REGISTRY_COLUMNS]


def load_player_match_stats(path: Path = PLAYER_MATCH_STATS_FILE) -> pd.DataFrame:
    """Load ``player_match_stats.csv`` with minimal participation validation."""
    raw = _read_optional_csv(path, PLAYER_MATCH_STATS_COLUMNS)
    if raw.empty:
        return _empty_frame(PLAYER_MATCH_STATS_COLUMNS)

    dataset = "player_match_stats.csv"
    required = {"date", "league", "team", "opponent", "player_id", "started", "minutes"}
    _require_columns(raw, required, dataset)
    out = _ensure_columns(
        raw,
        PLAYER_MATCH_STATS_COLUMNS,
        defaults={
            "home_away": "",
            "goals": np.nan,
            "assists": np.nan,
            "shots": np.nan,
            "key_passes": np.nan,
            "xg": np.nan,
            "xa": np.nan,
            "team_goals": np.nan,
            "opponent_goals": np.nan,
            "team_goal_diff": np.nan,
            "source": "unknown",
        },
    )
    out = _normalize_leagues(out)
    out = _normalize_team_columns_fast(out, ("team", "opponent"))
    out["date"] = _parse_dates(out["date"], dataset, "date")
    out["player_id"] = _normalize_string_series(out["player_id"])
    out = _validate_binary_columns(out, ("started",), dataset)
    out = _validate_nonnegative_numeric(out, ("minutes",), dataset)
    out = _coerce_optional_numeric(out, (
        "goals",
        "assists",
        "shots",
        "key_passes",
        "xg",
        "xa",
        "team_goals",
        "opponent_goals",
        "team_goal_diff",
    ))
    missing_diff = out["team_goal_diff"].isna()
    has_goal_parts = out["team_goals"].notna() & out["opponent_goals"].notna()
    out.loc[missing_diff & has_goal_parts, "team_goal_diff"] = (
        out.loc[missing_diff & has_goal_parts, "team_goals"]
        - out.loc[missing_diff & has_goal_parts, "opponent_goals"]
    )
    _validate_nonempty(out, ("league", "team", "opponent", "player_id"), dataset)
    _validate_no_duplicates(out, ["date", "league", "team", "opponent", "player_id"], dataset)
    return out[PLAYER_MATCH_STATS_COLUMNS]


def load_match_lineups(path: Path = MATCH_LINEUPS_FILE) -> pd.DataFrame:
    """Load ``match_lineups.csv`` and enforce pre-kickoff availability timestamps."""
    raw = _read_optional_csv(path, MATCH_LINEUPS_COLUMNS)
    if raw.empty:
        return _empty_frame(MATCH_LINEUPS_COLUMNS)

    dataset = "match_lineups.csv"
    required = {"date", "kickoff_at", "league", "home_team", "away_team", "team", "player_id", "is_starter", "is_sub", "available_at"}
    _require_columns(raw, required, dataset)
    out = _ensure_columns(raw, MATCH_LINEUPS_COLUMNS, defaults={"source": "unknown"})
    out = _normalize_leagues(out)
    out = _normalize_team_columns(out, ("home_team", "away_team", "team"))
    out["date"] = _parse_dates(out["date"], dataset, "date")
    out = _validate_availability_window(out, dataset)
    out["player_id"] = _normalize_string_series(out["player_id"])
    out = _validate_binary_columns(out, ("is_starter", "is_sub"), dataset)
    invalid_role = (out["is_starter"] + out["is_sub"]) > 1
    if invalid_role.any():
        sample = out.loc[invalid_role, ["date", "league", "team", "player_id"]].head(5).to_dict("records")
        raise ValueError(f"{dataset} has players marked as both starter and sub: {sample}")
    _validate_nonempty(out, ("league", "home_team", "away_team", "team", "player_id"), dataset)
    _validate_no_duplicates(out, ["date", "league", "home_team", "away_team", "team", "player_id"], dataset)
    return out[MATCH_LINEUPS_COLUMNS]


def load_match_absences(path: Path = MATCH_ABSENCES_FILE) -> pd.DataFrame:
    """Load ``match_absences.csv`` and enforce status/type plus pre-kickoff availability."""
    raw = _read_optional_csv(path, MATCH_ABSENCES_COLUMNS)
    if raw.empty:
        return _empty_frame(MATCH_ABSENCES_COLUMNS)

    dataset = "match_absences.csv"
    required = {"date", "kickoff_at", "league", "home_team", "away_team", "team", "player_id", "absence_type", "status", "available_at"}
    _require_columns(raw, required, dataset)
    out = _ensure_columns(raw, MATCH_ABSENCES_COLUMNS, defaults={"source": "unknown"})
    out = _normalize_leagues(out)
    out = _normalize_team_columns(out, ("home_team", "away_team", "team"))
    out["date"] = _parse_dates(out["date"], dataset, "date")
    out = _validate_availability_window(out, dataset)
    out["player_id"] = _normalize_string_series(out["player_id"])
    out["absence_type"] = _validate_allowed_values(out, "absence_type", VALID_ABSENCE_TYPES, dataset)
    out["status"] = _validate_allowed_values(out, "status", VALID_ABSENCE_STATUSES, dataset)
    _validate_nonempty(out, ("league", "home_team", "away_team", "team", "player_id"), dataset)
    _validate_no_duplicates(out, ["date", "league", "home_team", "away_team", "team", "player_id"], dataset)
    return out[MATCH_ABSENCES_COLUMNS]


def load_player_context_tables(data_dir: Path = EXTERNAL_DATA_DIR) -> PlayerContextTables:
    """Load every player-context CSV from ``data_dir`` with neutral empty fallbacks."""
    data_dir = Path(data_dir)
    return PlayerContextTables(
        registry=load_player_registry(data_dir / "player_registry.csv"),
        match_stats=load_player_match_stats(data_dir / "player_match_stats.csv"),
        lineups=load_match_lineups(data_dir / "match_lineups.csv"),
        absences=load_match_absences(data_dir / "match_absences.csv"),
    )


def _as_of_timestamp(value) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid as_of_date: {value!r}")
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _match_key_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["league"].astype(str)
        + "|"
        + df["team"].astype(str)
        + "|"
        + df["opponent"].astype(str)
    )


def _registry_row_for_player(
    registry: pd.DataFrame | None,
    player_id: str,
    team: str,
    league: str,
    as_of: pd.Timestamp,
) -> pd.Series | None:
    if registry is None or registry.empty:
        return None

    reg = registry.copy()
    if "date_from" in reg.columns:
        reg["date_from"] = pd.to_datetime(reg["date_from"], errors="coerce")
    else:
        reg["date_from"] = pd.NaT
    if "date_to" in reg.columns:
        reg["date_to"] = pd.to_datetime(reg["date_to"], errors="coerce")
    else:
        reg["date_to"] = pd.NaT

    mask = (
        (_normalize_string_series(reg["player_id"]) == str(player_id).strip())
        & (reg["league"].map(_normalize_league) == _normalize_league(league))
        & (reg["team"].map(lambda value: normalize_team_name(value, _normalize_league(league))) == normalize_team_name(team, _normalize_league(league)))
        & (reg["date_from"].isna() | (reg["date_from"] <= as_of))
        & (reg["date_to"].isna() | (reg["date_to"] >= as_of))
    )
    candidates = reg.loc[mask].sort_values("date_from", na_position="first")
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def fallback_player_strength_from_registry(
    player_id: str,
    team: str,
    league: str,
    as_of_date,
    registry: pd.DataFrame | None,
) -> PlayerStrength:
    """Return manual-tier fallback strength when no historical player stats exist."""
    row = _registry_row_for_player(registry, player_id, team, league, _as_of_timestamp(as_of_date))
    if row is None:
        strength = 0.0
        source = "neutral"
        context_available = False
    else:
        tier = str(row.get("importance_tier", "unknown")).strip().lower()
        strength = IMPORTANCE_TIER_STRENGTH.get(tier, 0.0)
        source = "importance_tier" if strength > 0 else "neutral"
        context_available = strength > 0

    return PlayerStrength(
        player_id=str(player_id).strip(),
        player_strength=float(strength),
        source=source,
        context_available=bool(context_available),
        history_matches=0,
        minutes_share_last_10=0.0,
        starter_rate_last_10=0.0,
        team_goal_diff_on_minus_off=0.0,
        on_off_component=0.0,
        contribution_score=0.0,
    )


def _contribution_component(player_rows: pd.DataFrame) -> float:
    if player_rows.empty:
        return 0.0
    minutes = float(pd.to_numeric(player_rows["minutes"], errors="coerce").fillna(0.0).sum())
    if minutes <= 0:
        return 0.0

    goals = pd.to_numeric(player_rows["goals"], errors="coerce").fillna(0.0).sum()
    assists = pd.to_numeric(player_rows["assists"], errors="coerce").fillna(0.0).sum()
    shots = pd.to_numeric(player_rows["shots"], errors="coerce").fillna(0.0).sum()
    key_passes = pd.to_numeric(player_rows["key_passes"], errors="coerce").fillna(0.0).sum()
    xg = pd.to_numeric(player_rows["xg"], errors="coerce").fillna(0.0).sum()
    xa = pd.to_numeric(player_rows["xa"], errors="coerce").fillna(0.0).sum()
    contribution = goals + assists + xg + xa + 0.10 * shots + 0.10 * key_passes
    per90 = contribution / max(minutes / 90.0, 1e-9)
    return _clip01(float(per90) / 1.5)


def _on_off_components(team_matches: pd.DataFrame, player_rows: pd.DataFrame) -> tuple[float, float]:
    if "team_goal_diff" not in team_matches.columns:
        return 0.0, 0.5

    diffs = pd.to_numeric(team_matches["team_goal_diff"], errors="coerce")
    if diffs.notna().sum() == 0:
        return 0.0, 0.5

    team = team_matches.copy()
    team["_match_key"] = _match_key_frame(team)
    player_played = player_rows[pd.to_numeric(player_rows["minutes"], errors="coerce").fillna(0.0) > 0].copy()
    player_keys = set(_match_key_frame(player_played)) if not player_played.empty else set()
    team["_played"] = team["_match_key"].isin(player_keys)
    team["_diff"] = diffs

    if not team["_played"].any() or team["_played"].all():
        return 0.0, 0.5

    on = team.loc[team["_played"], "_diff"].dropna()
    off = team.loc[~team["_played"], "_diff"].dropna()
    if on.empty or off.empty:
        return 0.0, 0.5

    raw_delta = float(on.mean() - off.mean())
    normalized = _clip01((raw_delta + 2.0) / 4.0)
    return raw_delta, normalized


def _team_window(team_stats: pd.DataFrame | None, as_of: pd.Timestamp, window: int):
    """Slice a team's normalized frame to the last ``window`` matches before ``as_of``.

    Returns ``(stats, team_matches, window_keys)`` or ``None`` when the team has no
    history before the date. Computing this once per (team, date) and reusing it for
    every starter is what makes batch feature-building fast.
    """
    if team_stats is None or team_stats.empty:
        return None
    stats = team_stats[team_stats["date"].notna() & (team_stats["date"] < as_of)]
    if stats.empty:
        return None
    team_matches = (
        stats.sort_values("date")
        .drop_duplicates(["date", "league", "team", "opponent"], keep="last")
        .tail(window)
        .copy()
    )
    if team_matches.empty:
        return None
    window_keys = set(_match_key_frame(team_matches))
    return stats, team_matches, window_keys


def _player_strength_from_window_rows(
    player_id: str,
    player_window: pd.DataFrame | None,
    team_matches: pd.DataFrame,
    n_matches: int,
) -> PlayerStrength:
    """Rolling strength from a player's already-windowed rows + the team window.

    Shared math for both the string-key path (:func:`_score_player_in_window`) and the
    fast integer-match-id batch path (:meth:`PlayerStrengthIndex.fixture_strengths`).
    """
    if player_window is None or player_window.empty:
        return PlayerStrength(
            player_id=player_id,
            player_strength=0.0,
            source="stats_no_recent",
            context_available=True,
            history_matches=int(n_matches),
            minutes_share_last_10=0.0,
            starter_rate_last_10=0.0,
            team_goal_diff_on_minus_off=0.0,
            on_off_component=0.0,
            contribution_score=0.0,
        )

    minutes = pd.to_numeric(player_window["minutes"], errors="coerce").fillna(0.0).clip(lower=0.0)
    minutes_share = _clip01(float(minutes.sum()) / (90.0 * n_matches))
    starter_rate = _clip01(float(pd.to_numeric(player_window["started"], errors="coerce").fillna(0.0).sum()) / n_matches)
    raw_on_off, on_off_component = _on_off_components(team_matches, player_window)
    contribution_score = _contribution_component(player_window)

    strength = (
        0.45 * minutes_share
        + 0.25 * starter_rate
        + 0.20 * on_off_component
        + 0.10 * contribution_score
    )
    return PlayerStrength(
        player_id=player_id,
        player_strength=_clip01(float(strength)),
        source="rolling_stats",
        context_available=True,
        history_matches=int(n_matches),
        minutes_share_last_10=minutes_share,
        starter_rate_last_10=starter_rate,
        team_goal_diff_on_minus_off=float(raw_on_off),
        on_off_component=on_off_component,
        contribution_score=contribution_score,
    )


def _score_player_in_window(
    player_id: str,
    stats: pd.DataFrame,
    team_matches: pd.DataFrame,
    window_keys: set,
) -> PlayerStrength | None:
    """Rolling strength for one player against a precomputed team window.

    Returns ``None`` when the player has no rows in the team's pre-date history, so the
    caller can apply the registry/neutral fallback.
    """
    player_history = stats[stats["player_id"] == player_id]
    if player_history.empty:
        return None
    player_window = player_history[_match_key_frame(player_history).isin(window_keys)].copy()
    return _player_strength_from_window_rows(player_id, player_window, team_matches, max(int(len(team_matches)), 1))


def _strength_from_team_frame(
    player_id: str,
    team_norm: str,
    league_norm: str,
    as_of: pd.Timestamp,
    team_stats: pd.DataFrame | None,
    registry: pd.DataFrame | None,
    *,
    window: int = 10,
) -> PlayerStrength:
    """Core rolling-strength kernel on a frame already filtered to one (league, team).

    ``team_stats`` must be normalized (date/league/team/opponent/player_id) and hold all
    of the team's rows; the date cutoff (``< as_of``) is applied here. Shared by
    :func:`compute_player_strength` and :class:`PlayerStrengthIndex` so the math stays
    in one place.
    """
    win = _team_window(team_stats, as_of, window)
    if win is None:
        return fallback_player_strength_from_registry(player_id, team_norm, league_norm, as_of, registry)
    stats, team_matches, window_keys = win
    result = _score_player_in_window(player_id, stats, team_matches, window_keys)
    if result is None:
        return fallback_player_strength_from_registry(player_id, team_norm, league_norm, as_of, registry)
    return result


@dataclass
class PlayerStrengthIndex:
    """Pre-normalized, per-(league, team) view of player match stats for fast lookups.

    Build once with :func:`build_player_strength_index`, then call :meth:`strength`
    many times. Each call slices a small per-team frame instead of re-normalizing and
    scanning the whole match-stats table, which is what makes serving a lineup (or
    batch-building historical features over hundreds of thousands of matches) feasible.
    """

    teams: dict[tuple[str, str], pd.DataFrame]
    registry: pd.DataFrame | None
    window: int = 10

    def strength(self, player_id, team, league, as_of_date, *, window: int | None = None) -> PlayerStrength:
        as_of = _as_of_timestamp(as_of_date)
        league_norm = _normalize_league(league)
        team_norm = normalize_team_name(team, league_norm)
        player_id = str(player_id).strip()
        team_stats = self.teams.get((league_norm, team_norm))
        return _strength_from_team_frame(
            player_id, team_norm, league_norm, as_of, team_stats, self.registry,
            window=self.window if window is None else window,
        )

    def fixture_strengths(self, team, league, as_of_date, player_ids, *, window: int | None = None) -> list[PlayerStrength]:
        """Strengths for many players of ONE team as of ONE date, sharing the team window.

        Equivalent to calling :meth:`strength` per player, but the (team, date) rolling
        window is computed a single time — the key speedup for scoring a whole lineup or
        batch-building historical features.
        """
        as_of = _as_of_timestamp(as_of_date)
        league_norm = _normalize_league(league)
        team_norm = normalize_team_name(team, league_norm)
        w = self.window if window is None else window
        team_stats = self.teams.get((league_norm, team_norm))
        ids = [str(raw_id).strip() for raw_id in player_ids]

        def _fallback(pid: str) -> PlayerStrength:
            return fallback_player_strength_from_registry(pid, team_norm, league_norm, as_of, self.registry)

        if team_stats is None or team_stats.empty or "_match_id" not in team_stats.columns:
            # No index/match-id available -> per-player path (still correct, just slower).
            return [self.strength(pid, team, league, as_of_date, window=w) for pid in ids]

        stats = team_stats[team_stats["date"].notna() & (team_stats["date"] < as_of)]
        if stats.empty:
            return [_fallback(pid) for pid in ids]

        # team_stats is pre-sorted by date, so the last `w` distinct match ids are the window.
        window_ids = set(stats["_match_id"].drop_duplicates().tail(w).tolist())
        window_stats = stats[stats["_match_id"].isin(window_ids)]
        team_matches = window_stats.drop_duplicates("_match_id", keep="last").copy()
        n_matches = max(len(window_ids), 1)
        any_history = set(stats["player_id"].unique().tolist())
        by_player = {pid: grp for pid, grp in window_stats.groupby("player_id", sort=False)}

        results = []
        for pid in ids:
            if pid not in any_history:
                results.append(_fallback(pid))
            else:
                results.append(_player_strength_from_window_rows(pid, by_player.get(pid), team_matches, n_matches))
        return results

    def likely_xi(self, team, league, as_of_date, *, n: int = 11, lookback: int = 5) -> list[str]:
        """Best-guess starting XI: the ``n`` players with the most minutes in the team's
        last ``lookback`` matches before ``as_of_date`` (used to pre-fill the UI lineup).
        """
        as_of = _as_of_timestamp(as_of_date)
        league_norm = _normalize_league(league)
        team_norm = normalize_team_name(team, league_norm)
        team_stats = self.teams.get((league_norm, team_norm))
        if team_stats is None or team_stats.empty:
            return []
        stats = team_stats[team_stats["date"].notna() & (team_stats["date"] < as_of)]
        if stats.empty:
            return []
        if "_match_id" in stats.columns:
            recent_ids = set(stats["_match_id"].drop_duplicates().tail(lookback).tolist())
            recent = stats[stats["_match_id"].isin(recent_ids)]
        else:
            recent = stats
        minutes = recent.groupby("player_id")["minutes"].sum().sort_values(ascending=False)
        return [str(pid) for pid in minutes.head(n).index.tolist()]


def build_player_strength_index(
    match_stats: pd.DataFrame | None,
    registry: pd.DataFrame | None = None,
    *,
    window: int = 10,
    assume_normalized: bool = False,
) -> PlayerStrengthIndex:
    """Normalize ``match_stats`` once and group it by (league, team) for fast lookups.

    The expensive normalization (dates, league, team/opponent names, player ids) and
    grouping happen a single time here; :meth:`PlayerStrengthIndex.strength` is then
    cheap regardless of how large the source table is. Pass ``assume_normalized=True``
    when the frame already came through :func:`load_player_match_stats` (which performs
    the same normalization), to skip it and only group.
    """
    if match_stats is None or match_stats.empty:
        return PlayerStrengthIndex(teams={}, registry=registry, window=window)

    if assume_normalized:
        stats = match_stats
    else:
        stats = match_stats.copy()
        stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
        stats = _normalize_leagues(stats)
        stats = _normalize_team_columns_fast(stats, ("team", "opponent"))
        stats["player_id"] = _normalize_string_series(stats["player_id"])

    teams = {}
    for (league_val, team_val), group in stats.groupby(["league", "team"], sort=False):
        # Pre-sort by date and assign an integer match id once, so per-fixture lookups
        # need neither a re-sort nor string match keys (the batch hot path).
        ordered = group.sort_values("date").copy()
        ordered["_match_id"] = pd.factorize(
            ordered["date"].astype(str) + "|" + ordered["opponent"].astype(str)
        )[0]
        teams[(str(league_val), str(team_val))] = ordered
    return PlayerStrengthIndex(teams=teams, registry=registry, window=window)


def compute_player_strength(
    player_id: str,
    team: str,
    league: str,
    as_of_date,
    match_stats: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    *,
    window: int = 10,
) -> PlayerStrength:
    """Compute a player strength score using only team/player rows before ``as_of_date``.

    The score is intentionally simple and auditable: minutes share, starter rate,
    optional team on/off goal-difference signal, and optional attacking contribution.
    If the player has no historical stat rows before the date, the function falls
    back to the manual ``importance_tier`` in ``player_registry.csv``. For repeated
    lookups over a large table, build a :class:`PlayerStrengthIndex` once instead of
    calling this per player.
    """
    as_of = _as_of_timestamp(as_of_date)
    league_norm = _normalize_league(league)
    team_norm = normalize_team_name(team, league_norm)
    player_id = str(player_id).strip()

    if match_stats is None or match_stats.empty:
        return fallback_player_strength_from_registry(player_id, team_norm, league_norm, as_of, registry)

    stats = match_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
    stats = _normalize_leagues(stats)
    stats = _normalize_team_columns(stats, ("team", "opponent"))
    stats["player_id"] = _normalize_string_series(stats["player_id"])
    team_stats = stats[(stats["league"] == league_norm) & (stats["team"] == team_norm)]
    return _strength_from_team_frame(
        player_id, team_norm, league_norm, as_of, team_stats, registry, window=window,
    )


def compute_player_strengths(
    player_ids: list[str] | tuple[str, ...],
    team: str,
    league: str,
    as_of_date,
    match_stats: pd.DataFrame,
    registry: pd.DataFrame | None = None,
    *,
    window: int = 10,
) -> pd.DataFrame:
    """Compute rolling strengths for many players and return a diagnostics DataFrame."""
    rows = [
        compute_player_strength(
            player_id,
            team,
            league,
            as_of_date,
            match_stats,
            registry,
            window=window,
        ).__dict__
        for player_id in player_ids
    ]
    return pd.DataFrame(rows)


def _empty_player_match_context() -> pd.DataFrame:
    return pd.DataFrame(columns=PLAYER_MATCH_CONTEXT_COLUMNS)


def _normalise_fixture_frame(df: pd.DataFrame, *, include_team: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = _normalize_leagues(out)
    team_cols = ["home_team", "away_team"]
    if include_team:
        team_cols.append("team")
    out = _normalize_team_columns(out, tuple(team_cols))
    out["player_id"] = _normalize_string_series(out["player_id"])
    return out.dropna(subset=["date", "league", "home_team", "away_team"])


def _fixture_key(row: pd.Series) -> tuple:
    return (
        pd.Timestamp(row["date"]).normalize(),
        row["league"],
        row["home_team"],
        row["away_team"],
    )


def _status_weight(status: str) -> float:
    status = str(status).strip().lower()
    if status == "out":
        return 1.0
    if status == "doubtful":
        return 0.5
    return 0.0


def _side_context_available(strengths: list[PlayerStrength]) -> float:
    return float(any(item.context_available for item in strengths))


def _lineup_strength(
    lineup_rows: pd.DataFrame,
    team: str,
    league: str,
    as_of_date,
    index: PlayerStrengthIndex,
    *,
    window: int,
) -> tuple[float, list[PlayerStrength]]:
    if lineup_rows is None or lineup_rows.empty:
        return np.nan, []
    starters = lineup_rows[pd.to_numeric(lineup_rows.get("is_starter", 0), errors="coerce").fillna(0.0) == 1]
    if starters.empty:
        return np.nan, []

    strengths = index.fixture_strengths(team, league, as_of_date, starters["player_id"].tolist(), window=window)
    available_strengths = [item.player_strength for item in strengths if item.context_available]
    if not available_strengths:
        return np.nan, strengths
    return float(np.mean(available_strengths)), strengths


def _absence_summary(
    absence_rows: pd.DataFrame,
    team: str,
    league: str,
    as_of_date,
    index: PlayerStrengthIndex,
    *,
    window: int,
) -> dict:
    summary = {
        "absence_count": np.nan,
        "injury_count": np.nan,
        "suspension_count": np.nan,
        "key_absence_count": np.nan,
        "absence_strength_loss": np.nan,
        "strengths": [],
    }
    if absence_rows.empty:
        return summary

    counts = {
        "absence_count": 0.0,
        "injury_count": 0.0,
        "suspension_count": 0.0,
        "key_absence_count": 0.0,
        "absence_strength_loss": 0.0,
    }
    strengths = []
    for _, row in absence_rows.iterrows():
        weight = _status_weight(row.get("status", "available"))
        if weight <= 0:
            continue
        strength = index.strength(row["player_id"], team, league, as_of_date, window=window)
        strengths.append(strength)
        counts["absence_count"] += weight
        if row.get("absence_type") == "injury":
            counts["injury_count"] += weight
        elif row.get("absence_type") == "suspension":
            counts["suspension_count"] += weight
        if strength.player_strength >= 0.75:
            counts["key_absence_count"] += weight
        counts["absence_strength_loss"] += weight * strength.player_strength

    summary.update(counts)
    summary["strengths"] = strengths
    return summary


def build_player_match_context(
    registry: pd.DataFrame,
    match_stats: pd.DataFrame,
    lineups: pd.DataFrame,
    absences: pd.DataFrame,
    *,
    window: int = 10,
    index: PlayerStrengthIndex | None = None,
) -> pd.DataFrame:
    """Convert player lineups/absences into match-level context rows.

    The returned frame is compatible with ``data/external/match_context.csv``. It
    contains only pre-match player-context fields; weather and API trace columns are
    intentionally left to the existing context updaters. Player strengths are
    computed through :func:`compute_player_strength`, so only historical rows before
    each fixture date are used.
    """
    lineups_norm = _normalise_fixture_frame(lineups, include_team=True)
    absences_norm = _normalise_fixture_frame(absences, include_team=True)
    if lineups_norm.empty and absences_norm.empty:
        return _empty_player_match_context()

    if index is None:
        index = build_player_strength_index(match_stats, registry, window=window)

    keys = set()
    for frame in (lineups_norm, absences_norm):
        if not frame.empty:
            keys.update(_fixture_key(row) for _, row in frame.iterrows())

    rows = []
    for date, league, home_team, away_team in sorted(keys):
        row = {
            "date": date,
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
        }
        fixture_lineups = lineups_norm[
            (lineups_norm["date"] == date)
            & (lineups_norm["league"] == league)
            & (lineups_norm["home_team"] == home_team)
            & (lineups_norm["away_team"] == away_team)
        ] if not lineups_norm.empty else pd.DataFrame()
        fixture_absences = absences_norm[
            (absences_norm["date"] == date)
            & (absences_norm["league"] == league)
            & (absences_norm["home_team"] == home_team)
            & (absences_norm["away_team"] == away_team)
        ] if not absences_norm.empty else pd.DataFrame()

        all_strengths: dict[str, list[PlayerStrength]] = {"home": [], "away": []}
        lineup_present = False
        team_news_present = not fixture_absences.empty
        for side, team in (("home", home_team), ("away", away_team)):
            side_lineups = fixture_lineups[fixture_lineups["team"] == team] if not fixture_lineups.empty else pd.DataFrame()
            if not side_lineups.empty:
                lineup_present = True
            lineup_strength, lineup_strengths = _lineup_strength(
                side_lineups,
                team,
                league,
                date,
                index,
                window=window,
            )
            all_strengths[side].extend(lineup_strengths)
            row[f"{side}_lineup_strength"] = lineup_strength

            side_absences = fixture_absences[fixture_absences["team"] == team] if not fixture_absences.empty else pd.DataFrame()
            summary = _absence_summary(
                side_absences,
                team,
                league,
                date,
                index,
                window=window,
            )
            all_strengths[side].extend(summary["strengths"])
            row[f"{side}_absence_count"] = summary["absence_count"]
            row[f"{side}_injury_count"] = summary["injury_count"]
            row[f"{side}_suspension_count"] = summary["suspension_count"]
            row[f"{side}_key_absence_count"] = summary["key_absence_count"]
            row[f"{side}_absence_strength_loss"] = summary["absence_strength_loss"]
            row[f"{side}_player_context_available"] = _side_context_available(all_strengths[side])

        row["lineup_available"] = 1.0 if lineup_present else np.nan
        row["team_news_available"] = 1.0 if team_news_present else np.nan
        if np.isfinite(row["home_absence_strength_loss"]) or np.isfinite(row["away_absence_strength_loss"]):
            row["absence_strength_loss_diff"] = (
                (row["home_absence_strength_loss"] if np.isfinite(row["home_absence_strength_loss"]) else 0.0)
                - (row["away_absence_strength_loss"] if np.isfinite(row["away_absence_strength_loss"]) else 0.0)
            )
        else:
            row["absence_strength_loss_diff"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    return _ensure_columns(out, PLAYER_MATCH_CONTEXT_COLUMNS)


def _clean_player_ids(player_ids: Sequence[str] | None) -> list[str]:
    if player_ids is None:
        return []
    clean: list[str] = []
    seen: set[str] = set()
    for value in player_ids:
        player_id = str(value).strip()
        if player_id and player_id not in seen:
            clean.append(player_id)
            seen.add(player_id)
    return clean


def _clean_absence_items(
    absences: Sequence[str | Mapping] | None,
    *,
    default_absence_type: str,
    default_status: str,
) -> list[dict]:
    if absences is None:
        return []
    clean: list[dict] = []
    seen: set[str] = set()
    for item in absences:
        if isinstance(item, Mapping):
            player_id = str(item.get("player_id", "")).strip()
            absence_type = str(item.get("absence_type", default_absence_type)).strip().lower()
            status = str(item.get("status", default_status)).strip().lower()
        else:
            player_id = str(item).strip()
            absence_type = default_absence_type
            status = default_status

        if not player_id or player_id in seen:
            continue
        if absence_type not in VALID_ABSENCE_TYPES:
            raise ValueError(f"Invalid absence_type for {player_id!r}: {absence_type!r}")
        if status not in VALID_ABSENCE_STATUSES:
            raise ValueError(f"Invalid status for {player_id!r}: {status!r}")
        clean.append({"player_id": player_id, "absence_type": absence_type, "status": status})
        seen.add(player_id)
    return clean


def _runtime_player_name(
    registry: pd.DataFrame | None,
    player_id: str,
    team: str,
    league: str,
    as_of_date,
) -> str:
    row = _registry_row_for_player(registry, player_id, team, league, _as_of_timestamp(as_of_date))
    if row is None:
        return ""
    return str(row.get("player_name", "")).strip()


def _runtime_diagnostic_rows(
    player_ids: Sequence[str],
    *,
    side: str,
    role: str,
    team: str,
    league: str,
    as_of_date,
    index: PlayerStrengthIndex,
    window: int,
    absence_type: str = "",
    status: str = "",
) -> list[dict]:
    rows = []
    for player_id in _clean_player_ids(player_ids):
        strength = index.strength(player_id, team, league, as_of_date, window=window)
        rows.append({
            "side": side,
            "role": role,
            "team": normalize_team_name(team, _normalize_league(league)),
            "player_id": player_id,
            "player_name": _runtime_player_name(index.registry, player_id, team, league, as_of_date),
            "absence_type": absence_type,
            "status": status,
            "player_strength": strength.player_strength,
            "source": strength.source,
            "context_available": strength.context_available,
            "history_matches": strength.history_matches,
        })
    return rows


def build_runtime_player_context(
    *,
    league: str,
    home_team: str,
    away_team: str,
    match_date,
    registry: pd.DataFrame | None = None,
    match_stats: pd.DataFrame | None = None,
    home_starters: Sequence[str] | None = None,
    away_starters: Sequence[str] | None = None,
    home_absences: Sequence[str | Mapping] | None = None,
    away_absences: Sequence[str | Mapping] | None = None,
    default_absence_type: str = "injury",
    default_absence_status: str = "out",
    window: int = 10,
    index: PlayerStrengthIndex | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Build a serve-time context dict from manually supplied lineups/absences.

    This is the runtime-only counterpart to the CSV-based player context path. It
    accepts player IDs from the UI, computes strengths with the same temporal
    :func:`compute_player_strength` logic, and returns a dict that can be passed as
    ``context`` to ``predict_custom_match``. If selected players have no stats and
    no usable registry tier, the returned context remains neutral through
    ``*_player_context_available = 0``.
    """
    registry = registry if registry is not None else _empty_frame(PLAYER_REGISTRY_COLUMNS)
    match_stats = match_stats if match_stats is not None else _empty_frame(PLAYER_MATCH_STATS_COLUMNS)
    league_norm = _normalize_league(league)
    home_norm = normalize_team_name(home_team, league_norm)
    away_norm = normalize_team_name(away_team, league_norm)
    match_day = _as_of_timestamp(match_date).normalize()
    kickoff_at = match_day.isoformat()

    home_starters = _clean_player_ids(home_starters)
    away_starters = _clean_player_ids(away_starters)
    home_absence_items = _clean_absence_items(
        home_absences,
        default_absence_type=default_absence_type,
        default_status=default_absence_status,
    )
    away_absence_items = _clean_absence_items(
        away_absences,
        default_absence_type=default_absence_type,
        default_status=default_absence_status,
    )

    lineup_rows = []
    for team, player_ids in ((home_norm, home_starters), (away_norm, away_starters)):
        for player_id in player_ids:
            lineup_rows.append({
                "date": match_day,
                "kickoff_at": kickoff_at,
                "league": league_norm,
                "home_team": home_norm,
                "away_team": away_norm,
                "team": team,
                "player_id": player_id,
                "is_starter": 1,
                "is_sub": 0,
                "source": "manual_runtime",
                "available_at": kickoff_at,
            })
    lineups = (
        pd.DataFrame(lineup_rows, columns=MATCH_LINEUPS_COLUMNS)
        if lineup_rows else _empty_frame(MATCH_LINEUPS_COLUMNS)
    )

    absence_rows = []
    for team, items in ((home_norm, home_absence_items), (away_norm, away_absence_items)):
        for item in items:
            absence_rows.append({
                "date": match_day,
                "kickoff_at": kickoff_at,
                "league": league_norm,
                "home_team": home_norm,
                "away_team": away_norm,
                "team": team,
                "player_id": item["player_id"],
                "absence_type": item["absence_type"],
                "status": item["status"],
                "source": "manual_runtime",
                "available_at": kickoff_at,
            })
    absences = (
        pd.DataFrame(absence_rows, columns=MATCH_ABSENCES_COLUMNS)
        if absence_rows else _empty_frame(MATCH_ABSENCES_COLUMNS)
    )

    if index is None:
        index = build_player_strength_index(match_stats, registry, window=window)
    context_frame = build_player_match_context(
        registry,
        match_stats,
        lineups,
        absences,
        window=window,
        index=index,
    )
    if context_frame.empty:
        diagnostics = pd.DataFrame(columns=RUNTIME_PLAYER_DIAGNOSTIC_COLUMNS)
        return {}, diagnostics

    context = {}
    fixture_keys = {"date", "league", "home_team", "away_team"}
    row = context_frame.iloc[0]
    for col in PLAYER_MATCH_CONTEXT_COLUMNS:
        if col in fixture_keys:
            continue
        value = row.get(col, np.nan)
        if pd.notna(value):
            context[col] = float(value)

    diagnostic_rows = []
    diagnostic_rows.extend(_runtime_diagnostic_rows(
        home_starters,
        side="home",
        role="starter",
        team=home_norm,
        league=league_norm,
        as_of_date=match_day,
        index=index,
        window=window,
    ))
    diagnostic_rows.extend(_runtime_diagnostic_rows(
        away_starters,
        side="away",
        role="starter",
        team=away_norm,
        league=league_norm,
        as_of_date=match_day,
        index=index,
        window=window,
    ))
    for side, team, items in (
        ("home", home_norm, home_absence_items),
        ("away", away_norm, away_absence_items),
    ):
        for item in items:
            diagnostic_rows.extend(_runtime_diagnostic_rows(
                [item["player_id"]],
                side=side,
                role="absence",
                team=team,
                league=league_norm,
                as_of_date=match_day,
                index=index,
                window=window,
                absence_type=item["absence_type"],
                status=item["status"],
            ))
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics = _ensure_columns(diagnostics, RUNTIME_PLAYER_DIAGNOSTIC_COLUMNS)
    return context, diagnostics


def merge_player_match_context(existing: pd.DataFrame, player_context: pd.DataFrame) -> pd.DataFrame:
    """Merge generated player context into an existing ``match_context`` export.

    Generated non-null player fields take precedence, while unrelated existing
    columns such as weather, API ids, and source trace fields are preserved.
    """
    keys = ["date", "league", "home_team", "away_team"]
    if existing is None or existing.empty:
        return player_context.copy() if player_context is not None else _empty_player_match_context()
    if player_context is None or player_context.empty:
        return existing.copy()

    left = existing.copy()
    right = player_context.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    left["league"] = left["league"].map(_normalize_league)
    left = _normalize_team_columns(left, ("home_team", "away_team"))
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    right["league"] = right["league"].map(_normalize_league)
    right = _normalize_team_columns(right, ("home_team", "away_team"))

    merged = left.merge(right, on=keys, how="outer", suffixes=("", "_player"))
    for col in [c for c in PLAYER_MATCH_CONTEXT_COLUMNS if c not in keys]:
        player_col = f"{col}_player"
        if player_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[player_col].combine_first(merged[col])
            else:
                merged[col] = merged[player_col]
            merged = merged.drop(columns=[player_col])
    return merged


def build_lineup_strength_context(
    match_stats: pd.DataFrame,
    index: PlayerStrengthIndex | None = None,
    *,
    window: int = 10,
) -> pd.DataFrame:
    """Bulk-generate per-fixture home/away ``lineup_strength`` from player match stats.

    This is the scalable counterpart to :func:`build_player_match_context` for the whole
    history: it reconstructs each match's starters from ``match_stats`` (``started == 1``;
    home/away recovered from the ``home_away`` flag), then scores each XI with a shared
    :class:`PlayerStrengthIndex` (the (team, date) window is computed once per fixture).
    Only matches before each fixture date feed a player's strength, so it is leakage-safe.

    Returns a frame with :data:`PLAYER_MATCH_CONTEXT_COLUMNS` (lineup fields filled,
    absence fields left NaN) ready for :func:`merge_player_match_context`.
    """
    if match_stats is None or match_stats.empty:
        return _empty_player_match_context()

    df = match_stats.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _normalize_leagues(df)
    df = _normalize_team_columns_fast(df, ("team", "opponent"))
    df["player_id"] = _normalize_string_series(df["player_id"])
    df = df.dropna(subset=["date"])

    if index is None:
        index = build_player_strength_index(df, None, window=window, assume_normalized=True)

    is_home = df["home_away"].astype(str).str.lower().eq("home")
    df["home_team"] = df["team"].where(is_home, df["opponent"])
    df["away_team"] = df["opponent"].where(is_home, df["team"])
    started = pd.to_numeric(df["started"], errors="coerce").fillna(0.0)
    starters = df[started == 1]

    fixtures: dict[tuple, dict] = {}
    for (date, league, home, away, team), group in starters.groupby(
        ["date", "league", "home_team", "away_team", "team"], sort=False
    ):
        strengths = index.fixture_strengths(team, league, date, group["player_id"].tolist(), window=window)
        available = [s.player_strength for s in strengths if s.context_available]
        lineup_strength = float(np.mean(available)) if available else np.nan
        side = "home" if team == home else "away"
        fixture = fixtures.setdefault((date, league, home, away), {})
        fixture[f"{side}_lineup_strength"] = lineup_strength
        fixture[f"{side}_available"] = 1.0 if available else 0.0

    rows = []
    for (date, league, home, away), fixture in fixtures.items():
        home_ls = fixture.get("home_lineup_strength", np.nan)
        away_ls = fixture.get("away_lineup_strength", np.nan)
        row = {col: np.nan for col in PLAYER_MATCH_CONTEXT_COLUMNS}
        row["date"], row["league"], row["home_team"], row["away_team"] = date, league, home, away
        row["home_lineup_strength"] = home_ls
        row["away_lineup_strength"] = away_ls
        row["home_player_context_available"] = fixture.get("home_available", 0.0)
        row["away_player_context_available"] = fixture.get("away_available", 0.0)
        row["lineup_available"] = 1.0 if (np.isfinite(home_ls) or np.isfinite(away_ls)) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    return _ensure_columns(out, PLAYER_MATCH_CONTEXT_COLUMNS)
