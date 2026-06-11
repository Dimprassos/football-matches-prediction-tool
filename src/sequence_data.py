"""Leakage-safe per-team match *sequences* for the deep-learning (FootyNet) model.

The tabular models consume a single engineered feature vector per fixture. The
recurrent model (see ``docs/DEEP_LEARNING_DESIGN.md``) instead needs, for each
fixture, the **last K matches** of the home team and of the away team as an ordered
sequence — so an LSTM can learn the form representation end-to-end (grounded in
paper11 Danisik "LSTM many-to-one" and paper3 "last 1-5 matches").

Leakage rule (identical to :func:`state_builder._recent_team_means` /
``streaming_block_probs_home_away``): a fixture on date ``D`` may only use a team's
matches with ``date < D``. Each historical match is encoded **from that team's own
perspective** (goals for/against, result one-hot, shots, xG, home flag, points,
rest). Market/Elo/understat-aggregate signals are intentionally left to the model's
static branch — the sequence branch is a pure on-pitch form representation.

Sequences are returned oldest->newest and front-padded with zeros (+ a 0/1 mask)
when a team has fewer than K prior matches, so the network can mask the padding.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# One ordered record per (team, past match), from that team's perspective.
TEAM_MATCH_FEATURES = [
    "goals_for",
    "goals_against",
    "result_win",      # one-hot of the match result for this team (paper11: prev-result one-hot)
    "result_draw",
    "result_loss",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "xg_for",          # understat (0.0 when missing; ~85% coverage)
    "xg_against",
    "is_home",         # paper11: home / was-home flag
    "points",          # 3 / 1 / 0 (paper11: obtained-points-percentage signal)
    "days_rest",       # days since this team's previous match, capped, in weeks
]
SEQ_FEATURE_DIM = len(TEAM_MATCH_FEATURES)

DEFAULT_SEQUENCE_LENGTH = 5  # K (paper11 last-5; tunable 5-10)
_REST_DEFAULT_DAYS = 7.0
_REST_CAP_DAYS = 21.0


def _finite(value, default: float = 0.0) -> float:
    """Coerce to float, returning ``default`` for missing/non-finite values."""
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if np.isfinite(value) else float(default)


def _team_perspective_features(row: pd.Series, team: str) -> dict:
    """Encode one played match from ``team``'s perspective (all but ``days_rest``).

    ``days_rest`` is filled later in :func:`build_team_sequences` once a team's
    matches are sorted chronologically.
    """
    is_home = row["home_team"] == team
    prefix = "home" if is_home else "away"
    opp = "away" if is_home else "home"

    goals_for = _finite(row.get(f"{prefix}_goals"))
    goals_against = _finite(row.get(f"{opp}_goals"))
    if goals_for > goals_against:
        win, draw, loss, points = 1.0, 0.0, 0.0, 3.0
    elif goals_for == goals_against:
        win, draw, loss, points = 0.0, 1.0, 0.0, 1.0
    else:
        win, draw, loss, points = 0.0, 0.0, 1.0, 0.0

    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "result_win": win,
        "result_draw": draw,
        "result_loss": loss,
        "shots_for": _finite(row.get(f"{prefix}_shots")),
        "shots_against": _finite(row.get(f"{opp}_shots")),
        "sot_for": _finite(row.get(f"{prefix}_shots_target")),
        "sot_against": _finite(row.get(f"{opp}_shots_target")),
        "xg_for": _finite(row.get(f"{prefix}_understat_xg")),
        "xg_against": _finite(row.get(f"{opp}_understat_xg")),
        "is_home": 1.0 if is_home else 0.0,
        "points": points,
        "days_rest": _REST_DEFAULT_DAYS / 7.0,  # placeholder, recomputed below
    }


def build_team_sequences(full_df: pd.DataFrame) -> dict[str, dict]:
    """Pre-compute each team's chronological match-feature matrix (played matches only).

    Returns ``{team: {"dates": np.ndarray[datetime64], "feats": np.ndarray[n, F]}}``
    sorted ascending by date, with ``days_rest`` derived from consecutive matches.
    Built once per league; lookups are then O(log n + K) per fixture.
    """
    if full_df is None or len(full_df) == 0:
        return {}
    played = full_df[full_df.get("is_played", True) == True].copy()  # noqa: E712
    played["date"] = pd.to_datetime(played["date"], errors="coerce")
    played = played.dropna(subset=["date"]).sort_values("date")

    per_team: dict[str, list[tuple[pd.Timestamp, dict]]] = defaultdict(list)
    for _, row in played.iterrows():
        for team in (row["home_team"], row["away_team"]):
            per_team[team].append((row["date"], _team_perspective_features(row, team)))

    out: dict[str, dict] = {}
    for team, records in per_team.items():
        records.sort(key=lambda r: r[0])
        dates = np.array([r[0] for r in records], dtype="datetime64[ns]")
        prev_date: pd.Timestamp | None = None
        rows = []
        for date, feat in records:
            if prev_date is not None:
                days = max(0, int((date - prev_date).days))
                feat["days_rest"] = float(min(days, _REST_CAP_DAYS)) / 7.0
            prev_date = date
            rows.append([feat[name] for name in TEAM_MATCH_FEATURES])
        out[team] = {"dates": dates, "feats": np.asarray(rows, dtype=float)}
    return out


def last_k_before(
    team_seq: dict | None,
    as_of_date,
    k: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Last ``k`` matches of a team strictly before ``as_of_date`` (oldest->newest).

    Returns ``(seq[k, F], mask[k])``; front-padded with zeros and ``mask=0`` when the
    team has fewer than ``k`` prior matches (or none / unknown team).
    """
    seq = np.zeros((k, SEQ_FEATURE_DIM), dtype=float)
    mask = np.zeros(k, dtype=float)
    if team_seq is None:
        return seq, mask
    as_of = np.datetime64(pd.Timestamp(as_of_date), "ns")
    # number of matches strictly before as_of (side="left" excludes equal dates)
    lo = int(np.searchsorted(team_seq["dates"], as_of, side="left"))
    if lo <= 0:
        return seq, mask
    start = max(0, lo - k)
    window = team_seq["feats"][start:lo]
    n = window.shape[0]
    seq[k - n:] = window      # right-align: newest at the end
    mask[k - n:] = 1.0
    return seq, mask


def build_fixture_sequences(
    full_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    k: int = DEFAULT_SEQUENCE_LENGTH,
    *,
    team_sequences: dict[str, dict] | None = None,
) -> dict[str, np.ndarray]:
    """Build aligned home/away sequence tensors for every fixture in ``fixtures_df``.

    ``full_df`` provides the match history (played matches feed the sequences);
    ``fixtures_df`` are the rows to score (must have ``date``/``home_team``/``away_team``).
    Each fixture uses only matches strictly before its date — leakage-safe and
    identical to the streaming feature builder.

    Returns a dict with ``seq_home``/``seq_away`` of shape ``[N, k, F]`` and
    ``mask_home``/``mask_away`` of shape ``[N, k]``.
    """
    if team_sequences is None:
        team_sequences = build_team_sequences(full_df)

    n = len(fixtures_df)
    seq_home = np.zeros((n, k, SEQ_FEATURE_DIM), dtype=float)
    seq_away = np.zeros((n, k, SEQ_FEATURE_DIM), dtype=float)
    mask_home = np.zeros((n, k), dtype=float)
    mask_away = np.zeros((n, k), dtype=float)

    for i, (_, row) in enumerate(fixtures_df.iterrows()):
        d = pd.to_datetime(row["date"])
        sh, mh = last_k_before(team_sequences.get(row["home_team"]), d, k)
        sa, ma = last_k_before(team_sequences.get(row["away_team"]), d, k)
        seq_home[i], mask_home[i] = sh, mh
        seq_away[i], mask_away[i] = sa, ma

    return {
        "seq_home": seq_home,
        "seq_away": seq_away,
        "mask_home": mask_home,
        "mask_away": mask_away,
    }
