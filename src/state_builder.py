"""Per-league state, recent-form/context features, and the leakage-safe streaming pass.

This module turns raw match history into the inputs every model consumes:

* :class:`LeagueState` holds the rolling Elo ratings, per-team strengths, and
  history needed to score a new fixture as of a given date.
* :func:`compute_match_components` produces the base-model probabilities plus the
  12-element "base aux" vector (elo/lambda/momentum/rest/form differentials).
* :func:`compute_pre_match_extra_features` reconstructs the "extra aux" vector
  (recent form, understat xG, market movement, lineup/injury/weather context) from
  past matches only — so it is identical at train and serve time.
* :func:`streaming_block_probs_home_away` walks the fixtures chronologically,
  updating state after each matchday, guaranteeing no future information leaks into
  a match's features.

Feature layout: the full feature vector is ``6 market/base outputs + BASE_AUX_LEN
base aux + EXTRA_AUX_LEN extra aux`` (see :data:`feature_builder.FEATURE_COLUMNS`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.elo import expected_score, match_result, margin_multiplier
from src.feature_builder import FEATURE_COLUMNS, feature_indices, market_probs_from_odds_row
from src.poisson_model import (
    apply_elo_to_lambdas,
    fit_team_strengths_home_away_weighted,
    match_outcome_probs_dc,
    predict_lambdas_home_away,
)


@dataclass
class LeagueState:
    """Snapshot of a league's modelling state as of the most recent processed match.

    Bundles the Elo ratings and history, the home/away attack/defence strengths,
    the league average scoring rates, the tuned base-model ``params``, and
    (optionally) ``played_df`` — the understat-merged match history kept so the
    runtime predictor can reconstruct pre-match extra features for an arbitrary
    fixture.
    """

    ratings: Dict[str, float]
    elo_history: Dict[str, list[float]]
    last_match_date: Dict[str, pd.Timestamp]
    points_history: Dict[str, list[float]]
    attack_home: dict
    defense_home: dict
    attack_away: dict
    defense_away: dict
    league_avg_home: float
    league_avg_away: float
    params: dict
    # Played-match history (incl. understat columns) used to reconstruct
    # pre-match extra features at serve time. Optional for backward compat.
    played_df: pd.DataFrame | None = None


# The feature vector is: 6 base/market outputs, then 12 "base aux" features
# (elo/lambda/momentum/rest/form), then the remaining "extra aux" features
# (recent form, understat, market movement, lineup/injury/weather context).
BASE_AUX_LEN = 12
EXTRA_AUX_LEN = len(FEATURE_COLUMNS) - 6 - BASE_AUX_LEN


def odds_triplet_from_row(row: pd.Series, source: str = "closing") -> tuple[float, float, float]:
    """Extract (home, draw, away) decimal odds for the requested source.

    ``source`` is "opening", "closing", or "legacy". Closing odds fall back to the
    legacy ``odds_*`` columns when any closing value is missing; non-finite values
    are returned as NaN so downstream code can detect missing markets.
    """
    if source == "opening":
        cols = ("open_odds_home", "open_odds_draw", "open_odds_away")
    elif source == "closing":
        cols = ("close_odds_home", "close_odds_draw", "close_odds_away")
    elif source == "legacy":
        cols = ("odds_home", "odds_draw", "odds_away")
    else:
        raise ValueError(f"Unknown odds source: {source!r}")

    values = [pd.to_numeric(row.get(col, np.nan), errors="coerce") for col in cols]
    if source == "closing" and not all(np.isfinite(value) for value in values):
        values = [
            pd.to_numeric(row.get("odds_home", np.nan), errors="coerce"),
            pd.to_numeric(row.get("odds_draw", np.nan), errors="coerce"),
            pd.to_numeric(row.get("odds_away", np.nan), errors="coerce"),
        ]
    return tuple(float(value) if np.isfinite(value) else np.nan for value in values)


def dynamic_init_rating(ratings: Dict[str, float], init_rating: float = 1500.0) -> float:
    """Initial Elo for a newly-seen team: the mean of the three weakest current teams.

    Promoted/new teams should not enter at the league-average 1500; once enough
    teams exist we seed them near the bottom of the table instead.
    """
    if len(ratings) >= 5:
        bottom_elos = sorted(ratings.values())[:3]
        return float(sum(bottom_elos) / len(bottom_elos))
    return float(init_rating)


def update_elo_state(
    matches_batch: pd.DataFrame,
    ratings: Dict[str, float],
    elo_history: Dict[str, list[float]],
    last_match_date: Dict[str, pd.Timestamp],
    points_history: Dict[str, list[float]],
    *,
    K: float,
    home_adv: float,
    init_rating: float = 1500.0,
):
    """Update Elo ratings, Elo history, last-played dates, and points history in place.

    Iterates the batch chronologically applying a margin-of-victory-scaled Elo
    update with a home advantage, and records 3/1/0 points per result. Mutates and
    returns the four state dicts.
    """
    for _, m in matches_batch.iterrows():
        h, a = m["home_team"], m["away_team"]
        dyn_init = dynamic_init_rating(ratings, init_rating)
        r_h = ratings.get(h, dyn_init)
        r_a = ratings.get(a, dyn_init)

        ratings.setdefault(h, r_h)
        ratings.setdefault(a, r_a)
        elo_history.setdefault(h, [])
        elo_history.setdefault(a, [])
        points_history.setdefault(h, [])
        points_history.setdefault(a, [])

        exp_h = expected_score(r_h + home_adv, r_a)
        s_h, s_a = match_result(int(m["home_goals"]), int(m["away_goals"]))
        mult = margin_multiplier(int(m["home_goals"]) - int(m["away_goals"]))

        new_r_h = r_h + (K * mult) * (s_h - exp_h)
        new_r_a = r_a + (K * mult) * (s_a - (1.0 - exp_h))

        ratings[h] = new_r_h
        ratings[a] = new_r_a
        elo_history[h].append(new_r_h)
        elo_history[a].append(new_r_a)
        if s_h == 1.0:
            points_history[h].append(3.0)
            points_history[a].append(0.0)
        elif s_h == 0.5:
            points_history[h].append(1.0)
            points_history[a].append(1.0)
        else:
            points_history[h].append(0.0)
            points_history[a].append(3.0)
        match_date = pd.Timestamp(m["date"])
        last_match_date[h] = match_date
        last_match_date[a] = match_date
    return ratings, elo_history, last_match_date, points_history


def get_team_momentum(team: str, current_rating: float, elo_history: Dict[str, list[float]], window: int = 4) -> float:
    """Elo change over the last ``window`` matches, scaled by 400 (0.0 if not enough history)."""
    if team not in elo_history or len(elo_history[team]) < window:
        return 0.0
    return (current_rating - elo_history[team][-window]) / 400.0


def get_recent_points_form(team: str, points_history: Dict[str, list[float]], window: int = 5) -> float:
    """Average points-per-match over the last ``window`` games, normalised to [0, 1]."""
    team_points = points_history.get(team, [])
    if not team_points:
        return 0.0
    recent = team_points[-window:]
    return float(sum(recent)) / (3.0 * len(recent))


def get_rest_days(team: str, match_date: pd.Timestamp, last_match_date: Dict[str, pd.Timestamp], default_days: int = 7, cap_days: int = 21) -> float:
    """Days since the team last played, capped at ``cap_days`` and expressed in weeks."""
    last_date = last_match_date.get(team)
    if last_date is None:
        return float(default_days) / 7.0
    days = max(0, int((pd.Timestamp(match_date) - pd.Timestamp(last_date)).days))
    days = min(days, cap_days)
    return float(days) / 7.0


def _recent_team_means(team: str, past_matches: pd.DataFrame, window: int = 5) -> dict:
    """Mean of a team's last ``window`` matches across goals/shots/corners/cards/understat.

    Looks at the team whether it played home or away (picking the right column
    prefix each time) and averages only finite values. Returns zeros when there is
    no history. This is the source of the recent-form and understat-xG extra features.
    """
    defaults = {
        "goals_for": 0.0,
        "goals_against": 0.0,
        "shots_for": 0.0,
        "shots_against": 0.0,
        "sot_for": 0.0,
        "sot_against": 0.0,
        "corners_for": 0.0,
        "corners_against": 0.0,
        "cards": 0.0,
        "uxg": 0.0,
        "unpxg": 0.0,
        "uxpts": 0.0,
    }
    if past_matches.empty:
        return defaults.copy()

    team_matches = past_matches[
        (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
    ].sort_values("date").tail(window)
    if team_matches.empty:
        return defaults.copy()

    values = {key: [] for key in defaults}
    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        prefix = "home" if is_home else "away"
        opp_prefix = "away" if is_home else "home"
        goals_for = pd.to_numeric(row.get(f"{prefix}_goals"), errors="coerce")
        goals_against = pd.to_numeric(row.get(f"{opp_prefix}_goals"), errors="coerce")
        shots = pd.to_numeric(row.get(f"{prefix}_shots"), errors="coerce")
        shots_against = pd.to_numeric(row.get(f"{opp_prefix}_shots"), errors="coerce")
        sot = pd.to_numeric(row.get(f"{prefix}_shots_target"), errors="coerce")
        sot_against = pd.to_numeric(row.get(f"{opp_prefix}_shots_target"), errors="coerce")
        corners = pd.to_numeric(row.get(f"{prefix}_corners"), errors="coerce")
        corners_against = pd.to_numeric(row.get(f"{opp_prefix}_corners"), errors="coerce")
        yellows = pd.to_numeric(row.get(f"{prefix}_yellows"), errors="coerce")
        reds = pd.to_numeric(row.get(f"{prefix}_reds"), errors="coerce")
        uxg = pd.to_numeric(row.get(f"{prefix}_understat_xg"), errors="coerce")
        unpxg = pd.to_numeric(row.get(f"{prefix}_understat_npxg"), errors="coerce")
        uxpts = pd.to_numeric(row.get(f"{prefix}_understat_xpts"), errors="coerce")

        if np.isfinite(goals_for):
            values["goals_for"].append(float(goals_for))
        if np.isfinite(goals_against):
            values["goals_against"].append(float(goals_against))
        if np.isfinite(shots):
            values["shots_for"].append(float(shots))
        if np.isfinite(shots_against):
            values["shots_against"].append(float(shots_against))
        if np.isfinite(sot):
            values["sot_for"].append(float(sot))
        if np.isfinite(sot_against):
            values["sot_against"].append(float(sot_against))
        if np.isfinite(corners):
            values["corners_for"].append(float(corners))
        if np.isfinite(corners_against):
            values["corners_against"].append(float(corners_against))
        card_score = (float(yellows) if np.isfinite(yellows) else 0.0) + 2.0 * (float(reds) if np.isfinite(reds) else 0.0)
        values["cards"].append(card_score)
        if np.isfinite(uxg):
            values["uxg"].append(float(uxg))
        if np.isfinite(unpxg):
            values["unpxg"].append(float(unpxg))
        if np.isfinite(uxpts):
            values["uxpts"].append(float(uxpts))

    return {key: float(np.mean(val)) if val else 0.0 for key, val in values.items()}


def neutral_extra_features() -> np.ndarray:
    """Zero-valued extra-aux vector with sensible neutral defaults for non-zero features.

    Used as a fallback when no history is available (e.g. ou25_over_prob=0.5,
    lineup strengths=1.0, temperature=15C). Positions are looked up by name so the
    vector stays aligned with FEATURE_COLUMNS.
    """
    extra = np.zeros(EXTRA_AUX_LEN, dtype=float)
    defaults = {
        "ou25_over_prob": 0.5,
        "home_lineup_strength": 1.0,
        "away_lineup_strength": 1.0,
        "temperature_c": 15.0,
    }
    for feature_name, value in defaults.items():
        idx = feature_indices([feature_name])[0] - 6 - BASE_AUX_LEN
        extra[idx] = value
    return extra


def _finite_or_default(value, default: float) -> float:
    """Coerce ``value`` to float, returning ``default`` when it is missing/non-finite."""
    value = pd.to_numeric(value, errors="coerce")
    if np.isfinite(value):
        return float(value)
    return float(default)


def _weather_severity(temperature_c: float, wind_kph: float, precipitation_mm: float) -> float:
    """Combine temperature discomfort, wind, and rain into a single 0-5 severity score."""
    temp_discomfort = abs(float(temperature_c) - 15.0) / 20.0
    wind_component = max(0.0, float(wind_kph)) / 40.0
    rain_component = max(0.0, float(precipitation_mm)) / 10.0
    return float(min(5.0, temp_discomfort + wind_component + rain_component))


def compute_pre_match_extra_features(
    row: pd.Series,
    past_matches: pd.DataFrame,
    window: int = 5,
    *,
    include_market_movement_features: bool = True,
) -> np.ndarray:
    """Build the EXTRA_AUX_LEN feature vector for one fixture from past matches only.

    Combines home/away recent-form means and their differentials, market movement
    (close minus open probabilities, zeroed when disabled or missing), over/under and
    Asian-handicap signals, understat xG/npxG/xpts form, and lineup/injury/weather
    context. The element order is fixed and must match FEATURE_COLUMNS; a length
    check guards against drift.
    """
    home = row["home_team"]
    away = row["away_team"]
    home_recent = _recent_team_means(home, past_matches, window=window)
    away_recent = _recent_team_means(away, past_matches, window=window)

    open_probs = market_probs_from_odds_row(*odds_triplet_from_row(row, "opening"))
    close_probs = market_probs_from_odds_row(*odds_triplet_from_row(row, "closing"))
    if include_market_movement_features and np.isfinite(open_probs).all() and np.isfinite(close_probs).all():
        market_move = close_probs - open_probs
    else:
        market_move = np.zeros(3, dtype=float)

    ou25_over_prob = _finite_or_default(row.get("ou25_over_prob", np.nan), 0.5)
    ah_line = _finite_or_default(row.get("ah_line", np.nan), 0.0)
    lineup_available = _finite_or_default(row.get("lineup_available", np.nan), 0.0)
    home_lineup_strength = _finite_or_default(row.get("home_lineup_strength", np.nan), 1.0)
    away_lineup_strength = _finite_or_default(row.get("away_lineup_strength", np.nan), 1.0)
    team_news_available = _finite_or_default(row.get("team_news_available", np.nan), 0.0)
    home_absence_count = _finite_or_default(row.get("home_absence_count", np.nan), 0.0)
    away_absence_count = _finite_or_default(row.get("away_absence_count", np.nan), 0.0)
    home_injury_count = _finite_or_default(row.get("home_injury_count", np.nan), 0.0)
    away_injury_count = _finite_or_default(row.get("away_injury_count", np.nan), 0.0)
    home_suspension_count = _finite_or_default(row.get("home_suspension_count", np.nan), 0.0)
    away_suspension_count = _finite_or_default(row.get("away_suspension_count", np.nan), 0.0)
    home_key_absence_count = _finite_or_default(row.get("home_key_absence_count", np.nan), 0.0)
    away_key_absence_count = _finite_or_default(row.get("away_key_absence_count", np.nan), 0.0)
    home_manager_change_recent = _finite_or_default(row.get("home_manager_change_recent", np.nan), 0.0)
    away_manager_change_recent = _finite_or_default(row.get("away_manager_change_recent", np.nan), 0.0)
    weather_available = _finite_or_default(row.get("weather_available", np.nan), 0.0)
    temperature_c = _finite_or_default(row.get("temperature_c", np.nan), 15.0)
    wind_kph = _finite_or_default(row.get("wind_kph", np.nan), 0.0)
    precipitation_mm = _finite_or_default(row.get("precipitation_mm", np.nan), 0.0)
    weather_severity = _weather_severity(temperature_c, wind_kph, precipitation_mm)
    home_absence_strength_loss = _finite_or_default(row.get("home_absence_strength_loss", np.nan), 0.0)
    away_absence_strength_loss = _finite_or_default(row.get("away_absence_strength_loss", np.nan), 0.0)
    absence_strength_loss_diff = _finite_or_default(
        row.get("absence_strength_loss_diff", np.nan),
        home_absence_strength_loss - away_absence_strength_loss,
    )
    home_player_context_available = _finite_or_default(row.get("home_player_context_available", np.nan), 0.0)
    away_player_context_available = _finite_or_default(row.get("away_player_context_available", np.nan), 0.0)

    extra = np.array([
        home_recent["goals_for"],
        away_recent["goals_for"],
        home_recent["goals_for"] - away_recent["goals_for"],
        home_recent["goals_against"],
        away_recent["goals_against"],
        home_recent["goals_against"] - away_recent["goals_against"],
        home_recent["shots_for"],
        away_recent["shots_for"],
        home_recent["shots_for"] - away_recent["shots_for"],
        home_recent["shots_against"],
        away_recent["shots_against"],
        home_recent["shots_against"] - away_recent["shots_against"],
        home_recent["sot_for"],
        away_recent["sot_for"],
        home_recent["sot_for"] - away_recent["sot_for"],
        home_recent["sot_against"],
        away_recent["sot_against"],
        home_recent["sot_against"] - away_recent["sot_against"],
        home_recent["corners_for"],
        away_recent["corners_for"],
        home_recent["corners_for"] - away_recent["corners_for"],
        home_recent["corners_against"],
        away_recent["corners_against"],
        home_recent["corners_against"] - away_recent["corners_against"],
        home_recent["cards"],
        away_recent["cards"],
        home_recent["cards"] - away_recent["cards"],
        market_move[0],
        market_move[1],
        market_move[2],
        ou25_over_prob,
        ah_line,
        home_recent["uxg"],
        away_recent["uxg"],
        home_recent["uxg"] - away_recent["uxg"],
        home_recent["unpxg"],
        away_recent["unpxg"],
        home_recent["unpxg"] - away_recent["unpxg"],
        home_recent["uxpts"],
        away_recent["uxpts"],
        home_recent["uxpts"] - away_recent["uxpts"],
        lineup_available,
        home_lineup_strength,
        away_lineup_strength,
        home_lineup_strength - away_lineup_strength,
        team_news_available,
        home_absence_count,
        away_absence_count,
        home_absence_count - away_absence_count,
        home_injury_count,
        away_injury_count,
        home_injury_count - away_injury_count,
        home_suspension_count,
        away_suspension_count,
        home_suspension_count - away_suspension_count,
        home_key_absence_count,
        away_key_absence_count,
        home_key_absence_count - away_key_absence_count,
        home_manager_change_recent,
        away_manager_change_recent,
        home_manager_change_recent - away_manager_change_recent,
        weather_available,
        temperature_c,
        wind_kph,
        precipitation_mm,
        weather_severity,
        home_absence_strength_loss,
        away_absence_strength_loss,
        absence_strength_loss_diff,
        home_player_context_available,
        away_player_context_available,
    ], dtype=float)
    if len(extra) != EXTRA_AUX_LEN:
        raise ValueError(f"Expected {EXTRA_AUX_LEN} extra features, got {len(extra)}")
    return extra


def build_league_state(played_df: pd.DataFrame, params: dict) -> LeagueState:
    """Construct a :class:`LeagueState` from the full played history and tuned params.

    Fits the time-decayed home/away strengths and replays Elo over every match so
    the returned state reflects all known results. Retains ``played_df`` for serve-time
    feature reconstruction. Used by the runtime predictor (not the streaming pass).
    """
    played_df = played_df.sort_values("date").reset_index(drop=True)
    l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(played_df, decay=params["decay"])

    ratings: Dict[str, float] = {}
    elo_history: Dict[str, list[float]] = {}
    last_match_date: Dict[str, pd.Timestamp] = {}
    points_history: Dict[str, list[float]] = {}
    update_elo_state(
        played_df,
        ratings,
        elo_history,
        last_match_date,
        points_history,
        K=params["K"],
        home_adv=params["ha"],
    )

    return LeagueState(
        ratings=ratings,
        elo_history=elo_history,
        last_match_date=last_match_date,
        points_history=points_history,
        attack_home=att_h,
        defense_home=def_h,
        attack_away=att_a,
        defense_away=def_a,
        league_avg_home=l_avg_h,
        league_avg_away=l_avg_a,
        params=params,
        played_df=played_df,
    )


def compute_match_components(home_team: str, away_team: str, state: LeagueState, match_date: pd.Timestamp | None = None, extra_aux=None):
    """Compute base-model probabilities and the aux feature vector for one fixture.

    Predicts Dixon-Coles goal rates (Elo-adjusted), derives the 12 base-aux features
    (elo/lambda/momentum/rest/form and their differentials), and appends ``extra_aux``
    when supplied. Returns a dict with ``probs``, ``aux``, and the individual components.
    """
    p = state.params
    dyn_init = dynamic_init_rating(state.ratings)
    elo_h = state.ratings.get(home_team, dyn_init)
    elo_a = state.ratings.get(away_team, dyn_init)
    if match_date is None:
        latest_seen = max(state.last_match_date.values()) if state.last_match_date else pd.Timestamp.today().normalize()
        match_date = latest_seen + pd.Timedelta(days=1)

    lam_h, lam_a = predict_lambdas_home_away(
        home_team,
        away_team,
        state.league_avg_home,
        state.league_avg_away,
        state.attack_home,
        state.defense_home,
        state.attack_away,
        state.defense_away,
    )
    lam_h, lam_a = apply_elo_to_lambdas(lam_h, lam_a, elo_h, elo_a, beta=p["beta"])
    probs = np.array(match_outcome_probs_dc(lam_h, lam_a, rho=p["rho"], max_goals=10), dtype=float)
    mom_h = get_team_momentum(home_team, elo_h, state.elo_history)
    mom_a = get_team_momentum(away_team, elo_a, state.elo_history)
    rest_h = get_rest_days(home_team, match_date, state.last_match_date)
    rest_a = get_rest_days(away_team, match_date, state.last_match_date)
    form_h = get_recent_points_form(home_team, state.points_history)
    form_a = get_recent_points_form(away_team, state.points_history)
    aux = np.array([
        (elo_h - elo_a) / 400.0,
        lam_h + lam_a,
        lam_h - lam_a,
        mom_h,
        mom_a,
        mom_h - mom_a,
        rest_h,
        rest_a,
        rest_h - rest_a,
        form_h,
        form_a,
        form_h - form_a,
    ], dtype=float)
    if extra_aux is not None:
        aux = np.concatenate([aux, np.asarray(extra_aux, dtype=float)])
    return {
        "elo_home": elo_h,
        "elo_away": elo_a,
        "lam_home": lam_h,
        "lam_away": lam_a,
        "probs": probs,
        "aux": aux,
        "mom_home": mom_h,
        "mom_away": mom_a,
        "rest_home": rest_h,
        "rest_away": rest_a,
        "form_home": form_h,
        "form_away": form_a,
    }


def streaming_block_probs_home_away(
    predict_df,
    full_df,
    beta,
    rho,
    decay,
    K,
    home_adv,
    init_rating=1500.0,
    max_goals=10,
    *,
    market_odds_source: str = "closing",
    betting_odds_source: str = "closing",
    include_market_movement_features: bool = True,
):
    """Score every fixture in ``predict_df`` chronologically, leakage-free.

    Walks matchday by matchday: state (Elo + strengths) is built only from matches
    strictly before each date, the day's fixtures are scored, then state is updated
    with that day's results. This is the canonical training/eval feature builder and
    guarantees a match never sees its own or future outcomes.

    Returns ``(model_probs, y_true, market_probs, aux_matrix, raw_betting_odds)``.
    """
    params = {"beta": beta, "rho": rho, "decay": decay, "K": K, "ha": home_adv}
    probs_model = []
    probs_mkt = []
    y_true = []
    aux = []
    raw_odds = []

    predict_df = predict_df.sort_values("date")
    full_df = full_df.sort_values("date")
    predict_dates = sorted(predict_df["date"].unique())
    if len(predict_dates) == 0:
        return (np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0, 3)), np.zeros((0, len(FEATURE_COLUMNS) - 6)), np.zeros((0, 3)))

    ratings = {}
    elo_history = {}
    last_match_date = {}
    points_history = {}
    history_matches = full_df[full_df["date"] < predict_dates[0]]
    update_elo_state(
        history_matches,
        ratings,
        elo_history,
        last_match_date,
        points_history,
        K=K,
        home_adv=home_adv,
        init_rating=init_rating,
    )

    for d in predict_dates:
        day_matches = predict_df[predict_df["date"] == d]
        past_matches = full_df[full_df["date"] < d]

        l_avg_h, l_avg_a, att_h, def_h, att_a, def_a = fit_team_strengths_home_away_weighted(
            past_matches, decay=decay
        )
        state = LeagueState(
            ratings=ratings.copy(),
            elo_history={k: v[:] for k, v in elo_history.items()},
            last_match_date=last_match_date.copy(),
            points_history={k: v[:] for k, v in points_history.items()},
            attack_home=att_h,
            defense_home=def_h,
            attack_away=att_a,
            defense_away=def_a,
            league_avg_home=l_avg_h,
            league_avg_away=l_avg_a,
            params={**params, "K": K, "ha": home_adv},
        )

        for _, row in day_matches.iterrows():
            market_odds = odds_triplet_from_row(row, market_odds_source)
            betting_odds = odds_triplet_from_row(row, betting_odds_source)
            raw_odds.append(list(betting_odds))
            extra_aux = compute_pre_match_extra_features(
                row,
                past_matches,
                include_market_movement_features=include_market_movement_features,
            )
            comp = compute_match_components(row["home_team"], row["away_team"], state, match_date=row["date"], extra_aux=extra_aux)
            probs_model.append(comp["probs"].tolist())
            probs_mkt.append(market_probs_from_odds_row(*market_odds).tolist())
            aux.append(comp["aux"].tolist())
            if row["home_goals"] > row["away_goals"]:
                y_true.append(0)
            elif row["home_goals"] == row["away_goals"]:
                y_true.append(1)
            else:
                y_true.append(2)

        update_elo_state(
            day_matches,
            ratings,
            elo_history,
            last_match_date,
            points_history,
            K=K,
            home_adv=home_adv,
            init_rating=init_rating,
        )

    return (
        np.array(probs_model, dtype=float),
        np.array(y_true, dtype=int),
        np.array(probs_mkt, dtype=float),
        np.array(aux, dtype=float),
        np.array(raw_odds, dtype=float),
    )
