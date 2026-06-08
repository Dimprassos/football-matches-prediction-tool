"""Elo rating computation with margin-of-victory scaling and dynamic team entry.

Provides the building blocks (expected score, match result, margin multiplier) and
a one-pass :func:`compute_elo_ratings` that returns the pre-match ratings for every
fixture in chronological order — the form of Elo used as a base-model feature.
"""
import pandas as pd
from typing import Dict, List, Tuple


def expected_score(r_home, r_away):
    """Logistic Elo win expectation for the home side given both ratings."""
    return 1 / (1 + 10 ** ((r_away - r_home) / 400))


def match_result(home_goals: int, away_goals: int) -> Tuple[float, float]:
    """Map a final score to Elo scores: (1,0) home win, (0,1) away win, (0.5,0.5) draw."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def margin_multiplier(goal_diff: int) -> float:
    """Scale the Elo update by margin of victory (bigger wins move ratings more)."""
    d = abs(int(goal_diff))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    if d == 3:
        return 1.75
    return 2.0


def get_dynamic_init(ratings: Dict[str, float], default_init: float) -> float:
    """
    Calculates the entry Elo rating for a newly promoted team based on the 
    average rating of the bottom 3 teams currently in the league.
    """
    if len(ratings) >= 5:
        bottom_elos = sorted(ratings.values())[:3]
        return float(sum(bottom_elos) / len(bottom_elos))
    return float(default_init)


def compute_elo_ratings(
    df: pd.DataFrame,
    K: float = 20.0,
    home_adv: float = 0.0,
    use_margin: bool = True,
    init_rating: float = 1500.0,
) -> List[Tuple[float, float]]:
    """
    Returns Elo ratings BEFORE each match in chronological order,
    using dynamic initialization for newly promoted teams.
    """
    ratings: Dict[str, float] = {}
    elo_history: List[Tuple[float, float]] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Determine the current baseline for new teams
        current_init = get_dynamic_init(ratings, init_rating)

        r_home = float(ratings.get(home, current_init))
        r_away = float(ratings.get(away, current_init))
        
        # Register new teams immediately to prevent double-counting as "new"
        if home not in ratings:
            ratings[home] = r_home
        if away not in ratings:
            ratings[away] = r_away

        # Store pre-match ratings
        elo_history.append((r_home, r_away))

        exp_home = expected_score(r_home + home_adv, r_away)
        exp_away = 1.0 - exp_home

        s_home, s_away = match_result(int(row["home_goals"]), int(row["away_goals"]))

        mult = 1.0
        if use_margin:
            mult = margin_multiplier(int(row["home_goals"]) - int(row["away_goals"]))

        r_home_new = r_home + (K * mult) * (s_home - exp_home)
        r_away_new = r_away + (K * mult) * (s_away - exp_away)

        ratings[home] = r_home_new
        ratings[away] = r_away_new

    return elo_history
