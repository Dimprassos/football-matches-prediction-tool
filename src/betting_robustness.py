"""Betting robustness and per-league strategy reports.

Stress-tests the betting edge by breaking ROI down across slices (season, league,
odds band) and, like :mod:`src.bet_selection`, selecting strategies on validation
and applying them to test. Answers "does the edge survive once we stop cherry-picking,
and is it concentrated in particular leagues/segments?".
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.artifact_store import append_rows_to_csv
from src.config import ExperimentConfig
from src.evaluation import betting_records


def _season_label(value: Any) -> str:
    date = pd.to_datetime(value)
    start_year = date.year if date.month >= 8 else date.year - 1
    return f"{start_year}-{start_year + 1}"


def _metric_row(run_ts: str, experiment_name: str, model: str, group_type: str, group_value: str, bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {
            "run_ts_utc": run_ts,
            "experiment_name": experiment_name,
            "model": model,
            "group_type": group_type,
            "group_value": group_value,
            "bets": 0,
            "wins": 0,
            "hit_rate": 0.0,
            "avg_odds": 0.0,
            "invested": 0.0,
            "returned": 0.0,
            "profit": 0.0,
            "roi": 0.0,
        }

    invested = float(bets["stake"].sum())
    returned = float(bets["return"].sum())
    profit = float(bets["profit"].sum())
    wins = int(bets["won"].sum())
    bet_count = int(len(bets))
    return {
        "run_ts_utc": run_ts,
        "experiment_name": experiment_name,
        "model": model,
        "group_type": group_type,
        "group_value": group_value,
        "bets": bet_count,
        "wins": wins,
        "hit_rate": round((wins / bet_count * 100.0) if bet_count else 0.0, 4),
        "avg_odds": round(float(bets["odds_taken"].mean()) if bet_count else 0.0, 4),
        "invested": round(invested, 6),
        "returned": round(returned, 6),
        "profit": round(profit, 6),
        "roi": round((profit / invested * 100.0) if invested > 0 else 0.0, 4),
    }


def _append_group_rows(rows: list[dict], run_ts: str, experiment_name: str, model: str, bets: pd.DataFrame, group_type: str, column: str):
    for value, group in bets.groupby(column, dropna=False):
        rows.append(_metric_row(run_ts, experiment_name, model, group_type, str(value), group))


def _model_betting_stats(probs: np.ndarray, raw_odds: np.ndarray, y_true: np.ndarray, match_info: list[dict], edge_threshold: float) -> tuple[dict, pd.DataFrame]:
    bets = betting_records(
        probs,
        raw_odds,
        y_true,
        edge_threshold=edge_threshold,
        match_info=match_info,
    )
    stats = _metric_row("", "", "", "", "", bets)
    return stats, bets


def _chronological_fold_masks(match_info: list[dict], n_folds: int = 2) -> list[np.ndarray]:
    dates = pd.to_datetime([row["date"] for row in match_info])
    order = np.argsort(dates.to_numpy())
    folds = []
    for fold_indices in np.array_split(order, n_folds):
        mask = np.zeros(len(match_info), dtype=bool)
        mask[fold_indices] = True
        folds.append(mask)
    return folds


def write_betting_robustness_report(
    config: ExperimentConfig,
    run_ts: str,
    strategy_probs: Mapping[str, np.ndarray],
    raw_odds: np.ndarray,
    y_true: np.ndarray,
    match_info: list[dict],
    *,
    edge_threshold: float,
) -> list[dict]:
    """Write per-model betting ROI broken down by season/league/odds-band plus a bet curve."""
    summary_rows: list[dict] = []
    curve_rows: list[dict] = []

    for model, probs in strategy_probs.items():
        bets = betting_records(
            probs,
            raw_odds,
            y_true,
            edge_threshold=edge_threshold,
            match_info=match_info,
        )
        if bets.empty:
            summary_rows.append(_metric_row(run_ts, config.experiment_name, model, "all", "all", bets))
            continue

        bets = bets.sort_values(["date", "idx"]).reset_index(drop=True)
        bets["date"] = pd.to_datetime(bets["date"])
        bets["league"] = bets["league"].fillna("unknown") if "league" in bets.columns else "unknown"
        bets["season"] = bets["date"].map(_season_label)
        bets["league_season"] = bets["league"].astype(str) + "|" + bets["season"].astype(str)

        summary_rows.append(_metric_row(run_ts, config.experiment_name, model, "all", "all", bets))
        _append_group_rows(summary_rows, run_ts, config.experiment_name, model, bets, "league", "league")
        _append_group_rows(summary_rows, run_ts, config.experiment_name, model, bets, "season", "season")
        _append_group_rows(summary_rows, run_ts, config.experiment_name, model, bets, "league_season", "league_season")

        cumulative_profit = 0.0
        cumulative_invested = 0.0
        for bet_number, row in enumerate(bets.itertuples(index=False), start=1):
            cumulative_profit += float(row.profit)
            cumulative_invested += float(row.stake)
            curve_rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "model": model,
                "bet_number": bet_number,
                "date": row.date.date().isoformat(),
                "league": row.league,
                "season": row.season,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "pred_label": row.pred_label,
                "true_label": row.y_true_label,
                "odds_taken": round(float(row.odds_taken), 4),
                "best_ev": round(float(row.best_ev), 6),
                "stake": round(float(row.stake), 6),
                "profit": round(float(row.profit), 6),
                "cumulative_profit": round(cumulative_profit, 6),
                "cumulative_roi": round((cumulative_profit / cumulative_invested * 100.0) if cumulative_invested > 0 else 0.0, 4),
            })

    append_rows_to_csv(config.final_betting_robustness_file, summary_rows)
    append_rows_to_csv(config.final_bet_curve_file, curve_rows)
    if config.print_full_reports:
        _print_betting_robustness_summary(summary_rows)
    return summary_rows


def _print_betting_robustness_summary(rows: list[dict]):
    print("\n" + "=" * 70)
    print("=== BETTING ROBUSTNESS SUMMARY ===")
    print("=" * 70)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No betting robustness rows produced.")
        return

    overall = df[df["group_type"] == "all"].sort_values("roi", ascending=False)
    print("\nOverall:")
    print(overall[["model", "bets", "hit_rate", "avg_odds", "profit", "roi"]].to_string(index=False))

    by_league = df[df["group_type"] == "league"].sort_values(["model", "roi"], ascending=[True, False])
    if not by_league.empty:
        print("\nBy league:")
        print(by_league[["model", "group_value", "bets", "hit_rate", "profit", "roi"]].to_string(index=False))

    by_season = df[df["group_type"] == "season"].sort_values(["model", "group_value"])
    if not by_season.empty:
        print("\nBy season:")
        print(by_season[["model", "group_value", "bets", "hit_rate", "profit", "roi"]].to_string(index=False))


def write_league_specific_strategy_report(
    config: ExperimentConfig,
    run_ts: str,
    validation_probs: Mapping[str, np.ndarray],
    validation_raw_odds: np.ndarray,
    validation_y: np.ndarray,
    validation_match_info: list[dict],
    test_probs: Mapping[str, np.ndarray],
    test_raw_odds: np.ndarray,
    test_y: np.ndarray,
    test_match_info: list[dict],
    *,
    edge_threshold: float,
    min_validation_bets: int = 20,
    min_fold_bets: int = 5,
    min_logloss_improvement_vs_market: float = 0.0,
) -> list[dict]:
    """Select a betting strategy per league on validation and report its test ROI.

    For each league it picks the model/strategy that bets well on validation (subject
    to minimum-bet, chronological-fold, and log-loss-vs-market gates), then evaluates
    that locked choice on the test set. Writes the league-selection and strategy CSVs.
    """
    candidate_models = [
        model
        for model in ["market_corr", "meta", "logreg", "mlp"]
        if model in validation_probs and model in test_probs
    ]
    selection_rows: list[dict] = []
    selected_test_bets: list[pd.DataFrame] = []
    leagues = sorted({str(row["league"]) for row in validation_match_info})

    val_leagues = np.array([str(row["league"]) for row in validation_match_info])
    test_leagues = np.array([str(row["league"]) for row in test_match_info])

    for league in leagues:
        val_mask = val_leagues == league
        test_mask = test_leagues == league
        if not np.any(val_mask) or not np.any(test_mask):
            continue

        val_info = [row for row, keep in zip(validation_match_info, val_mask) if keep]
        test_info = [row for row, keep in zip(test_match_info, test_mask) if keep]
        val_y_league = validation_y[val_mask]
        val_odds_league = validation_raw_odds[val_mask]
        test_y_league = test_y[test_mask]
        test_odds_league = test_raw_odds[test_mask]

        market_val_logloss = float(log_loss(val_y_league, validation_probs["market"][val_mask]))
        fold_masks = _chronological_fold_masks(val_info, n_folds=2)
        candidates = []
        for model in candidate_models:
            probs = validation_probs[model][val_mask]
            stats, _ = _model_betting_stats(probs, val_odds_league, val_y_league, val_info, edge_threshold)
            ll = float(log_loss(val_y_league, probs))
            fold_checks = []
            for fold_mask in fold_masks:
                fold_info = [row for row, keep in zip(val_info, fold_mask) if keep]
                fold_y = val_y_league[fold_mask]
                fold_odds = val_odds_league[fold_mask]
                fold_model_probs = probs[fold_mask]
                fold_market_probs = validation_probs["market"][val_mask][fold_mask]
                fold_stats, _ = _model_betting_stats(fold_model_probs, fold_odds, fold_y, fold_info, edge_threshold)
                fold_checks.append(
                    fold_stats["bets"] >= min_fold_bets
                    and fold_stats["roi"] > 0.0
                    and float(log_loss(fold_y, fold_model_probs)) <= float(log_loss(fold_y, fold_market_probs)) - min_logloss_improvement_vs_market
                )
            eligible = (
                stats["bets"] >= min_validation_bets
                and stats["roi"] > 0.0
                and ll <= market_val_logloss - min_logloss_improvement_vs_market
                and all(fold_checks)
            )
            candidates.append({
                "model": model,
                "val_logloss": ll,
                "val_bets": stats["bets"],
                "val_hit_rate": stats["hit_rate"],
                "val_profit": stats["profit"],
                "val_roi": stats["roi"],
                "eligible": eligible,
            })

        eligible_candidates = [row for row in candidates if row["eligible"]]
        if eligible_candidates:
            selected = sorted(eligible_candidates, key=lambda row: (-row["val_roi"], row["val_logloss"]))[0]
            reason = "positive_roi_and_logloss_in_all_validation_folds"
        else:
            selected = {
                "model": "market",
                "val_logloss": market_val_logloss,
                "val_bets": 0,
                "val_hit_rate": 0.0,
                "val_profit": 0.0,
                "val_roi": 0.0,
                "eligible": True,
            }
            reason = "fallback_to_market"

        test_stats, test_bets = _model_betting_stats(
            test_probs[selected["model"]][test_mask],
            test_odds_league,
            test_y_league,
            test_info,
            edge_threshold,
        )
        if not test_bets.empty:
            test_bets = test_bets.copy()
            test_bets["selected_model"] = selected["model"]
            selected_test_bets.append(test_bets)

        selection_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "league": league,
            "selected_model": selected["model"],
            "reason": reason,
            "market_val_logloss": round(market_val_logloss, 6),
            "selected_val_logloss": round(float(selected["val_logloss"]), 6),
            "selected_val_bets": int(selected["val_bets"]),
            "selected_val_hit_rate": round(float(selected["val_hit_rate"]), 4),
            "selected_val_profit": round(float(selected["val_profit"]), 6),
            "selected_val_roi": round(float(selected["val_roi"]), 4),
            "test_bets": int(test_stats["bets"]),
            "test_hit_rate": round(float(test_stats["hit_rate"]), 4),
            "test_avg_odds": round(float(test_stats["avg_odds"]), 4),
            "test_profit": round(float(test_stats["profit"]), 6),
            "test_roi": round(float(test_stats["roi"]), 4),
        })

    strategy_rows: list[dict] = []
    if selected_test_bets:
        all_bets = pd.concat(selected_test_bets, ignore_index=True)
        all_bets["date"] = pd.to_datetime(all_bets["date"])
        all_bets["league"] = all_bets["league"].fillna("unknown")
        all_bets["season"] = all_bets["date"].map(_season_label)
        all_bets["league_season"] = all_bets["league"].astype(str) + "|" + all_bets["season"].astype(str)
        strategy_rows.append(_metric_row(run_ts, config.experiment_name, "league_selector", "all", "all", all_bets))
        _append_group_rows(strategy_rows, run_ts, config.experiment_name, "league_selector", all_bets, "league", "league")
        _append_group_rows(strategy_rows, run_ts, config.experiment_name, "league_selector", all_bets, "season", "season")
        _append_group_rows(strategy_rows, run_ts, config.experiment_name, "league_selector", all_bets, "league_season", "league_season")
    else:
        strategy_rows.append(_metric_row(run_ts, config.experiment_name, "league_selector", "all", "all", pd.DataFrame()))

    append_rows_to_csv(config.final_league_model_selection_file, selection_rows)
    append_rows_to_csv(config.final_league_strategy_file, strategy_rows)
    if config.print_full_reports:
        _print_league_specific_strategy_summary(selection_rows, strategy_rows)
    return selection_rows


def _print_league_specific_strategy_summary(selection_rows: list[dict], strategy_rows: list[dict]):
    print("\n" + "=" * 70)
    print("=== LEAGUE-SPECIFIC BETTING STRATEGY ===")
    print("=" * 70)
    if not selection_rows:
        print("No league selections produced.")
        return
    selection_df = pd.DataFrame(selection_rows)
    print("\nValidation-selected models:")
    print(selection_df[["league", "selected_model", "reason", "selected_val_bets", "selected_val_roi", "test_bets", "test_profit", "test_roi"]].to_string(index=False))

    strategy_df = pd.DataFrame(strategy_rows)
    overall = strategy_df[strategy_df["group_type"] == "all"]
    if not overall.empty:
        print("\nFinal test performance:")
        print(overall[["model", "bets", "hit_rate", "avg_odds", "profit", "roi"]].to_string(index=False))
