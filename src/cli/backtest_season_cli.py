"""CLI for running a leakage-safe single-season betting backtest.

Builds an :class:`ExperimentConfig` whose train/validation/test cuts isolate one
target season (with the prior season as validation) and runs the standard pipeline
on it, so each historical season can be evaluated as if predicted in real time.
"""
from __future__ import annotations

import argparse

from src.config import ExperimentConfig
from src.trainer import run_training_pipeline


def season_window(season_start: int) -> tuple[str, str, str]:
    """Return (train_cut, test_cut, test_end) ISO dates isolating one season for backtesting."""
    if season_start < 2014:
        raise ValueError("season must be 2014 or later so the pipeline has a full prior validation season")
    return (
        f"{season_start - 1}-07-01",
        f"{season_start}-07-01",
        f"{season_start + 1}-07-01",
    )


def build_backtest_config(
    season_start: int,
    *,
    force_refit: bool = False,
    force_retune: bool = False,
    full_report: bool = False,
    market_odds_source: str = "closing",
    betting_odds_source: str = "closing",
    include_market_movement_features: bool | None = None,
) -> ExperimentConfig:
    train_cut, test_cut, test_end = season_window(season_start)
    if include_market_movement_features is None:
        include_market_movement_features = market_odds_source == "closing"
    experiment_name = f"season_backtest_{season_start}_{season_start + 1}"
    if (
        market_odds_source != "closing"
        or betting_odds_source != "closing"
        or not include_market_movement_features
    ):
        experiment_name = (
            f"{experiment_name}_{market_odds_source}_market_"
            f"{betting_odds_source}_price"
        )
        if not include_market_movement_features:
            experiment_name = f"{experiment_name}_no_move"
    return ExperimentConfig(
        experiment_name=experiment_name,
        train_cut=train_cut,
        test_cut=test_cut,
        test_end=test_end,
        force_retune_leagues=force_retune,
        force_retune_meta=force_retune,
        force_refit_meta_model=force_refit or force_retune,
        force_retune_mlp=force_retune,
        force_refit_mlp_model=force_refit or force_retune,
        force_retune_blend=force_refit or force_retune,
        allow_partial_param_cache=True,
        generate_upcoming_picks=False,
        print_full_reports=full_report,
        print_parameter_impact=True,
        market_odds_source=market_odds_source,
        betting_odds_source=betting_odds_source,
        include_market_movement_features=include_market_movement_features,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a leakage-safe single-season betting backtest.",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season start year. Example: --season 2024 evaluates 2024-2025.",
    )
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--force-refit",
        action="store_true",
        help="Reuse cached season-specific tuning when available, but refit final models and retune the blend.",
    )
    refresh_group.add_argument(
        "--force-retune",
        action="store_true",
        help="Retune league, XGBoost, MLP, and blend settings for this season-specific backtest.",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Print the old detailed diagnostic tables. By default, detailed tables are written only to artifacts.",
    )
    parser.add_argument(
        "--market-odds",
        choices=["opening", "closing", "legacy"],
        default="closing",
        help="Odds snapshot used to build market-implied probabilities/features.",
    )
    parser.add_argument(
        "--betting-odds",
        choices=["opening", "closing", "legacy"],
        default="closing",
        help="Odds snapshot used as the simulated bet price.",
    )
    movement_group = parser.add_mutually_exclusive_group()
    movement_group.add_argument(
        "--include-market-movement",
        action="store_true",
        default=None,
        help="Use closing minus opening market movement as model features.",
    )
    movement_group.add_argument(
        "--no-market-movement",
        action="store_false",
        dest="include_market_movement",
        help="Zero out closing-movement features for a cleaner pre-match setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_backtest_config(
        args.season,
        force_refit=args.force_refit,
        force_retune=args.force_retune,
        full_report=args.full_report,
        market_odds_source=args.market_odds,
        betting_odds_source=args.betting_odds,
        include_market_movement_features=args.include_market_movement,
    )
    print("=== LEAKAGE-SAFE SEASON BACKTEST ===")
    print(f"Target season: {args.season}-{args.season + 1}")
    print(f"Fit history: dates before {config.train_cut}")
    print(f"Validation/meta window: [{config.train_cut}, {config.test_cut})")
    print(f"Test-only betting window: [{config.test_cut}, {config.test_end})")
    print(f"Market odds snapshot: {config.market_odds_source}")
    print(f"Betting price snapshot: {config.betting_odds_source}")
    print(f"Market movement features: {config.include_market_movement_features}")
    print("Match-state and rolling features are recomputed chronologically using only earlier dates.")
    run_training_pipeline(config)
    print("\nBacktest artifacts:")
    print(f"  Betting robustness: {config.final_betting_robustness_file}")
    print(f"  Bet selector: {config.final_bet_selector_file}")
    print(f"  League strategy: {config.final_league_strategy_file}")


if __name__ == "__main__":
    main()
