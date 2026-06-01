from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "baseline_xgboost_v3_formpoints"
    artifacts_dir: Path = Path("artifacts")
    train_cut: str = "2024-07-01"
    test_cut: str = "2025-07-01"
    test_end: str | None = None
    leagues: tuple[str, ...] = ("england", "spain", "italy", "germany", "france")
    use_cached_artifacts: bool = True
    force_retune_leagues: bool = False
    force_retune_meta: bool = False
    force_refit_meta_model: bool = True
    force_retune_mlp: bool = False
    force_refit_mlp_model: bool = True
    force_retune_blend: bool = True
    allow_partial_param_cache: bool = False
    random_state: int = 42
    max_upcoming_window_days: int = 4
    detailed_betting_log: bool = False
    print_full_reports: bool = False
    print_verbose_audits: bool = False
    print_parameter_impact: bool = False
    generate_upcoming_picks: bool = True
    market_odds_source: str = "closing"
    betting_odds_source: str = "closing"
    include_market_movement_features: bool = True

    @property
    def params_file(self) -> Path:
        return self.artifacts_dir / f"best_params_{self.experiment_name}.json"

    @property
    def meta_file(self) -> Path:
        return self.artifacts_dir / f"best_meta_{self.experiment_name}.json"

    @property
    def model_file(self) -> Path:
        return self.artifacts_dir / f"meta_model_{self.experiment_name}.json"

    @property
    def mlp_meta_file(self) -> Path:
        return self.artifacts_dir / f"best_mlp_{self.experiment_name}.json"

    @property
    def mlp_model_file(self) -> Path:
        return self.artifacts_dir / f"mlp_model_{self.experiment_name}.pkl"

    @property
    def logreg_meta_file(self) -> Path:
        return self.artifacts_dir / f"best_logreg_{self.experiment_name}.json"

    @property
    def logreg_model_file(self) -> Path:
        return self.artifacts_dir / f"logreg_model_{self.experiment_name}.pkl"

    @property
    def blend_file(self) -> Path:
        return self.artifacts_dir / f"best_blend_{self.experiment_name}.json"

    @property
    def manifest_file(self) -> Path:
        return self.artifacts_dir / f"manifest_{self.experiment_name}.json"

    @property
    def results_csv_file(self) -> Path:
        return self.artifacts_dir / "experiment_results.csv"

    @property
    def ablations_csv_file(self) -> Path:
        return self.artifacts_dir / "feature_ablations.csv"

    @property
    def final_model_summary_file(self) -> Path:
        return self.artifacts_dir / f"final_model_summary_{self.experiment_name}.csv"

    @property
    def final_ablation_summary_file(self) -> Path:
        return self.artifacts_dir / f"final_ablation_summary_{self.experiment_name}.csv"

    @property
    def final_betting_robustness_file(self) -> Path:
        return self.artifacts_dir / f"final_betting_robustness_{self.experiment_name}.csv"

    @property
    def final_bet_curve_file(self) -> Path:
        return self.artifacts_dir / f"final_bet_curve_{self.experiment_name}.csv"

    @property
    def final_league_model_selection_file(self) -> Path:
        return self.artifacts_dir / f"final_league_model_selection_{self.experiment_name}.csv"

    @property
    def final_league_strategy_file(self) -> Path:
        return self.artifacts_dir / f"final_league_strategy_{self.experiment_name}.csv"

    @property
    def final_probability_quality_file(self) -> Path:
        return self.artifacts_dir / f"final_probability_quality_{self.experiment_name}.csv"

    @property
    def final_bet_selector_file(self) -> Path:
        return self.artifacts_dir / f"final_bet_selector_{self.experiment_name}.csv"

    @property
    def final_bet_bucket_file(self) -> Path:
        return self.artifacts_dir / f"final_bet_buckets_{self.experiment_name}.csv"

    @property
    def final_alternative_markets_file(self) -> Path:
        return self.artifacts_dir / f"final_alternative_markets_{self.experiment_name}.csv"

    @property
    def final_data_enrichment_file(self) -> Path:
        return self.artifacts_dir / f"final_data_enrichment_audit_{self.experiment_name}.csv"

    def as_manifest(self) -> dict:
        data = asdict(self)
        data["artifacts_dir"] = str(self.artifacts_dir)
        data["leagues"] = list(self.leagues)
        return data


DEFAULT_CONFIG = ExperimentConfig()
FINAL_CONFIG = ExperimentConfig(
    experiment_name="final_opening_market_pre_match",
    market_odds_source="opening",
    betting_odds_source="opening",
    include_market_movement_features=False,
    allow_partial_param_cache=True,
)
CLOSING_MARKET_CONFIG = ExperimentConfig(experiment_name="final_closing_market_xg_comparison")
