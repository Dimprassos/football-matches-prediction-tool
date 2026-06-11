"""Experiment configuration and the canonical experiment presets.

Every run of the pipeline (training, backtests, the Streamlit app) is driven by an
:class:`ExperimentConfig`. The config decides the train/test split, which cached
artifacts may be reused, where artifacts are written (all paths are derived from
``experiment_name``), and the modelling knobs (odds source, market-movement
features, forced feature sets).

The pre-built configs at the bottom of the module are the ones the project ships
with:

* ``FINAL_CONFIG`` — the canonical, market-dominated experiment whose metrics the
  thesis reports (this is what the app serves by default).
* ``CONTEXT_AWARE_CONFIG`` — a feature-rich variant forced to use understat/form
  features, for comparison against the market-only model.
* ``DEFAULT_CONFIG`` / ``CLOSING_MARKET_CONFIG`` — older baseline setups.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable settings for a single experiment.

    Field groups:

    * **Data window** — ``train_cut``/``test_cut``/``test_end`` define the fixed,
      date-based split; ``leagues`` selects which leagues are loaded.
    * **Caching / force flags** — ``use_cached_artifacts`` plus the ``force_*``
      switches control whether the expensive Optuna tuning and model fits are
      reused or redone. ``allow_partial_param_cache`` lets per-league params be
      reused even when the rest of the cache is incompatible.
    * **Odds & features** — ``market_odds_source``/``betting_odds_source``
      ("opening" vs "closing"), ``include_market_movement_features``, and the
      ``*_feature_set`` overrides described below.
    * **Output paths** — every ``*_file`` property is derived from
      ``experiment_name`` so distinct experiments never overwrite each other.
    """

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
    # When set, forces the learned models to use a named feature set instead of
    # auto-selecting the best-scoring subset. Used to build feature-rich variants
    # (e.g. understat/form-aware) whose features the user can actually exercise.
    meta_feature_set: str | None = None
    mlp_feature_set: str | None = None

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

# Feature-rich variant: same data/odds setup as the canonical experiment, but the
# learned models are forced to use understat-xG-aware feature sets. This makes the
# reconstructed pre-match features (see src/predictor.build_runtime_extra_features)
# actually influence the prediction, so the user can compare a market-only model
# against a feature-rich one. Trained via scripts/train_context_variant.py.
CONTEXT_AWARE_CONFIG = ExperimentConfig(
    experiment_name="context_aware_understat_xg",
    market_odds_source="opening",
    betting_odds_source="opening",
    include_market_movement_features=False,
    use_cached_artifacts=True,
    allow_partial_param_cache=True,
    force_retune_meta=True,
    force_refit_meta_model=True,
    force_retune_mlp=True,
    force_refit_mlp_model=True,
    force_retune_blend=True,
    generate_upcoming_picks=False,
    meta_feature_set="market_plus_understat_xg",
    mlp_feature_set="default_plus_understat_xg",
)


# Player-aware variant: same setup as the context-aware one, but its forced feature
# sets add the lineup-strength signal (home/away/diff) built from the Understat starters.
# Lets the served prediction respond to the chosen XI for an honest ablation vs market.
PLAYER_CONTEXT_CONFIG = ExperimentConfig(
    experiment_name="player_context_aware",
    market_odds_source="opening",
    betting_odds_source="opening",
    include_market_movement_features=False,
    use_cached_artifacts=True,
    allow_partial_param_cache=True,
    force_retune_meta=True,
    force_refit_meta_model=True,
    force_retune_mlp=True,
    force_refit_mlp_model=True,
    force_retune_blend=True,
    generate_upcoming_picks=False,
    meta_feature_set="market_plus_lineup",
    mlp_feature_set="default_plus_lineup",
)


# Deep-learning variant: the recurrent FootyNet (src/models/footynet.py), trained by
# its own runner (scripts/train_footynet.py) rather than run_training_pipeline. Shares
# the canonical opening-odds / market-movement-off setup and seeds the base
# Elo/Poisson params from FINAL_CONFIG, so its static branch matches the tabular
# models' inputs exactly. Only the eval-report path/name is taken from here.
FOOTYNET_CONFIG = ExperimentConfig(
    experiment_name="footynet_deep",
    market_odds_source="opening",
    betting_odds_source="opening",
    include_market_movement_features=False,
    allow_partial_param_cache=True,
    generate_upcoming_picks=False,
)
