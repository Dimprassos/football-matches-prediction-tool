"""End-to-end training/evaluation pipeline orchestration.

:func:`run_training_pipeline` is the heart of the project. For each league it:

1. splits the data on the config's fixed date cuts (train / late-validation / test);
2. tunes or reuses the base (Elo + Dixon-Coles) parameters;
3. builds leakage-safe streaming features, then tunes/fits the learned meta-models
   (XGBoost, MLP, Logistic Regression), each selecting its best feature subset;
4. tunes the ensemble blend weights and the market-logit correction;
5. evaluates everything on the held-out test set with many metrics plus a realistic
   betting simulation, and writes the artifacts + reports.

Artifact caching is governed by :data:`PIPELINE_VERSION` and a data/config
fingerprint (see :func:`_cached_artifacts_are_compatible`): expensive tuning is
reused whenever it is provably still valid, and each experiment writes to its own
files so runs never overwrite each other. The bulk of this module is the reporting
helpers that produce the human-readable summaries and CSVs.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.artifact_store import append_rows_to_csv, load_json_if_exists, load_pickle_if_exists, save_json, save_manifest, save_pickle
from src.bet_selection import (
    write_alternative_market_report,
    write_data_enrichment_audit,
    write_probability_quality_report,
    write_validation_selected_betting_reports,
)
from src.betting_robustness import write_betting_robustness_report, write_league_specific_strategy_report
from src.calibration import temperature_scale_probs
from src.config import DEFAULT_CONFIG, ExperimentConfig
from src.data_loader import load_league_data
from src.evaluation import (
    simulate_value_betting,
    print_alignment_audit,
    print_strategy_comparison,
    print_market_dependency_audit,
    print_profit_profile_audit,
)
from src.feature_builder import (
    BASE_NON_MARKET_FEATURE_COLUMNS,
    EXTERNAL_CONTEXT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LOCAL_STATS_FEATURE_COLUMNS,
    MARKET_CONTEXT_FEATURE_COLUMNS,
    MLP_DEFAULT_COLS,
    MLP_DEFAULT_FEATURE_COLUMNS,
    MLP_NO_REST_COLS,
    NEW_DATA_FEATURE_COLUMNS,
    NEW_LOCAL_FEATURE_COLUMNS,
    ODDS_MOVEMENT_FEATURE_COLUMNS,
    TEAM_NEWS_FEATURE_COLUMNS,
    UNDERSTAT_XG_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    build_meta_features,
    ensure_market_probs,
    feature_indices,
    time_split_val,
    market_probs_from_odds_row,
)
from src.models.base import tune_league_params
from src.models.meta import (
    apply_market_logit_correction,
    blend_probabilities,
    fit_xgb_model,
    make_logreg_pipeline,
    make_mlp_pipeline,
    probs_from_meta_features,
    tune_feature_subset,
    tune_blend_weights,
    tune_logreg_hyperparams,
    tune_market_logit_correction,
    tune_mlp_feature_subset,
    tune_mlp_hyperparams,
    tune_xgb_hyperparams,
)
from src.reporting import print_per_league_test_metrics
from src.reporting_ext import print_confusion, print_prob_report
from src.services.upcoming import generate_upcoming_matchday_picks
from src.state_builder import streaming_block_probs_home_away
from src.external_context import EXTERNAL_CONTEXT_VALUE_COLUMNS, MATCH_CONTEXT_FILE
from src.metrics import class_metric_summary


# Bump this whenever the artifact format or feature layout changes; cached
# artifacts from an older version are treated as incompatible and recomputed.
PIPELINE_VERSION = 13

PARAMETER_IMPACT_BASELINE = {
    "K": 50,
    "ha": 80,
    "beta": 0.12,
    "decay": 0.001,
    "rho": 0.0,
    "T": 1.0,
}


MATCH_INFO_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "odds_home",
    "odds_draw",
    "odds_away",
    "open_odds_home",
    "open_odds_draw",
    "open_odds_away",
    "close_odds_home",
    "close_odds_draw",
    "close_odds_away",
    "ou25_over_prob",
    "ou25_over_odds",
    "ou25_under_odds",
    "ah_line",
    *EXTERNAL_CONTEXT_VALUE_COLUMNS,
]


def _match_info_records(df, league: str) -> list[dict]:
    cols = [col for col in MATCH_INFO_COLUMNS if col in df.columns]
    records = df[cols].to_dict("records")
    for row in records:
        row["league"] = league
    return records


def _chronological_fold_masks(match_info: list[dict], n_folds: int = 2) -> list[np.ndarray]:
    if n_folds <= 1 or len(match_info) < n_folds:
        return []
    order = np.array(sorted(range(len(match_info)), key=lambda i: str(match_info[i].get("date", ""))))
    masks = []
    for fold_indices in np.array_split(order, n_folds):
        mask = np.zeros(len(match_info), dtype=bool)
        mask[fold_indices] = True
        masks.append(mask)
    return masks


def _split_played_periods(df, config: ExperimentConfig):
    train_fit = df[df["date"] < config.train_cut].copy()
    val = df[(df["date"] >= config.train_cut) & (df["date"] < config.test_cut)].copy()
    if config.test_end is None:
        test = df[df["date"] >= config.test_cut].copy()
    else:
        test = df[(df["date"] >= config.test_cut) & (df["date"] < config.test_end)].copy()
    return train_fit, val, test


def _file_fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "exists": True,
        "size": int(stat.st_size),
        "sha1": digest.hexdigest(),
    }


def _data_fingerprint() -> dict:
    return {
        "match_context": _file_fingerprint(MATCH_CONTEXT_FILE),
    }


def _cached_artifacts_are_compatible(config: ExperimentConfig) -> bool:
    """True if the saved artifacts can be safely reused for this config.

    Reuse is allowed only when the manifest exists, was produced by the current
    :data:`PIPELINE_VERSION`, matches the experiment's config fields (split cuts,
    odds setup, etc.), and the input-data fingerprint is unchanged. Any mismatch
    forces a retune/refit so stale numbers never leak into a run.
    """
    manifest = load_json_if_exists(config.manifest_file)
    if manifest is None:
        print("Cached artifact manifest missing; retuning/refitting artifacts.")
        return False
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        print("Cached artifacts were produced by an older pipeline; retuning/refitting artifacts.")
        return False

    manifest_config = manifest.get("config", {})
    expected_config = config.as_manifest()
    for key in (
        "experiment_name",
        "train_cut",
        "test_cut",
        "leagues",
        "market_odds_source",
        "betting_odds_source",
        "include_market_movement_features",
    ):
        if manifest_config.get(key) != expected_config.get(key):
            print(f"Cached artifact config mismatch for {key}; retuning/refitting artifacts.")
            return False

    if manifest.get("feature_columns") != FEATURE_COLUMNS:
        print("Cached artifact feature schema mismatch; retuning/refitting artifacts.")
        return False
    if manifest.get("data_fingerprint") != _data_fingerprint():
        print("Cached artifact data fingerprint mismatch; retuning/refitting artifacts.")
        return False
    mlp_feature_columns = manifest.get("mlp_feature_columns")
    if not mlp_feature_columns or any(col not in FEATURE_COLUMNS for col in mlp_feature_columns):
        print("Cached artifact MLP feature schema invalid; retuning/refitting artifacts.")
        return False
    return True


def _model_summary_row(name: str, probs: np.ndarray, y_true: np.ndarray, raw_odds: np.ndarray, *, edge_threshold: float):
    bets, wins, profit, roi, avg_odds = simulate_value_betting(
        probs,
        raw_odds,
        y_true,
        edge_threshold=edge_threshold,
        match_info=None,
        verbose=False,
    )
    accuracy = float((np.argmax(probs, axis=1) == y_true).mean())
    hit_rate = (wins / bets * 100.0) if bets > 0 else 0.0
    class_metrics = class_metric_summary(probs, y_true)
    return {
        "name": name,
        "logloss": float(log_loss(y_true, probs)),
        "accuracy": accuracy,
        **class_metrics,
        "bets": int(bets),
        "hit_rate": float(hit_rate),
        "roi": float(roi),
        "profit": float(profit),
        "avg_odds": float(avg_odds),
    }


def write_class_marginal_calibration(config, run_ts, probs_by_split, y_by_split):
    """Diagnose per-class marginal calibration: how close is each model's *average*
    predicted probability for an outcome to how often that outcome actually happens.

    This separates two different explanations for the near-zero draw recall:
      - mean draw prob ~= actual draw rate -> draws are priced correctly on average and
        the low draw *recall* is an argmax artifact (a draw is rarely the single most
        likely outcome), not a probability defect to fix.
      - mean draw prob << actual draw rate -> the model systematically under-predicts
        draws and there is a real probability gap worth closing.
    """
    class_names = ("home", "draw", "away")
    rows = []
    for split, probs_by_model in probs_by_split.items():
        y = np.asarray(y_by_split[split])
        actual = {c: float((y == i).mean()) for i, c in enumerate(class_names)}
        for model, probs in probs_by_model.items():
            probs = np.asarray(probs)
            pred = np.argmax(probs, axis=1)
            row = {
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "split": split,
                "model": model,
                "n": int(len(y)),
            }
            for i, c in enumerate(class_names):
                row[f"mean_prob_{c}"] = round(float(probs[:, i].mean()), 6)
                row[f"actual_rate_{c}"] = round(actual[c], 6)
                row[f"pick_rate_{c}"] = round(float((pred == i).mean()), 6)
            rows.append(row)
    out_file = config.artifacts_dir / f"final_class_calibration_{config.experiment_name}.csv"
    append_rows_to_csv(out_file, rows)
    print("\nPer-class marginal calibration (draw focus: mean predicted prob vs actual rate):")
    print(f"  {'split':<11}{'model':<9}{'mean_p_draw':>12}{'actual':>8}{'pick':>7}")
    for row in rows:
        print(
            f"  {row['split']:<11}{row['model']:<9}"
            f"{row['mean_prob_draw']:>12.3f}{row['actual_rate_draw']:>8.3f}{row['pick_rate_draw']:>7.3f}"
        )
    return rows


def write_per_match_predictions(config, run_ts, probs_by_model, y_true, raw_odds, match_info):
    """Dump per-match test predictions so confidence intervals can be computed offline
    from the exact probabilities and odds used in the pipeline -- no re-derivation, no
    drift. The file is overwritten each run (one row per test match, latest run only)."""
    import csv as _csv

    y_true = np.asarray(y_true)
    raw_odds = np.asarray(raw_odds)
    model_names = list(probs_by_model)
    rows = []
    for i, info in enumerate(match_info):
        row = {
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "league": info.get("league", ""),
            "date": str(info.get("date", "")),
            "home_team": info.get("home_team", ""),
            "away_team": info.get("away_team", ""),
            "y_true": int(y_true[i]),
            "odds_h": float(raw_odds[i, 0]),
            "odds_d": float(raw_odds[i, 1]),
            "odds_a": float(raw_odds[i, 2]),
        }
        # Carry opening/closing odds through so closing-line value can be computed offline.
        for key in ("open_odds_home", "open_odds_draw", "open_odds_away",
                    "close_odds_home", "close_odds_draw", "close_odds_away"):
            try:
                value = float(info.get(key))
            except (TypeError, ValueError):
                value = float("nan")
            row[key] = value if np.isfinite(value) else ""
        for model in model_names:
            p = np.asarray(probs_by_model[model])[i]
            row[f"{model}_p_h"] = round(float(p[0]), 6)
            row[f"{model}_p_d"] = round(float(p[1]), 6)
            row[f"{model}_p_a"] = round(float(p[2]), 6)
        rows.append(row)
    if not rows:
        return rows
    out_file = config.artifacts_dir / f"final_per_match_predictions_{config.experiment_name}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote per-match predictions: {out_file}")
    return rows


def _validate_odds_source_config(config: ExperimentConfig) -> None:
    valid_sources = {"opening", "closing", "legacy"}
    if config.market_odds_source not in valid_sources:
        raise ValueError(f"market_odds_source must be one of {sorted(valid_sources)}, got {config.market_odds_source!r}")
    if config.betting_odds_source not in valid_sources:
        raise ValueError(f"betting_odds_source must be one of {sorted(valid_sources)}, got {config.betting_odds_source!r}")
    if config.market_odds_source == "opening" and config.include_market_movement_features:
        print(
            "WARNING: opening-market experiment includes closing movement features. "
            "Disable include_market_movement_features for a clean pre-match/opening-odds setup."
        )


def _league_param_text(params: dict) -> str:
    return (
        f"K={params.get('K')}, ha={params.get('ha')}, beta={params.get('beta')}, "
        f"decay={params.get('decay')}, rho={params.get('rho')}, T={params.get('T')}"
    )


def _fmt_int_param(value) -> str:
    if value is None:
        return "n/a"
    return f"{int(value)}"


def _fmt_float_param(value, *, decimals: int = 4, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}f}"


def _print_league_run_header(league: str) -> None:
    print("\n" + "-" * 64)
    print(f"LEAGUE  {league.upper()}")
    print("-" * 64)


def _print_split_counts(train_fit, val, test) -> None:
    print(f"Rows    train_fit={len(train_fit):,} | validation={len(val):,} | test={len(test):,}")


def _print_league_params(params: dict, *, source: str) -> None:
    print(f"Params  {source}")
    print(
        f"        Elo:   K={_fmt_int_param(params.get('K')):<3} | "
        f"home_adv={_fmt_int_param(params.get('ha')):<3}"
    )
    print(
        f"        Goals: beta={_fmt_float_param(params.get('beta'))} | "
        f"decay={_fmt_float_param(params.get('decay'))} | "
        f"rho={_fmt_float_param(params.get('rho'), signed=True)}"
    )
    print(f"        Calib: T={_fmt_float_param(params.get('T'))}")


def _parameter_impact_row(
    league: str,
    y_true: np.ndarray,
    tuned_probs: np.ndarray,
    baseline_probs: np.ndarray,
    params: dict | None,
) -> dict:
    tuned_ll = float(log_loss(y_true, tuned_probs, labels=[0, 1, 2]))
    baseline_ll = float(log_loss(y_true, baseline_probs, labels=[0, 1, 2]))
    return {
        "league": league,
        "matches": int(len(y_true)),
        "tuned_logloss": tuned_ll,
        "baseline_logloss": baseline_ll,
        "delta_logloss": tuned_ll - baseline_ll,
        "avg_abs_prob_diff": float(np.mean(np.abs(tuned_probs - baseline_probs))),
        "params": None if params is None else {key: params.get(key) for key in PARAMETER_IMPACT_BASELINE},
    }


def _effective_blend_weights(blend_cfg: dict) -> dict:
    raw_weights = blend_cfg.get("weights", {})
    active_names = ["base", "market", "xgb"]
    if blend_cfg.get("mlp_allowed", True):
        active_names.append("mlp")

    total = sum(float(raw_weights.get(name, 0.0)) for name in active_names)
    if total <= 0:
        return {name: 1.0 if name == "xgb" else 0.0 for name in ["base", "market", "xgb", "mlp"]}
    return {
        name: float(raw_weights.get(name, 0.0)) / total if name in active_names else 0.0
        for name in ["base", "market", "xgb", "mlp"]
    }


def _print_blend_weight_summary(blend_cfg: dict) -> None:
    effective = _effective_blend_weights(blend_cfg)
    print("Effective blend weights:", {name: round(weight, 4) for name, weight in effective.items()})


def _print_parameter_impact_summary(rows: list[dict]) -> None:
    if not rows:
        return

    print("\nLeague parameter impact on base model:")
    print(f"{'league':<10} {'matches':>7} {'tuned_ll':>9} {'baseline_ll':>11} {'delta':>9} {'avg_prob_diff':>13}")
    for row in rows:
        print(
            f"{row['league']:<10} {row['matches']:>7d} "
            f"{row['tuned_logloss']:>9.4f} {row['baseline_logloss']:>11.4f} "
            f"{row['delta_logloss']:>+9.4f} {row['avg_abs_prob_diff']:>13.4f}"
        )
        if row["params"] is not None and row["league"] != "ALL":
            print(f"  params: {_league_param_text(row['params'])}")

    baseline = _league_param_text(PARAMETER_IMPACT_BASELINE)
    print(f"  baseline: {baseline}")


def _no_bet_row() -> dict:
    return {
        "name": "no_bet",
        "logloss": 0.0,
        "accuracy": 0.0,
        "bets": 0,
        "hit_rate": 0.0,
        "roi": 0.0,
        "profit": 0.0,
        "avg_odds": 0.0,
    }


def _betting_roi_candidates(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if int(row.get("bets", 0)) > 0 and np.isfinite(float(row.get("roi", np.nan)))
    ]


def _best_betting_roi_row(rows: list[dict]) -> dict:
    candidates = _betting_roi_candidates(rows)
    if not candidates:
        return _no_bet_row()
    return max(candidates, key=lambda r: r["roi"])


def _print_model_selection_summary(rows: list[dict], *, full_report: bool = True):
    best_logloss = min(rows, key=lambda r: r["logloss"])
    best_roi = _best_betting_roi_row(rows)

    if full_report:
        print("\n" + "=" * 70)
        print("=== MODEL SELECTION SUMMARY ===")
        print("=" * 70)

        print(
            f"{'Model':<12} | {'LogLoss':>8} | {'Acc%':>6} | {'Bets':>5} | {'Hit%':>6} | {'ROI%':>7} | {'Profit':>8}"
        )
        print("-" * 70)
        for row in sorted(rows, key=lambda r: (r["logloss"], -r["roi"])):
            print(
                f"{row['name']:<12} | {row['logloss']:>8.4f} | {row['accuracy'] * 100:>6.2f} | {row['bets']:>5} | "
                f"{row['hit_rate']:>6.2f} | {row['roi']:>7.2f} | {row['profit']:>8.3f}"
            )

        print("\nWinner by LogLoss:", best_logloss["name"], f"({best_logloss['logloss']:.4f})")
        if best_roi["name"] == "no_bet":
            print("Best Betting ROI among models with bets: none (no bets placed)")
        else:
            print("Best Betting ROI among models with bets:", best_roi["name"], f"({best_roi['roi']:.2f}%)")
    return best_logloss, best_roi


def _select_recommended_betting_model(validation_rows: list[dict], test_rows: list[dict]):
    """Pick the betting model using validation ROI only, then report its test row.

    The decision (which model, and whether to bet at all) is locked on the validation
    split so the crowned winner is not chosen by peeking at the test set -- with several
    models, the model with the best *test* ROI is upward biased by selection alone. The
    returned row is the selected model's *test* row, so the reported ROI/profit stay
    honest out-of-sample numbers; they are just no longer used to make the choice.
    """
    eligible_names = {"market", "market_corr", "logreg", "meta", "ensemble"}
    test_by_name = {row["name"]: row for row in test_rows}

    val_eligible = [
        row
        for row in _betting_roi_candidates(validation_rows)
        if row["name"] in eligible_names and np.isfinite(row["logloss"])
    ]
    if not val_eligible:
        return _no_bet_row(), "no_eligible_model_with_validation_bets"

    val_best = max(val_eligible, key=lambda r: r["roi"])
    if val_best["roi"] <= 0:
        return _no_bet_row(), "no_positive_validation_roi"

    val_probability_winner = min(validation_rows, key=lambda r: r["logloss"])
    logloss_gap = val_best["logloss"] - val_probability_winner["logloss"]
    if logloss_gap > 0.01:
        return _no_bet_row(), "validation_roi_candidate_failed_logloss_guard"

    test_row = test_by_name.get(val_best["name"])
    if test_row is None or int(test_row.get("bets", 0)) <= 0:
        return _no_bet_row(), "validation_selected_model_made_no_test_bets"
    return test_row, "validation_locked_positive_roi"


def _write_final_summary(config: ExperimentConfig, selection_rows: list[dict], ablation_rows: list[dict], run_ts: str):
    final_rows = []
    for row in selection_rows:
        final_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "model": row["name"],
            "logloss": round(row["logloss"], 6),
            "accuracy": round(row["accuracy"], 6),
            "macro_f1": round(row["macro_f1"], 6),
            "home_recall": round(row["home_recall"], 6),
            "draw_recall": round(row["draw_recall"], 6),
            "away_recall": round(row["away_recall"], 6),
            "home_precision": round(row["home_precision"], 6),
            "draw_precision": round(row["draw_precision"], 6),
            "away_precision": round(row["away_precision"], 6),
            "bets": row["bets"],
            "hit_rate": round(row["hit_rate"], 4),
            "roi": round(row["roi"], 4),
            "profit": round(row["profit"], 4),
            "avg_odds": round(row["avg_odds"], 4),
        })
    append_rows_to_csv(config.final_model_summary_file, final_rows)

    thesis_feature_sets = {
        "market_only",
        "market_plus_context",
        "market_plus_understat_xg",
        "market_context_plus_understat_xg",
        "market_plus_external_context",
        "market_context_plus_external_context",
        "core_18",
        "no_market",
        "no_understat_xg",
        "no_external_context",
    }
    final_ablation_rows = []
    for row in ablation_rows:
        if row["feature_set"] in thesis_feature_sets:
            final_ablation_rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "model": row["model"],
                "feature_set": row["feature_set"],
                "logloss": round(row["logloss"], 6),
                "accuracy": round(row["accuracy"], 6),
                "macro_f1": round(row["macro_f1"], 6),
                "home_recall": round(row["home_recall"], 6),
                "draw_recall": round(row["draw_recall"], 6),
                "away_recall": round(row["away_recall"], 6),
            })
    append_rows_to_csv(config.final_ablation_summary_file, final_ablation_rows)


def _print_feature_ablation_summary(rows: list[dict]):
    print("\n" + "=" * 70)
    print("=== FEATURE GROUP ABLATION ===")
    print("=" * 70)
    print(f"{'Model':<8} | {'Feature Set':<18} | {'LogLoss':>8} | {'Accuracy%':>9}")
    print("-" * 70)
    for row in sorted(rows, key=lambda r: (r["model"], r["logloss"])):
        print(
            f"{row['model']:<8} | {row['feature_set']:<18} | {row['logloss']:>8.4f} | {row['accuracy'] * 100:>9.2f}"
        )


def _coverage_row(rows: list[dict], split: str, data_group: str) -> dict | None:
    for row in rows:
        if row.get("split") == split and row.get("data_group") == data_group:
            return row
    return None


def _coverage_text(rows: list[dict], split: str, data_group: str) -> str:
    row = _coverage_row(rows, split, data_group)
    if row is None:
        return "n/a"
    return f"{int(row['available_rows'])}/{int(row['matches'])} ({float(row['coverage']):.1f}%)"


def _coverage_warning(rows: list[dict], split: str, data_group: str, *, min_pct: float) -> str:
    row = _coverage_row(rows, split, data_group)
    if row is None:
        return "missing"
    coverage = float(row["coverage"])
    if coverage <= 0:
        return "missing"
    if coverage < min_pct:
        return "too sparse"
    return "usable"


def _print_compact_model_table(rows: list[dict]) -> None:
    print("\nModel quality on test set:")
    print(f"{'model':<10} {'logloss':>8} {'vs_market':>10} {'acc%':>7} {'bets':>6} {'roi%':>8}")
    market = next((row for row in rows if row["name"] == "market"), min(rows, key=lambda r: r["logloss"]))
    for row in sorted(rows, key=lambda item: item["logloss"]):
        delta = row["logloss"] - market["logloss"]
        print(
            f"{row['name']:<10} {row['logloss']:>8.4f} {delta:>+10.4f} "
            f"{row['accuracy'] * 100:>7.2f} {row['bets']:>6d} {row['roi']:>8.2f}"
        )


def _best_ablation(rows: list[dict], model: str) -> dict | None:
    candidates = [row for row in rows if row["model"] == model]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["logloss"])


def _feature_set_row(rows: list[dict], model: str, feature_set: str) -> dict | None:
    for row in rows:
        if row["model"] == model and row["feature_set"] == feature_set:
            return row
    return None


def _print_external_context_note(ablation_rows: list[dict], data_audit_rows: list[dict]) -> None:
    print("\nExternal data status:")
    for data_group in ("injuries", "confirmed_lineups", "weather"):
        print(
            f"  {data_group:<17} validation {_coverage_text(data_audit_rows, 'validation', data_group):<18} "
            f"test {_coverage_text(data_audit_rows, 'test', data_group):<18} "
            f"status={_coverage_warning(data_audit_rows, 'test', data_group, min_pct=20.0)}"
        )

    market_only = _feature_set_row(ablation_rows, "mlp", "market_only")
    market_external = _feature_set_row(ablation_rows, "mlp", "market_plus_external_context")
    if market_only and market_external:
        delta = market_external["logloss"] - market_only["logloss"]
        verdict = "better" if delta < -0.001 else "worse/noisy" if delta > 0.001 else "neutral"
        print(
            f"  MLP external-context ablation: {market_external['logloss']:.4f} vs "
            f"market-only {market_only['logloss']:.4f} ({delta:+.4f}, {verdict})"
        )


def _print_concise_run_summary(
    *,
    config: ExperimentConfig,
    selection_rows: list[dict],
    ablation_rows: list[dict],
    data_audit_rows: list[dict],
    selector_rows: list[dict],
    parameter_impact_rows: list[dict],
    blend_cfg: dict,
    best_logloss: dict,
    best_roi: dict,
    recommended_betting: dict,
    betting_reason: str,
) -> None:
    print("\n" + "=" * 70)
    print("=== RUN SUMMARY ===")
    print("=" * 70)

    _print_compact_model_table(selection_rows)

    print("\nDecision:")
    print(f"  Best probability model: {best_logloss['name']} (LogLoss {best_logloss['logloss']:.4f})")
    if best_roi["name"] == "no_bet":
        print("  Best betting ROI with actual bets: none")
    else:
        print(f"  Best betting ROI with actual bets: {best_roi['name']} ({best_roi['roi']:.2f}%)")
    print(f"  Recommended betting action: {recommended_betting['name']} ({betting_reason})")

    _print_blend_weight_summary(blend_cfg)
    _print_parameter_impact_summary(parameter_impact_rows)

    _print_external_context_note(ablation_rows, data_audit_rows)

    xgb_best = _best_ablation(ablation_rows, "xgb")
    mlp_best = _best_ablation(ablation_rows, "mlp")
    if xgb_best or mlp_best:
        print("\nFeature-set check:")
        if xgb_best:
            print(f"  Best XGB feature set: {xgb_best['feature_set']} (LogLoss {xgb_best['logloss']:.4f})")
        if mlp_best:
            print(f"  Best MLP feature set: {mlp_best['feature_set']} (LogLoss {mlp_best['logloss']:.4f})")

    robust_positive = [
        row for row in selector_rows
        if row.get("selector_status") == "validation_locked_positive_fold_roi"
        and float(row.get("test_roi", 0.0)) > 0.0
        and int(row.get("test_bets", 0)) >= 50
    ]
    print("\nBetting robustness:")
    if robust_positive:
        best = max(robust_positive, key=lambda row: row["test_roi"])
        print(f"  Robust positive candidate: {best['model']} test ROI {best['test_roi']:.2f}% over {best['test_bets']} bets")
    else:
        print("  No robust positive test strategy found.")

    print("\nDetailed CSV reports written:")
    print(f"  Model summary: {config.final_model_summary_file}")
    print(f"  Data coverage: {config.final_data_enrichment_file}")
    print(f"  Betting robustness: {config.final_betting_robustness_file}")
    print("  Use --full-report to print the long diagnostic tables in the terminal.")


def _ablation_row(model_name: str, feature_set: str, probs: np.ndarray, y_true: np.ndarray):
    class_metrics = class_metric_summary(probs, y_true)
    return {
        "model": model_name,
        "feature_set": feature_set,
        "logloss": float(log_loss(y_true, probs)),
        "accuracy": float((np.argmax(probs, axis=1) == y_true).mean()),
        **class_metrics,
    }


def run_training_pipeline(config: ExperimentConfig = DEFAULT_CONFIG):
    """Run the full per-league training + evaluation pipeline for an experiment.

    Drives every stage end to end (see the module docstring) and persists all
    artifacts/reports under paths derived from ``config.experiment_name``. Cached
    tuning is reused when :func:`_cached_artifacts_are_compatible` (and the config's
    ``force_*`` flags) allow it; otherwise it is recomputed. The canonical artifacts
    of other experiments are never touched.
    """
    _validate_odds_source_config(config)
    print(
        "Odds setup: "
        f"market={config.market_odds_source}, "
        f"betting_price={config.betting_odds_source}, "
        f"market_movement_features={config.include_market_movement_features}"
    )
    compatible_cache = config.use_cached_artifacts and _cached_artifacts_are_compatible(config)
    can_load_param_cache = compatible_cache or config.allow_partial_param_cache
    cached_params = load_json_if_exists(config.params_file) if config.use_cached_artifacts and can_load_param_cache and not config.force_retune_leagues else None
    cached_meta = load_json_if_exists(config.meta_file) if compatible_cache and not config.force_retune_meta else None
    cached_mlp = load_json_if_exists(config.mlp_meta_file) if compatible_cache and not config.force_retune_mlp else None
    cached_blend = load_json_if_exists(config.blend_file) if compatible_cache and not config.force_retune_blend else None
    league_best_params = {} if cached_params is None else cached_params
    all_X_early, all_y_early = [], []
    all_X_late, all_y_late = [], []
    all_late_raw_odds, all_late_info = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
    all_v_info = []
    all_t_probs_model, all_t_mkt_fixed, all_t_raw_odds, all_t_info = [], [], [], []
    all_t_param_baseline_probs = []
    parameter_impact_rows = []
    per_league_test = {}

    for league in config.leagues:
        _print_league_run_header(league)
        df_all = load_league_data(league).sort_values("date").reset_index(drop=True)
        df = df_all[df_all["is_played"] == True].copy().reset_index(drop=True)
        if df.empty:
            continue

        train_fit, val, test = _split_played_periods(df, config)
        _print_split_counts(train_fit, val, test)
        if len(train_fit) == 0 or len(val) == 0 or len(test) == 0:
            continue

        base_train, base_param_val = time_split_val(train_fit)

        if league in league_best_params and not config.force_retune_leagues:
            params = league_best_params[league]
            _print_league_params(params, source="cached")
        else:
            params = tune_league_params(
                base_train,
                base_param_val,
                df[df["date"] < config.train_cut].copy(),
            )
            league_best_params[league] = params
            save_json(config.params_file, league_best_params)
            print(f"Saved tuned league params checkpoint to: {config.params_file}")
            _print_league_params(params, source="tuned")

        val_early, val_late = time_split_val(val)
        odds_kwargs = {
            "market_odds_source": config.market_odds_source,
            "betting_odds_source": config.betting_odds_source,
            "include_market_movement_features": config.include_market_movement_features,
        }
        ve_probs_raw, ve_y, ve_mkt, ve_aux, _ = streaming_block_probs_home_away(
            val_early,
            df,
            params["beta"],
            params["rho"],
            params["decay"],
            params["K"],
            params["ha"],
            **odds_kwargs,
        )
        vl_probs_raw, vl_y, vl_mkt, vl_aux, vl_raw_odds = streaming_block_probs_home_away(
            val_late,
            df,
            params["beta"],
            params["rho"],
            params["decay"],
            params["K"],
            params["ha"],
            **odds_kwargs,
        )
        ve_probs_model = temperature_scale_probs(ve_probs_raw, params["T"])
        vl_probs_model = temperature_scale_probs(vl_probs_raw, params["T"])

        all_X_early.extend(build_meta_features(ve_probs_model, ve_mkt, ve_aux))
        all_y_early.extend(ve_y)
        all_X_late.extend(build_meta_features(vl_probs_model, vl_mkt, vl_aux))
        all_y_late.extend(vl_y)
        all_late_raw_odds.extend(vl_raw_odds)
        all_late_info.extend(_match_info_records(val_late, league))

        v_probs_raw, v_y, v_mkt, v_aux, _ = streaming_block_probs_home_away(
            val,
            df,
            params["beta"],
            params["rho"],
            params["decay"],
            params["K"],
            params["ha"],
            **odds_kwargs,
        )
        v_probs_model = temperature_scale_probs(v_probs_raw, params["T"])
        v_mkt_fixed = ensure_market_probs(v_probs_model, v_mkt)
        all_X_val.extend(build_meta_features(v_probs_model, v_mkt_fixed, v_aux))
        all_y_val.extend(v_y)
        val_info = _match_info_records(val, league)
        all_v_info.extend(val_info)

        t_probs_raw, t_y, t_mkt, t_aux, t_raw_odds = streaming_block_probs_home_away(
            test,
            df,
            params["beta"],
            params["rho"],
            params["decay"],
            params["K"],
            params["ha"],
            **odds_kwargs,
        )
        t_probs_model = temperature_scale_probs(t_probs_raw, params["T"])
        t_mkt_fixed = ensure_market_probs(t_probs_model, t_mkt)

        if config.print_parameter_impact:
            baseline_raw, baseline_y, _, _, _ = streaming_block_probs_home_away(
                test,
                df,
                PARAMETER_IMPACT_BASELINE["beta"],
                PARAMETER_IMPACT_BASELINE["rho"],
                PARAMETER_IMPACT_BASELINE["decay"],
                PARAMETER_IMPACT_BASELINE["K"],
                PARAMETER_IMPACT_BASELINE["ha"],
                **odds_kwargs,
            )
            baseline_probs = temperature_scale_probs(baseline_raw, PARAMETER_IMPACT_BASELINE["T"])
            if not np.array_equal(np.array(t_y, dtype=int), np.array(baseline_y, dtype=int)):
                raise ValueError(f"Parameter impact row alignment failed for {league}")
            parameter_impact_rows.append(
                _parameter_impact_row(
                    league,
                    np.array(t_y, dtype=int),
                    np.array(t_probs_model, dtype=float),
                    np.array(baseline_probs, dtype=float),
                    params,
                )
            )
            all_t_param_baseline_probs.extend(baseline_probs)

        all_X_test.extend(build_meta_features(t_probs_model, t_mkt_fixed, t_aux))
        all_y_test.extend(t_y)
        all_t_probs_model.extend(t_probs_model)
        all_t_mkt_fixed.extend(t_mkt_fixed)
        all_t_raw_odds.extend(t_raw_odds)
        test_info = _match_info_records(test, league)
        all_t_info.extend(test_info)
        per_league_test[league] = {"y": np.array(t_y, dtype=int), "p_model": np.array(t_probs_model, dtype=float), "p_mkt": np.array(t_mkt_fixed, dtype=float)}

    if cached_params is None or config.force_retune_leagues:
        save_json(config.params_file, league_best_params)
        print(f"\nSaved tuned league params to: {config.params_file}")

    X_early_arr, y_early_arr = np.array(all_X_early), np.array(all_y_early)
    X_late_arr, y_late_arr = np.array(all_X_late), np.array(all_y_late)
    X_val_arr, y_val_arr = np.array(all_X_val), np.array(all_y_val)
    X_test_arr, y_test_arr = np.array(all_X_test), np.array(all_y_test)
    t_probs_model_arr, t_mkt_fixed_arr = np.array(all_t_probs_model), np.array(all_t_mkt_fixed)
    late_raw_odds_arr = np.array(all_late_raw_odds)
    raw_odds_arr = np.array(all_t_raw_odds)
    if config.print_parameter_impact and len(all_t_param_baseline_probs) == len(y_test_arr):
        parameter_impact_rows.append(
            _parameter_impact_row(
                "ALL",
                y_test_arr,
                t_probs_model_arr,
                np.array(all_t_param_baseline_probs, dtype=float),
                None,
            )
        )

    threshold = 0.05

    all_cols = list(range(len(FEATURE_COLUMNS)))
    core_feature_names = FEATURE_COLUMNS[:18]
    CORE_COLS = feature_indices(core_feature_names)
    LOCAL_STATS_COLS = feature_indices(LOCAL_STATS_FEATURE_COLUMNS)
    MARKET_CONTEXT_COLS = feature_indices(MARKET_CONTEXT_FEATURE_COLUMNS)
    NEW_LOCAL_COLS = feature_indices(NEW_LOCAL_FEATURE_COLUMNS)
    CORE_PLUS_ODDS_COLS = feature_indices(core_feature_names + MARKET_CONTEXT_FEATURE_COLUMNS)
    CORE_PLUS_STATS_COLS = feature_indices(core_feature_names + LOCAL_STATS_FEATURE_COLUMNS)
    CORE_PLUS_NEW_LOCAL_COLS = feature_indices(core_feature_names + NEW_LOCAL_FEATURE_COLUMNS)
    CORE_PLUS_UNDERSTAT_XG_COLS = feature_indices(core_feature_names + UNDERSTAT_XG_FEATURE_COLUMNS)
    CORE_PLUS_EXTERNAL_CONTEXT_COLS = feature_indices(core_feature_names + EXTERNAL_CONTEXT_FEATURE_COLUMNS)
    MARKET_PLUS_UNDERSTAT_XG_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *UNDERSTAT_XG_FEATURE_COLUMNS,
    ])
    MARKET_PLUS_EXTERNAL_CONTEXT_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *EXTERNAL_CONTEXT_FEATURE_COLUMNS,
    ])
    LINEUP_FEATURE_COLUMNS = ["home_lineup_strength", "away_lineup_strength", "lineup_strength_diff"]
    MARKET_PLUS_LINEUP_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *LINEUP_FEATURE_COLUMNS,
    ])
    MARKET_CONTEXT_PLUS_UNDERSTAT_XG_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *UNDERSTAT_XG_FEATURE_COLUMNS,
    ])
    MARKET_CONTEXT_PLUS_EXTERNAL_CONTEXT_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *EXTERNAL_CONTEXT_FEATURE_COLUMNS,
    ])
    NO_NEW_LOCAL_FEATURES_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in NEW_LOCAL_FEATURE_COLUMNS]
    NO_NEW_DATA_FEATURES_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in NEW_DATA_FEATURE_COLUMNS]
    NO_NEW_LOCAL_STATS_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in LOCAL_STATS_FEATURE_COLUMNS]
    NO_ODDS_MOVEMENT_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in MARKET_CONTEXT_FEATURE_COLUMNS]
    NO_UNDERSTAT_XG_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in UNDERSTAT_XG_FEATURE_COLUMNS]
    NO_EXTERNAL_CONTEXT_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in EXTERNAL_CONTEXT_FEATURE_COLUMNS]
    market_feature_names = {
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        "market_move_home",
        "market_move_draw",
        "market_move_away",
        "ou25_over_prob",
        "ah_line",
    }
    NO_MARKET_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in market_feature_names]
    NO_ELO_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col != "elo_diff"]
    NO_XG_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if col not in {"total_xg", "xg_diff"}]
    NO_MOMENTUM_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if not col.startswith("mom_")]
    NO_REST_COLS = MLP_NO_REST_COLS
    NO_FORM_COLS = [i for i, col in enumerate(FEATURE_COLUMNS) if not col.startswith("form_")]
    MODEL_ONLY_COLS = [0, 1, 2]
    MARKET_ONLY_COLS = [3, 4, 5]
    MARKET_PLUS_CONTEXT_COLS = feature_indices([
        "market_logit_home",
        "market_logit_draw",
        "market_logit_away",
        *MARKET_CONTEXT_FEATURE_COLUMNS,
    ])
    # Candidate feature subsets the XGBoost meta-model chooses among (by late-val
    # log loss). On the canonical experiment the market-only subset wins; a config
    # with meta_feature_set set restricts this to a single named subset instead.
    learned_model_feature_sets = {
        "all_features": all_cols,
        "core_18": CORE_COLS,
        "market_only": MARKET_ONLY_COLS,
        "market_plus_context": MARKET_PLUS_CONTEXT_COLS,
        "core_plus_odds": CORE_PLUS_ODDS_COLS,
        "core_plus_stats": CORE_PLUS_STATS_COLS,
        "core_plus_new_local": CORE_PLUS_NEW_LOCAL_COLS,
        "core_plus_understat_xg": CORE_PLUS_UNDERSTAT_XG_COLS,
        "core_plus_external_context": CORE_PLUS_EXTERNAL_CONTEXT_COLS,
        "market_plus_understat_xg": MARKET_PLUS_UNDERSTAT_XG_COLS,
        "market_plus_external_context": MARKET_PLUS_EXTERNAL_CONTEXT_COLS,
        "market_plus_lineup": MARKET_PLUS_LINEUP_COLS,
        "market_context_plus_understat_xg": MARKET_CONTEXT_PLUS_UNDERSTAT_XG_COLS,
        "market_context_plus_external_context": MARKET_CONTEXT_PLUS_EXTERNAL_CONTEXT_COLS,
        "no_new_local_stats": NO_NEW_LOCAL_STATS_COLS,
        "no_odds_context": NO_ODDS_MOVEMENT_COLS,
        "no_understat_xg": NO_UNDERSTAT_XG_COLS,
        "no_external_context": NO_EXTERNAL_CONTEXT_COLS,
    }

    X_early_nm = X_early_arr[:, NO_MARKET_COLS]
    X_late_nm = X_late_arr[:, NO_MARKET_COLS]
    X_val_nm = X_val_arr[:, NO_MARKET_COLS]
    X_test_nm = X_test_arr[:, NO_MARKET_COLS]

    recomputed_market = np.array(
        [market_probs_from_odds_row(o[0], o[1], o[2]) for o in raw_odds_arr],
        dtype=float,
    )

    mask = np.isfinite(recomputed_market).all(axis=1)
    if np.any(mask):
        max_diff = np.max(np.abs(recomputed_market[mask] - t_mkt_fixed_arr[mask]))
        print("SANITY max market diff:", max_diff)

        ev_check = recomputed_market[mask] * raw_odds_arr[mask] - 1.0
        print("SANITY max raw market EV:", np.max(ev_check))
    else:
        print("SANITY market odds check skipped: no valid raw market odds in test set.")

    print("\n" + "=" * 50)
    print("=== META-MODEL Evaluation (XGBoost) ===")
    print("=" * 50)

    if cached_meta is not None and not config.force_retune_meta:
        best_meta_cfg = cached_meta
        print("Using cached XGBoost hyperparameters...")
    else:
        best_meta_cfg = tune_xgb_hyperparams(X_early_arr, y_early_arr, X_late_arr, y_late_arr)
        meta_candidate_sets = learned_model_feature_sets
        if config.meta_feature_set:
            if config.meta_feature_set not in learned_model_feature_sets:
                raise ValueError(
                    f"Unknown meta_feature_set {config.meta_feature_set!r}; "
                    f"available: {sorted(learned_model_feature_sets)}"
                )
            meta_candidate_sets = {config.meta_feature_set: learned_model_feature_sets[config.meta_feature_set]}
            print(f"Forcing XGBoost feature set: {config.meta_feature_set}")
        xgb_subset, xgb_subset_rows = tune_feature_subset(
            lambda: XGBClassifier(
                n_estimators=int(best_meta_cfg["n_estimators"]),
                learning_rate=float(best_meta_cfg["learning_rate"]),
                max_depth=int(best_meta_cfg["max_depth"]),
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=config.random_state,
                n_jobs=-1,
            ),
            X_early_arr,
            y_early_arr,
            X_late_arr,
            y_late_arr,
            meta_candidate_sets,
        )
        best_meta_cfg["feature_set"] = xgb_subset["name"]
        best_meta_cfg["feature_columns"] = [FEATURE_COLUMNS[i] for i in xgb_subset["cols"]]
        best_meta_cfg["late_val_logloss"] = float(xgb_subset["late_val_logloss"])
        best_meta_cfg["feature_subset_scores"] = [
            {
                "name": row["name"],
                "late_val_logloss": row["late_val_logloss"],
                "feature_columns": [FEATURE_COLUMNS[i] for i in row["cols"]],
            }
            for row in xgb_subset_rows
        ]
        save_json(config.meta_file, best_meta_cfg)
        print(f"Saved XGBoost hyperparams to: {config.meta_file}")
    print(f"Best XGBoost Config -> LR: {best_meta_cfg['learning_rate']}, Depth: {best_meta_cfg['max_depth']}, Trees: {best_meta_cfg['n_estimators']}")
    print(f"Late VAL LogLoss: {round(best_meta_cfg['late_val_logloss'], 4)}")
    xgb_feature_columns = best_meta_cfg.get("feature_columns", FEATURE_COLUMNS)
    xgb_cols = feature_indices(xgb_feature_columns)
    print(f"XGBoost feature set: {best_meta_cfg.get('feature_set', 'all_features')} ({len(xgb_cols)} features)")

    can_load_meta_model = compatible_cache and config.model_file.exists() and not config.force_retune_meta and not config.force_refit_meta_model
    if can_load_meta_model:
        print(f"Loading trained XGBoost meta-model from: {config.model_file}")
        meta_final = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=config.random_state, n_jobs=-1)
        meta_final.load_model(str(config.model_file))
    else:
        print("Fitting final XGBoost meta-model...")
        meta_final = fit_xgb_model(X_val_arr[:, xgb_cols], y_val_arr, best_meta_cfg)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        meta_final.save_model(str(config.model_file))
        print(f"Saved trained XGBoost meta-model to: {config.model_file}")

    # Candidate feature subsets the MLP chooses among (mirrors the XGBoost search);
    # mlp_feature_set in the config can force a single named subset.
    mlp_candidate_feature_sets = {
        "default": MLP_DEFAULT_COLS,
        "default_plus_odds_context": sorted(set(MLP_DEFAULT_COLS + MARKET_CONTEXT_COLS)),
        "default_plus_rolling_stats": sorted(set(MLP_DEFAULT_COLS + LOCAL_STATS_COLS)),
        "default_plus_new_local": sorted(set(MLP_DEFAULT_COLS + NEW_LOCAL_COLS)),
        "default_plus_understat_xg": sorted(set(MLP_DEFAULT_COLS + feature_indices(UNDERSTAT_XG_FEATURE_COLUMNS))),
        "default_plus_lineup": sorted(set(MLP_DEFAULT_COLS + feature_indices(["home_lineup_strength", "away_lineup_strength", "lineup_strength_diff"]))),
        "default_plus_team_news": sorted(set(MLP_DEFAULT_COLS + feature_indices(TEAM_NEWS_FEATURE_COLUMNS))),
        "default_plus_weather": sorted(set(MLP_DEFAULT_COLS + feature_indices(WEATHER_FEATURE_COLUMNS))),
        "default_plus_external_context": sorted(set(MLP_DEFAULT_COLS + feature_indices(EXTERNAL_CONTEXT_FEATURE_COLUMNS))),
        "core_18": CORE_COLS,
        "core_plus_odds_context": CORE_PLUS_ODDS_COLS,
        "core_plus_rolling_stats": CORE_PLUS_STATS_COLS,
        "core_plus_understat_xg": CORE_PLUS_UNDERSTAT_XG_COLS,
        "core_plus_external_context": CORE_PLUS_EXTERNAL_CONTEXT_COLS,
        "all_features": all_cols,
    }

    if cached_mlp is not None and not config.force_retune_mlp:
        best_mlp_cfg = cached_mlp
        print("Using cached MLP hyperparameters...")
    else:
        base_mlp_cfg = tune_mlp_hyperparams(
            X_early_arr[:, MLP_DEFAULT_COLS],
            y_early_arr,
            X_late_arr[:, MLP_DEFAULT_COLS],
            y_late_arr,
        )
        mlp_candidate_sets = mlp_candidate_feature_sets
        if config.mlp_feature_set:
            if config.mlp_feature_set not in mlp_candidate_feature_sets:
                raise ValueError(
                    f"Unknown mlp_feature_set {config.mlp_feature_set!r}; "
                    f"available: {sorted(mlp_candidate_feature_sets)}"
                )
            mlp_candidate_sets = {config.mlp_feature_set: mlp_candidate_feature_sets[config.mlp_feature_set]}
            print(f"Forcing MLP feature set: {config.mlp_feature_set}")
        best_subset, subset_rows = tune_mlp_feature_subset(
            X_early_arr,
            y_early_arr,
            X_late_arr,
            y_late_arr,
            base_mlp_cfg,
            mlp_candidate_sets,
        )
        best_mlp_cfg = dict(base_mlp_cfg)
        best_mlp_cfg["feature_set"] = best_subset["name"]
        best_mlp_cfg["feature_columns"] = [FEATURE_COLUMNS[i] for i in best_subset["cols"]]
        best_mlp_cfg["late_val_logloss"] = float(best_subset["late_val_logloss"])
        best_mlp_cfg["temperature"] = float(best_subset["temperature"])
        best_mlp_cfg["feature_subset_scores"] = [
            {
                "name": row["name"],
                "late_val_logloss": row["late_val_logloss"],
                "feature_columns": [FEATURE_COLUMNS[i] for i in row["cols"]],
            }
            for row in subset_rows
        ]
        save_json(config.mlp_meta_file, best_mlp_cfg)
        print(f"Saved MLP hyperparams to: {config.mlp_meta_file}")

    mlp_feature_columns = best_mlp_cfg.get("feature_columns", MLP_DEFAULT_FEATURE_COLUMNS)
    mlp_cols = feature_indices(mlp_feature_columns)
    X_val_mlp = X_val_arr[:, mlp_cols]
    X_test_mlp = X_test_arr[:, mlp_cols]
    X_early_mlp = X_early_arr[:, mlp_cols]
    X_late_mlp = X_late_arr[:, mlp_cols]

    can_load_mlp_model = compatible_cache and config.mlp_model_file.exists() and not config.force_retune_mlp and not config.force_refit_mlp_model
    if can_load_mlp_model:
        print(f"Loading trained MLP model from: {config.mlp_model_file}")
        mlp_model = load_pickle_if_exists(config.mlp_model_file)
    else:
        print("Fitting final MLP model...")
        mlp_model = make_mlp_pipeline(best_mlp_cfg)
        mlp_model.fit(X_val_mlp, y_val_arr)
        save_pickle(config.mlp_model_file, mlp_model)
        print(f"Saved trained MLP model to: {config.mlp_model_file}")

    logreg_cfg = tune_logreg_hyperparams(
        X_early_arr[:, MARKET_PLUS_CONTEXT_COLS],
        y_early_arr,
        X_late_arr[:, MARKET_PLUS_CONTEXT_COLS],
        y_late_arr,
    )
    logreg_subset, logreg_subset_rows = tune_feature_subset(
        lambda: make_logreg_pipeline(logreg_cfg["C"]),
        X_early_arr,
        y_early_arr,
        X_late_arr,
        y_late_arr,
        learned_model_feature_sets,
        temperature_scale=True,
    )
    logreg_feature_columns = [FEATURE_COLUMNS[i] for i in logreg_subset["cols"]]
    logreg_cols = feature_indices(logreg_feature_columns)
    logreg_cfg["feature_set"] = logreg_subset["name"]
    logreg_cfg["feature_columns"] = logreg_feature_columns
    logreg_cfg["late_val_logloss"] = float(logreg_subset["late_val_logloss"])
    logreg_cfg["temperature"] = float(logreg_subset["temperature"])
    logreg_cfg["feature_subset_scores"] = [
        {
            "name": row["name"],
            "late_val_logloss": row["late_val_logloss"],
            "feature_columns": [FEATURE_COLUMNS[i] for i in row["cols"]],
        }
        for row in logreg_subset_rows
    ]
    print(f"LogReg feature set: {logreg_cfg['feature_set']} ({len(logreg_cols)} features)")
    logreg_final = make_logreg_pipeline(logreg_cfg["C"])
    logreg_final.fit(X_val_arr[:, logreg_cols], y_val_arr)
    save_json(config.logreg_meta_file, logreg_cfg)
    save_pickle(config.logreg_model_file, logreg_final)

    xgb_late_model = fit_xgb_model(X_early_arr[:, xgb_cols], y_early_arr, best_meta_cfg)
    late_probs_xgb = xgb_late_model.predict_proba(X_late_arr[:, xgb_cols])
    mlp_late_model = make_mlp_pipeline(best_mlp_cfg)
    mlp_late_model.fit(X_early_mlp, y_early_arr)
    late_probs_mlp_raw = mlp_late_model.predict_proba(X_late_mlp)
    late_probs_mlp = temperature_scale_probs(late_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    late_probs_base = probs_from_meta_features(X_late_arr, 0)
    late_probs_mkt = probs_from_meta_features(X_late_arr, 3)
    logreg_late_model = make_logreg_pipeline(logreg_cfg["C"])
    logreg_late_model.fit(X_early_arr[:, logreg_cols], y_early_arr)
    late_probs_logreg_raw = logreg_late_model.predict_proba(X_late_arr[:, logreg_cols])
    late_probs_logreg = temperature_scale_probs(late_probs_logreg_raw, float(logreg_cfg["temperature"]))
    late_market_ll = float(log_loss(y_late_arr, late_probs_mkt))
    late_mlp_ll = float(log_loss(y_late_arr, late_probs_mlp))
    if cached_blend is not None and "market_correction" in cached_blend and not config.force_retune_blend:
        market_corr_cfg = cached_blend["market_correction"]
    else:
        market_corr_cfg = tune_market_logit_correction(
            y_late_arr,
            late_probs_mkt,
            {
                "base": late_probs_base,
                "xgb": late_probs_xgb,
                "mlp": late_probs_mlp,
            },
            min_improvement=0.0005,
            fold_masks=_chronological_fold_masks(all_late_info, n_folds=2),
            min_fold_improvement=0.0,
        )
    late_probs_market_corr = apply_market_logit_correction(
        late_probs_mkt,
        {
            "base": late_probs_base,
            "xgb": late_probs_xgb,
            "mlp": late_probs_mlp,
        },
        market_corr_cfg,
    )
    corr_status = "accepted" if market_corr_cfg.get("accepted") else "collapsed_to_market"
    print(
        "Market correction "
        f"({corr_status}): source={market_corr_cfg.get('source_model')} "
        f"alpha={float(market_corr_cfg.get('alpha', 0.0)):.3f} "
        f"late_val_logloss={float(market_corr_cfg.get('late_val_logloss', late_market_ll)):.4f} "
        f"market={late_market_ll:.4f}"
    )
    if market_corr_cfg.get("rejection_reason"):
        print(f"Market correction rejected: {market_corr_cfg['rejection_reason']}")
    mlp_allowed_in_blend = late_mlp_ll < late_market_ll - 0.002
    if not mlp_allowed_in_blend:
        print(
            f"MLP excluded from blend: late_val_logloss={late_mlp_ll:.4f}, "
            f"market={late_market_ll:.4f}"
        )

    if cached_blend is not None and not config.force_retune_blend:
        blend_cfg = cached_blend
        print("Using cached blend weights...")
    else:
        blend_cfg = tune_blend_weights(
            y_late_arr,
            late_probs_base,
            late_probs_mkt,
            late_probs_xgb,
            late_probs_mlp if mlp_allowed_in_blend else None,
            step=0.05,
        )
        blend_cfg["mlp_allowed"] = bool(mlp_allowed_in_blend)
        blend_cfg["mlp_gate"] = {"late_mlp_logloss": late_mlp_ll, "late_market_logloss": late_market_ll, "margin_required": 0.002}
        blend_cfg["market_correction"] = market_corr_cfg
        save_json(config.blend_file, blend_cfg)
        print(f"Saved blend weights to: {config.blend_file}")
    if "market_correction" not in blend_cfg:
        blend_cfg["market_correction"] = market_corr_cfg
    print(f"Blend weights: {blend_cfg['weights']}")
    _print_blend_weight_summary(blend_cfg)
    print(f"Late VAL LogLoss: {round(blend_cfg['late_val_logloss'], 4)}")
    late_probs_blend = blend_probabilities(
        blend_cfg["weights"],
        {
            "base": late_probs_base,
            "market": late_probs_mkt,
            "xgb": late_probs_xgb,
            "mlp": late_probs_mlp if blend_cfg.get("mlp_allowed", mlp_allowed_in_blend) else None,
        },
    )

    t_probs_meta = meta_final.predict_proba(X_test_arr[:, xgb_cols])
    t_probs_mlp_raw = mlp_model.predict_proba(X_test_mlp)
    t_probs_mlp = temperature_scale_probs(t_probs_mlp_raw, float(best_mlp_cfg["temperature"]))
    t_probs_logreg_raw = logreg_final.predict_proba(X_test_arr[:, logreg_cols])
    t_probs_logreg = temperature_scale_probs(t_probs_logreg_raw, float(logreg_cfg["temperature"]))
    t_probs_market_corr = apply_market_logit_correction(
        t_mkt_fixed_arr,
        {
            "base": t_probs_model_arr,
            "xgb": t_probs_meta,
            "mlp": t_probs_mlp,
        },
        blend_cfg.get("market_correction", market_corr_cfg),
    )
    t_probs_blend = blend_probabilities(
        blend_cfg["weights"],
        {
            "base": t_probs_model_arr,
            "market": t_mkt_fixed_arr,
            "xgb": t_probs_meta,
            "mlp": t_probs_mlp if blend_cfg.get("mlp_allowed", mlp_allowed_in_blend) else None,
        },
    )
    if config.print_full_reports:
        print("\n" + "=" * 50)
        print("=== QUICK NO-MARKET ABLATION ===")
        print("=" * 50)

    # Reuse current best hyperparams, but retrain on no-market features only
    meta_nm = fit_xgb_model(X_val_nm, y_val_arr, best_meta_cfg)
    t_probs_meta_nm = meta_nm.predict_proba(X_test_nm)

    mlp_nm = make_mlp_pipeline(best_mlp_cfg)
    mlp_nm.fit(X_val_nm, y_val_arr)
    t_probs_mlp_nm_raw = mlp_nm.predict_proba(X_test_nm)
    t_probs_mlp_nm = temperature_scale_probs(t_probs_mlp_nm_raw, float(best_mlp_cfg["temperature"]))

    if config.print_full_reports:
        print_prob_report("META NO-MARKET", t_probs_meta_nm, y_test_arr)
        print_prob_report("MLP NO-MARKET", t_probs_mlp_nm, y_test_arr)

    if config.print_full_reports:
        print("\n--- Betting simulation: no-market ablation ---")
    bets_nm_xgb, wins_nm_xgb, profit_nm_xgb, roi_nm_xgb, avg_odds_nm_xgb = simulate_value_betting(
        t_probs_meta_nm,
        raw_odds_arr,
        y_test_arr,
        edge_threshold=threshold,
        match_info=None,
        verbose=config.print_full_reports,
    )
    if config.print_full_reports:
        print(f"META NO-MARKET ROI: {round(roi_nm_xgb, 2)}% | Bets: {bets_nm_xgb}")

    bets_nm_mlp, wins_nm_mlp, profit_nm_mlp, roi_nm_mlp, avg_odds_nm_mlp = simulate_value_betting(
        t_probs_mlp_nm,
        raw_odds_arr,
        y_test_arr,
        edge_threshold=threshold,
        match_info=None,
        verbose=config.print_full_reports,
    )
    if config.print_full_reports:
        print(f"MLP NO-MARKET ROI: {round(roi_nm_mlp, 2)}% | Bets: {bets_nm_mlp}")

        print_prob_report("BASE (Model only, calibrated)", t_probs_model_arr, y_test_arr)
        print_prob_report("MARKET (odds implied)", t_mkt_fixed_arr, y_test_arr)
        print_prob_report("META (Market + Model)", t_probs_meta, y_test_arr)
        print_prob_report("LOGREG (regularized baseline)", t_probs_logreg, y_test_arr)
        print_prob_report("DEEP LEARNING (MLP, calibrated)", t_probs_mlp, y_test_arr)
        print_prob_report("ENSEMBLE (XGB + MLP + Market + Base)", t_probs_blend, y_test_arr)
        print_confusion("ENSEMBLE", t_probs_blend, y_test_arr)
        print_per_league_test_metrics(list(config.leagues), per_league_test, t_probs_meta, t_probs_mlp, t_probs_blend)

    ablation_specs = [
        ("all_features", all_cols),
        ("core_18", CORE_COLS),
        ("no_new_data_features", NO_NEW_DATA_FEATURES_COLS),
        ("no_new_local_features", NO_NEW_LOCAL_FEATURES_COLS),
        ("no_new_local_stats", NO_NEW_LOCAL_STATS_COLS),
        ("no_understat_xg", NO_UNDERSTAT_XG_COLS),
        ("no_odds_context", NO_ODDS_MOVEMENT_COLS),
        ("core_plus_odds", CORE_PLUS_ODDS_COLS),
        ("core_plus_stats", CORE_PLUS_STATS_COLS),
        ("core_plus_new_local", CORE_PLUS_NEW_LOCAL_COLS),
        ("core_plus_understat_xg", CORE_PLUS_UNDERSTAT_XG_COLS),
        ("core_plus_external_context", CORE_PLUS_EXTERNAL_CONTEXT_COLS),
        ("market_plus_understat_xg", MARKET_PLUS_UNDERSTAT_XG_COLS),
        ("market_plus_external_context", MARKET_PLUS_EXTERNAL_CONTEXT_COLS),
        ("market_context_plus_understat_xg", MARKET_CONTEXT_PLUS_UNDERSTAT_XG_COLS),
        ("market_context_plus_external_context", MARKET_CONTEXT_PLUS_EXTERNAL_CONTEXT_COLS),
        ("no_external_context", NO_EXTERNAL_CONTEXT_COLS),
        ("no_market", NO_MARKET_COLS),
        ("no_elo", NO_ELO_COLS),
        ("no_xg", NO_XG_COLS),
        ("no_momentum", NO_MOMENTUM_COLS),
        ("no_rest", NO_REST_COLS),
        ("no_form", NO_FORM_COLS),
        ("model_only", MODEL_ONLY_COLS),
        ("market_only", MARKET_ONLY_COLS),
        ("market_plus_context", MARKET_PLUS_CONTEXT_COLS),
    ]
    ablation_rows = []
    for feature_set_name, cols in ablation_specs:
        X_val_slice = X_val_arr[:, cols]
        X_test_slice = X_test_arr[:, cols]

        xgb_ablation = fit_xgb_model(X_val_slice, y_val_arr, best_meta_cfg)
        xgb_probs = xgb_ablation.predict_proba(X_test_slice)
        ablation_rows.append(_ablation_row("xgb", feature_set_name, xgb_probs, y_test_arr))

        mlp_ablation = make_mlp_pipeline(best_mlp_cfg)
        mlp_ablation.fit(X_val_slice, y_val_arr)
        mlp_probs_raw = mlp_ablation.predict_proba(X_test_slice)
        mlp_probs = temperature_scale_probs(mlp_probs_raw, float(best_mlp_cfg["temperature"]))
        ablation_rows.append(_ablation_row("mlp", feature_set_name, mlp_probs, y_test_arr))
    if config.print_full_reports:
        _print_feature_ablation_summary(ablation_rows)

    selection_rows = [
        _model_summary_row("base", t_probs_model_arr, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("market", t_mkt_fixed_arr, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("market_corr", t_probs_market_corr, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("meta", t_probs_meta, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("logreg", t_probs_logreg, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("mlp", t_probs_mlp, y_test_arr, raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("ensemble", t_probs_blend, y_test_arr, raw_odds_arr, edge_threshold=threshold),
    ]
    validation_selection_rows = [
        _model_summary_row("base", late_probs_base, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("market", late_probs_mkt, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("market_corr", late_probs_market_corr, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("meta", late_probs_xgb, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("logreg", late_probs_logreg, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("mlp", late_probs_mlp, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
        _model_summary_row("ensemble", late_probs_blend, y_late_arr, late_raw_odds_arr, edge_threshold=threshold),
    ]
    best_logloss, best_roi = _print_model_selection_summary(selection_rows, full_report=config.print_full_reports)
    recommended_betting, betting_reason = _select_recommended_betting_model(validation_selection_rows, selection_rows)
    if config.print_full_reports:
        print("Recommended betting model:", recommended_betting["name"], f"({betting_reason})")

    run_ts = datetime.now(UTC).isoformat()
    result_rows = []
    for row in selection_rows:
        result_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "train_cut": config.train_cut,
            "test_cut": config.test_cut,
            "test_end": config.test_end,
            "model": row["name"],
            "logloss": round(row["logloss"], 6),
            "accuracy": round(row["accuracy"], 6),
            "macro_f1": round(row["macro_f1"], 6),
            "home_recall": round(row["home_recall"], 6),
            "draw_recall": round(row["draw_recall"], 6),
            "away_recall": round(row["away_recall"], 6),
            "home_precision": round(row["home_precision"], 6),
            "draw_precision": round(row["draw_precision"], 6),
            "away_precision": round(row["away_precision"], 6),
            "bets": row["bets"],
            "hit_rate": round(row["hit_rate"], 4),
            "roi": round(row["roi"], 4),
            "profit": round(row["profit"], 4),
            "avg_odds": round(row["avg_odds"], 4),
            "winner_logloss": int(row["name"] == min(selection_rows, key=lambda r: r["logloss"])["name"]),
            "winner_roi": int(row["name"] == best_roi["name"] and row["bets"] > 0),
        })
    append_rows_to_csv(config.results_csv_file, result_rows)

    ablation_csv_rows = []
    for row in ablation_rows:
        ablation_csv_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "model": row["model"],
            "feature_set": row["feature_set"],
            "logloss": round(row["logloss"], 6),
            "accuracy": round(row["accuracy"], 6),
            "macro_f1": round(row["macro_f1"], 6),
            "home_recall": round(row["home_recall"], 6),
            "draw_recall": round(row["draw_recall"], 6),
            "away_recall": round(row["away_recall"], 6),
        })
    append_rows_to_csv(config.ablations_csv_file, ablation_csv_rows)
    _write_final_summary(config, selection_rows, ablation_rows, run_ts)
    late_validation_strategy_probs = {
        "base": late_probs_base,
        "market": late_probs_mkt,
        "market_corr": late_probs_market_corr,
        "meta": late_probs_xgb,
        "logreg": late_probs_logreg,
        "mlp": late_probs_mlp,
        "ensemble": late_probs_blend,
    }
    test_strategy_probs = {
        "base": t_probs_model_arr,
        "market": t_mkt_fixed_arr,
        "market_corr": t_probs_market_corr,
        "meta": t_probs_meta,
        "logreg": t_probs_logreg,
        "mlp": t_probs_mlp,
        "ensemble": t_probs_blend,
    }
    write_probability_quality_report(
        config,
        run_ts,
        {
            "validation": late_validation_strategy_probs,
            "test": test_strategy_probs,
        },
        {
            "validation": y_late_arr,
            "test": y_test_arr,
        },
    )
    write_class_marginal_calibration(
        config,
        run_ts,
        {
            "validation": late_validation_strategy_probs,
            "test": test_strategy_probs,
        },
        {
            "validation": y_late_arr,
            "test": y_test_arr,
        },
    )
    write_per_match_predictions(
        config,
        run_ts,
        test_strategy_probs,
        y_test_arr,
        raw_odds_arr,
        all_t_info,
    )
    selector_rows = write_validation_selected_betting_reports(
        config,
        run_ts,
        late_validation_strategy_probs,
        late_raw_odds_arr,
        y_late_arr,
        all_late_info,
        test_strategy_probs,
        raw_odds_arr,
        y_test_arr,
        all_t_info,
    )
    write_alternative_market_report(
        config,
        run_ts,
        {
            "validation": late_validation_strategy_probs,
            "test": test_strategy_probs,
        },
        {
            "validation": X_late_arr,
            "test": X_test_arr,
        },
        {
            "validation": all_late_info,
            "test": all_t_info,
        },
    )
    data_audit_rows = write_data_enrichment_audit(
        config,
        run_ts,
        {
            "validation": all_v_info,
            "test": all_t_info,
        },
    )
    write_betting_robustness_report(
        config,
        run_ts,
        test_strategy_probs,
        raw_odds_arr,
        y_test_arr,
        all_t_info,
        edge_threshold=threshold,
    )
    write_league_specific_strategy_report(
        config,
        run_ts,
        late_validation_strategy_probs,
        late_raw_odds_arr,
        y_late_arr,
        all_late_info,
        test_strategy_probs,
        raw_odds_arr,
        y_test_arr,
        all_t_info,
        edge_threshold=threshold,
    )

    bets, wins, profit, roi, avg_odds = simulate_value_betting(
        t_probs_blend,
        raw_odds_arr,
        y_test_arr,
        edge_threshold=threshold,
        match_info=all_t_info if config.detailed_betting_log else None,
        verbose=config.print_full_reports,
    )
    if config.print_full_reports:
        print("\n--- Betting simulation - All Top 5 Leagues ---")
        print(f"Ensemble Strategy (Edge > {threshold*100}%):")
        print(f"Total Bets Placed: {bets} (out of {len(X_test_arr)} matches)")
        if bets > 0:
            print(f"Won Bets: {wins} ({round(wins / bets * 100, 1)}% Hit Rate)")
            print(f"Average Odds Played: {round(avg_odds, 2)}")
        print(f"Net Profit: {round(profit, 2)} units")
        print(f"ROI: {round(roi, 2)}%")
    else:
        _print_concise_run_summary(
            config=config,
            selection_rows=selection_rows,
            ablation_rows=ablation_rows,
            data_audit_rows=data_audit_rows,
            selector_rows=selector_rows,
            parameter_impact_rows=parameter_impact_rows,
            blend_cfg=blend_cfg,
            best_logloss=best_logloss,
            best_roi=best_roi,
            recommended_betting=recommended_betting,
            betting_reason=betting_reason,
        )

    if config.print_verbose_audits:
        print_alignment_audit(
            probs=t_probs_blend,
            raw_odds=raw_odds_arr,
            y_true=y_test_arr,
            match_info=all_t_info,
            title="ALIGNMENT AUDIT (first 20 test matches)",
            n=20,
        )

        print_strategy_comparison(
            strategy_probs={
                "base": t_probs_model_arr,
                "market": t_mkt_fixed_arr,
                "market_corr": t_probs_market_corr,
                "meta": t_probs_meta,
                "mlp": t_probs_mlp,
                "logreg": t_probs_logreg,
                "met_no_market": t_probs_meta_nm,
                "mlp_no_market": t_probs_mlp_nm,
                "ensemble": t_probs_blend,
            },
            raw_odds=raw_odds_arr,
            y_true=y_test_arr,
            edge_threshold=threshold,
        )

        print_market_dependency_audit(
            y_true=y_test_arr,
            p_base=t_probs_model_arr,
            p_market=t_mkt_fixed_arr,
            p_meta=t_probs_meta,
            p_mlp=t_probs_mlp,
            p_ens=t_probs_blend,
        )

        print_profit_profile_audit(
            probs=t_probs_blend,
            raw_odds=raw_odds_arr,
            y_true=y_test_arr,
            match_info=all_t_info,
            edge_threshold=threshold,
        )

    for league in config.leagues:
        if league in league_best_params:
            league_best_params[league]["_blend_cfg"] = blend_cfg

    save_manifest(config.manifest_file, {
        "pipeline_version": PIPELINE_VERSION,
        "config": config.as_manifest(),
        "data_fingerprint": _data_fingerprint(),
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "xgb": best_meta_cfg,
        "xgb_feature_columns": xgb_feature_columns,
        "xgb_n_features": len(xgb_feature_columns),
        "logreg": logreg_cfg,
        "logreg_feature_columns": logreg_feature_columns,
        "logreg_n_features": len(logreg_feature_columns),
        "mlp": best_mlp_cfg,
        "mlp_feature_columns": mlp_feature_columns,
        "mlp_n_features": len(mlp_feature_columns),
        "market_correction": blend_cfg.get("market_correction", market_corr_cfg),
        "blend": blend_cfg,
        "parameter_impact": parameter_impact_rows,
        "selection_summary": {
            "winner_by_logloss": best_logloss["name"],
            "winner_by_logloss_value": round(best_logloss["logloss"], 6),
            "winner_by_betting_roi": best_roi["name"],
            "winner_by_betting_roi_value": round(best_roi["roi"], 4),
            "recommended_probability_model": best_logloss["name"],
            "recommended_betting_model": recommended_betting["name"],
            "recommended_betting_reason": betting_reason,
        },
    })

    if config.generate_upcoming_picks:
        generate_upcoming_matchday_picks(
            config.leagues,
            league_best_params,
            meta_final,
            best_meta_cfg,
            mlp_model,
            best_mlp_cfg,
            logreg_model=logreg_final,
            logreg_cfg=logreg_cfg,
            max_window_days=config.max_upcoming_window_days,
            pick_model=best_logloss["name"],
        )
    return {
        "league_best_params": league_best_params,
        "xgb_cfg": best_meta_cfg,
        "mlp_cfg": best_mlp_cfg,
        "blend_cfg": blend_cfg,
        "test_logloss": float(log_loss(y_test_arr, t_probs_blend)),
    }
