from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.artifact_store import append_rows_to_csv
from src.config import ExperimentConfig
from src.evaluation import betting_records
from src.feature_builder import FEATURE_COLUMNS
from src.metrics import class_metric_summary, multiclass_brier, top_label_ece


EDGE_THRESHOLDS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
MIN_PROBABILITIES = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
ODDS_BANDS = (
    (1.01, 10.0),
    (1.20, 5.00),
    (1.20, 3.50),
    (1.50, 5.00),
    (2.00, 7.00),
    (3.00, 10.00),
)


def _finite_float(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(value, errors="coerce")
    if np.isfinite(out):
        return float(out)
    return float(default)


def _metric_row(
    *,
    run_ts: str,
    experiment_name: str,
    report: str,
    split: str,
    model: str,
    group_type: str,
    group_value: str,
    bets: pd.DataFrame,
    total_matches: int,
    selector_status: str = "",
    edge_threshold: float | None = None,
    min_probability: float | None = None,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> dict:
    if bets.empty:
        return {
            "run_ts_utc": run_ts,
            "experiment_name": experiment_name,
            "report": report,
            "split": split,
            "model": model,
            "group_type": group_type,
            "group_value": group_value,
            "selector_status": selector_status,
            "edge_threshold": edge_threshold,
            "min_probability": min_probability,
            "min_odds": min_odds,
            "max_odds": max_odds,
            "matches": int(total_matches),
            "bets": 0,
            "coverage": 0.0,
            "wins": 0,
            "hit_rate": 0.0,
            "avg_prob": 0.0,
            "avg_edge": 0.0,
            "avg_odds": 0.0,
            "avg_clv_pct": 0.0,
            "invested": 0.0,
            "returned": 0.0,
            "profit": 0.0,
            "roi": 0.0,
        }

    invested = float(bets["stake"].sum())
    returned = float(bets["return"].sum())
    profit = float(bets["profit"].sum())
    bet_count = int(len(bets))
    wins = int(bets["won"].sum())
    avg_clv = float(bets["clv_decimal"].mean() * 100.0) if "clv_decimal" in bets and bets["clv_decimal"].notna().any() else 0.0
    return {
        "run_ts_utc": run_ts,
        "experiment_name": experiment_name,
        "report": report,
        "split": split,
        "model": model,
        "group_type": group_type,
        "group_value": group_value,
        "selector_status": selector_status,
        "edge_threshold": edge_threshold,
        "min_probability": min_probability,
        "min_odds": min_odds,
        "max_odds": max_odds,
        "matches": int(total_matches),
        "bets": bet_count,
        "coverage": round((bet_count / total_matches * 100.0) if total_matches else 0.0, 4),
        "wins": wins,
        "hit_rate": round((wins / bet_count * 100.0) if bet_count else 0.0, 4),
        "avg_prob": round(float(bets["prob_taken"].mean()), 6),
        "avg_edge": round(float(bets["best_ev"].mean()), 6),
        "avg_odds": round(float(bets["odds_taken"].mean()), 4),
        "avg_clv_pct": round(avg_clv, 4),
        "invested": round(invested, 6),
        "returned": round(returned, 6),
        "profit": round(profit, 6),
        "roi": round((profit / invested * 100.0) if invested > 0 else 0.0, 4),
    }


def _candidate_bets(
    probs: np.ndarray,
    raw_odds: np.ndarray,
    y_true: np.ndarray,
    match_info: list[dict],
    *,
    edge_threshold: float,
    min_probability: float,
    min_odds: float,
    max_odds: float,
) -> pd.DataFrame:
    bets = betting_records(
        probs,
        raw_odds,
        y_true,
        edge_threshold=edge_threshold,
        max_odds=max_odds,
        match_info=match_info,
    )
    if bets.empty:
        return bets
    return bets[
        (bets["prob_taken"] >= min_probability)
        & (bets["odds_taken"] >= min_odds)
        & (bets["odds_taken"] <= max_odds)
    ].copy()


def _chronological_fold_masks(match_info: list[dict], n_folds: int = 2) -> list[np.ndarray]:
    dates = pd.to_datetime([row["date"] for row in match_info])
    order = np.argsort(dates.to_numpy())
    masks = []
    for fold_indices in np.array_split(order, n_folds):
        mask = np.zeros(len(match_info), dtype=bool)
        mask[fold_indices] = True
        masks.append(mask)
    return masks


def _info_slice(match_info: list[dict], mask: np.ndarray) -> list[dict]:
    return [row for row, keep in zip(match_info, mask) if keep]


def _summarize_candidate(bets: pd.DataFrame, total_matches: int) -> dict:
    row = _metric_row(
        run_ts="",
        experiment_name="",
        report="",
        split="",
        model="",
        group_type="",
        group_value="",
        bets=bets,
        total_matches=total_matches,
    )
    return {
        "bets": row["bets"],
        "coverage": row["coverage"],
        "wins": row["wins"],
        "hit_rate": row["hit_rate"],
        "avg_prob": row["avg_prob"],
        "avg_edge": row["avg_edge"],
        "avg_odds": row["avg_odds"],
        "avg_clv_pct": row["avg_clv_pct"],
        "profit": row["profit"],
        "roi": row["roi"],
    }


def _add_bucket_rows(
    rows: list[dict],
    *,
    run_ts: str,
    experiment_name: str,
    split: str,
    model: str,
    bets: pd.DataFrame,
    total_matches: int,
    selector_status: str,
    edge_threshold: float,
    min_probability: float,
    min_odds: float,
    max_odds: float,
) -> None:
    base_kwargs = {
        "run_ts": run_ts,
        "experiment_name": experiment_name,
        "report": "validation_locked_selector",
        "split": split,
        "model": model,
        "total_matches": total_matches,
        "selector_status": selector_status,
        "edge_threshold": edge_threshold,
        "min_probability": min_probability,
        "min_odds": min_odds,
        "max_odds": max_odds,
    }
    rows.append(_metric_row(group_type="all", group_value="all", bets=bets, **base_kwargs))
    if bets.empty:
        return

    bucketed = bets.copy()
    bucketed["league"] = bucketed["league"].fillna("unknown")
    bucketed["odds_bucket"] = pd.cut(
        bucketed["odds_taken"],
        bins=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        labels=["1.00-1.50", "1.50-2.00", "2.00-3.00", "3.00-5.00", "5.00-10.00"],
        include_lowest=True,
    ).astype(str)
    bucketed["edge_bucket"] = pd.cut(
        bucketed["best_ev"],
        bins=[-np.inf, 0.025, 0.05, 0.075, 0.10, 0.15, np.inf],
        labels=["<=2.5%", "2.5-5%", "5-7.5%", "7.5-10%", "10-15%", ">15%"],
    ).astype(str)
    bucketed["confidence_bucket"] = pd.cut(
        bucketed["prob_taken"],
        bins=[0.0, 0.35, 0.45, 0.55, 0.65, 1.0],
        labels=["<=35%", "35-45%", "45-55%", "55-65%", ">65%"],
        include_lowest=True,
    ).astype(str)

    for group_type, column in [
        ("league", "league"),
        ("pick", "pred_label"),
        ("odds_bucket", "odds_bucket"),
        ("edge_bucket", "edge_bucket"),
        ("confidence_bucket", "confidence_bucket"),
    ]:
        for value, group in bucketed.groupby(column, dropna=False):
            rows.append(_metric_row(group_type=group_type, group_value=str(value), bets=group, **base_kwargs))


def write_probability_quality_report(
    config: ExperimentConfig,
    run_ts: str,
    split_probs: Mapping[str, Mapping[str, np.ndarray]],
    split_y: Mapping[str, np.ndarray],
) -> list[dict]:
    rows = []
    for split, probs_by_model in split_probs.items():
        y_true = split_y[split]
        for model, probs in probs_by_model.items():
            pred = np.argmax(probs, axis=1)
            class_metrics = class_metric_summary(probs, y_true)
            rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "split": split,
                "model": model,
                "matches": int(len(y_true)),
                "logloss": round(float(log_loss(y_true, probs)), 6),
                "brier": round(float(multiclass_brier(probs, y_true)), 6),
                "ece": round(float(top_label_ece(probs, y_true)), 6),
                "accuracy": round(float((pred == y_true).mean()), 6),
                "macro_f1": round(class_metrics["macro_f1"], 6),
                "home_precision": round(class_metrics["home_precision"], 6),
                "home_recall": round(class_metrics["home_recall"], 6),
                "home_f1": round(class_metrics["home_f1"], 6),
                "draw_precision": round(class_metrics["draw_precision"], 6),
                "draw_recall": round(class_metrics["draw_recall"], 6),
                "draw_f1": round(class_metrics["draw_f1"], 6),
                "away_precision": round(class_metrics["away_precision"], 6),
                "away_recall": round(class_metrics["away_recall"], 6),
                "away_f1": round(class_metrics["away_f1"], 6),
                "avg_confidence": round(float(np.max(probs, axis=1).mean()), 6),
                "draw_pick_rate": round(float((pred == 1).mean()), 6),
            })

    append_rows_to_csv(config.final_probability_quality_file, rows)
    if config.print_full_reports:
        print("\n" + "=" * 70)
        print("=== PROBABILITY QUALITY REPORT ===")
        print("=" * 70)
        print(pd.DataFrame(rows).sort_values(["split", "logloss"]).to_string(index=False))
    return rows


def write_validation_selected_betting_reports(
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
    min_validation_bets: int = 50,
    min_fold_bets: int = 10,
) -> list[dict]:
    selector_rows: list[dict] = []
    bucket_rows: list[dict] = []
    fold_masks = _chronological_fold_masks(validation_match_info, n_folds=2)

    for model, probs in validation_probs.items():
        candidates = []
        for edge_threshold in EDGE_THRESHOLDS:
            for min_probability in MIN_PROBABILITIES:
                for min_odds, max_odds in ODDS_BANDS:
                    val_bets = _candidate_bets(
                        probs,
                        validation_raw_odds,
                        validation_y,
                        validation_match_info,
                        edge_threshold=edge_threshold,
                        min_probability=min_probability,
                        min_odds=min_odds,
                        max_odds=max_odds,
                    )
                    val_stats = _summarize_candidate(val_bets, len(validation_y))
                    fold_stats = []
                    for mask in fold_masks:
                        fold_bets = _candidate_bets(
                            probs[mask],
                            validation_raw_odds[mask],
                            validation_y[mask],
                            _info_slice(validation_match_info, mask),
                            edge_threshold=edge_threshold,
                            min_probability=min_probability,
                            min_odds=min_odds,
                            max_odds=max_odds,
                        )
                        fold_stats.append(_summarize_candidate(fold_bets, int(mask.sum())))

                    eligible = (
                        val_stats["bets"] >= min_validation_bets
                        and val_stats["roi"] > 0.0
                        and all(item["bets"] >= min_fold_bets and item["roi"] > 0.0 for item in fold_stats)
                    )
                    candidates.append({
                        "edge_threshold": edge_threshold,
                        "min_probability": min_probability,
                        "min_odds": min_odds,
                        "max_odds": max_odds,
                        "eligible": eligible,
                        "fold_min_bets": min(item["bets"] for item in fold_stats),
                        "fold_min_roi": min(item["roi"] for item in fold_stats),
                        "fold_avg_roi": float(np.mean([item["roi"] for item in fold_stats])),
                        **{f"val_{key}": value for key, value in val_stats.items()},
                    })

        eligible_candidates = [row for row in candidates if row["eligible"]]
        if eligible_candidates:
            selected = sorted(
                eligible_candidates,
                key=lambda row: (-row["val_roi"], -row["val_profit"], -row["val_bets"]),
            )[0]
            selector_status = "validation_locked_positive_fold_roi"
        else:
            selected = next(
                row for row in candidates
                if row["edge_threshold"] == 0.05
                and row["min_probability"] == 0.0
                and row["min_odds"] == 1.01
                and row["max_odds"] == 10.0
            )
            selector_status = "fallback_not_robust_validation"

        test_bets = _candidate_bets(
            test_probs[model],
            test_raw_odds,
            test_y,
            test_match_info,
            edge_threshold=float(selected["edge_threshold"]),
            min_probability=float(selected["min_probability"]),
            min_odds=float(selected["min_odds"]),
            max_odds=float(selected["max_odds"]),
        )
        test_stats = _summarize_candidate(test_bets, len(test_y))

        selector_rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "model": model,
            "selector_status": selector_status,
            "edge_threshold": selected["edge_threshold"],
            "min_probability": selected["min_probability"],
            "min_odds": selected["min_odds"],
            "max_odds": selected["max_odds"],
            "eligible_candidates": len(eligible_candidates),
            "val_fold_min_bets": selected["fold_min_bets"],
            "val_fold_min_roi": round(float(selected["fold_min_roi"]), 4),
            "val_fold_avg_roi": round(float(selected["fold_avg_roi"]), 4),
            **{key: selected[f"val_{key}"] for key in ["bets", "coverage", "wins", "hit_rate", "avg_prob", "avg_edge", "avg_odds", "avg_clv_pct", "profit", "roi"]},
            **{f"test_{key}": value for key, value in test_stats.items()},
        })

        val_selected_bets = _candidate_bets(
            probs,
            validation_raw_odds,
            validation_y,
            validation_match_info,
            edge_threshold=float(selected["edge_threshold"]),
            min_probability=float(selected["min_probability"]),
            min_odds=float(selected["min_odds"]),
            max_odds=float(selected["max_odds"]),
        )
        for split, bets, total_matches in [
            ("validation", val_selected_bets, len(validation_y)),
            ("test", test_bets, len(test_y)),
        ]:
            _add_bucket_rows(
                bucket_rows,
                run_ts=run_ts,
                experiment_name=config.experiment_name,
                split=split,
                model=model,
                bets=bets,
                total_matches=total_matches,
                selector_status=selector_status,
                edge_threshold=float(selected["edge_threshold"]),
                min_probability=float(selected["min_probability"]),
                min_odds=float(selected["min_odds"]),
                max_odds=float(selected["max_odds"]),
            )

    append_rows_to_csv(config.final_bet_selector_file, selector_rows)
    append_rows_to_csv(config.final_bet_bucket_file, bucket_rows)
    if config.print_full_reports:
        print("\n" + "=" * 70)
        print("=== VALIDATION-LOCKED BET SELECTOR ===")
        print("=" * 70)
        print(
            pd.DataFrame(selector_rows)[
                [
                    "model",
                    "selector_status",
                    "edge_threshold",
                    "min_probability",
                    "min_odds",
                    "max_odds",
                    "bets",
                    "roi",
                    "test_bets",
                    "test_hit_rate",
                    "test_roi",
                ]
            ].sort_values("test_roi", ascending=False).to_string(index=False)
        )
    return selector_rows


def _outcome_goals(match_info: list[dict]) -> np.ndarray:
    goals = []
    for row in match_info:
        home = _finite_float(row.get("home_goals"))
        away = _finite_float(row.get("away_goals"))
        goals.append(home + away if np.isfinite(home) and np.isfinite(away) else np.nan)
    return np.array(goals, dtype=float)


def _poisson_over25_prob(total_xg: np.ndarray) -> np.ndarray:
    lam = np.clip(np.asarray(total_xg, dtype=float), 0.01, 8.0)
    under = np.exp(-lam) * (1.0 + lam + (lam**2 / 2.0))
    return np.clip(1.0 - under, 1e-6, 1.0 - 1e-6)


def _binary_value_bets(
    probs: np.ndarray,
    odds: np.ndarray,
    y_true: np.ndarray,
    match_info: list[dict],
    *,
    labels: tuple[str, str],
    edge_threshold: float = 0.05,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.05,
    max_odds: float = 10.0,
) -> pd.DataFrame:
    rows = []
    for i in range(len(probs)):
        p0, p1 = probs[i]
        o0, o1 = odds[i]
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(o0) and np.isfinite(o1)):
            continue
        evs = [(p0 * o0 - 1.0, 0, o0, p0), (p1 * o1 - 1.0, 1, o1, p1)]
        best_ev, choice, odds_taken, prob_taken = max(evs, key=lambda item: item[0])
        if best_ev <= edge_threshold or odds_taken > max_odds:
            continue
        b = odds_taken - 1.0
        stake = min((best_ev / b if b > 0 else 0.0) * kelly_fraction, max_stake)
        if stake < 0.001:
            continue
        won = int(choice == y_true[i])
        returned = stake * odds_taken if won else 0.0
        info = match_info[i]
        rows.append({
            "idx": i,
            "date": info.get("date"),
            "league": info.get("league"),
            "home_team": info.get("home_team"),
            "away_team": info.get("away_team"),
            "pred_label": labels[choice],
            "true_label": labels[int(y_true[i])],
            "prob_taken": float(prob_taken),
            "odds_taken": float(odds_taken),
            "best_ev": float(best_ev),
            "stake": float(stake),
            "return": float(returned),
            "profit": float(returned - stake),
            "won": won,
        })
    return pd.DataFrame(rows)


def _append_alt_summary(
    rows: list[dict],
    *,
    run_ts: str,
    experiment_name: str,
    split: str,
    market: str,
    model: str,
    price_available: bool,
    matches: int,
    picks: int,
    wins: int,
    pushes: int = 0,
    profit: float = 0.0,
    invested: float = 0.0,
    avg_odds: float = 0.0,
    note: str = "",
) -> None:
    resolved = max(picks - pushes, 0)
    rows.append({
        "run_ts_utc": run_ts,
        "experiment_name": experiment_name,
        "split": split,
        "market": market,
        "model": model,
        "price_available": int(price_available),
        "matches": int(matches),
        "picks": int(picks),
        "wins": int(wins),
        "pushes": int(pushes),
        "resolved_picks": int(resolved),
        "hit_rate": round((wins / resolved * 100.0) if resolved else 0.0, 4),
        "invested": round(float(invested), 6),
        "profit": round(float(profit), 6),
        "roi": round((profit / invested * 100.0) if invested > 0 else 0.0, 4),
        "avg_odds": round(float(avg_odds), 4),
        "note": note,
    })


def write_alternative_market_report(
    config: ExperimentConfig,
    run_ts: str,
    split_probs: Mapping[str, Mapping[str, np.ndarray]],
    split_X: Mapping[str, np.ndarray],
    split_match_info: Mapping[str, list[dict]],
) -> list[dict]:
    rows: list[dict] = []
    total_xg_idx = FEATURE_COLUMNS.index("total_xg")

    for split, probs_by_model in split_probs.items():
        match_info = split_match_info[split]
        goals = _outcome_goals(match_info)
        valid_goals = np.isfinite(goals)
        y_ou = (goals > 2.5).astype(int)
        matches = int(valid_goals.sum())

        for model, probs in probs_by_model.items():
            p = probs[valid_goals]
            y_class = np.array([
                0 if row["home_goals"] > row["away_goals"] else 1 if row["home_goals"] == row["away_goals"] else 2
                for row, keep in zip(match_info, valid_goals) if keep
            ], dtype=int)

            dc_probs = np.column_stack([p[:, 0] + p[:, 1], p[:, 0] + p[:, 2], p[:, 1] + p[:, 2]])
            dc_pick = np.argmax(dc_probs, axis=1)
            dc_wins = (
                ((dc_pick == 0) & np.isin(y_class, [0, 1]))
                | ((dc_pick == 1) & np.isin(y_class, [0, 2]))
                | ((dc_pick == 2) & np.isin(y_class, [1, 2]))
            )
            _append_alt_summary(
                rows,
                run_ts=run_ts,
                experiment_name=config.experiment_name,
                split=split,
                market="double_chance",
                model=model,
                price_available=False,
                matches=matches,
                picks=matches,
                wins=int(dc_wins.sum()),
                note="Accuracy-only audit; repo has no double-chance odds.",
            )

            non_draw = y_class != 1
            dnb_pick_home = p[:, 0] >= p[:, 2]
            dnb_wins = ((dnb_pick_home & (y_class == 0)) | (~dnb_pick_home & (y_class == 2))) & non_draw
            _append_alt_summary(
                rows,
                run_ts=run_ts,
                experiment_name=config.experiment_name,
                split=split,
                market="draw_no_bet",
                model=model,
                price_available=False,
                matches=matches,
                picks=matches,
                wins=int(dnb_wins.sum()),
                pushes=int((y_class == 1).sum()),
                note="Pushes on draws; accuracy-only audit because repo has no draw-no-bet odds.",
            )

        X = split_X[split][valid_goals]
        p_over = _poisson_over25_prob(X[:, total_xg_idx])
        ou_probs = np.column_stack([1.0 - p_over, p_over])
        ou_odds = np.array([
            [
                _finite_float(row.get("ou25_under_odds")),
                _finite_float(row.get("ou25_over_odds")),
            ]
            for row, keep in zip(match_info, valid_goals) if keep
        ], dtype=float)
        valid_ou_odds = np.isfinite(ou_odds).all(axis=1)
        ou_info = [row for row, keep in zip(match_info, valid_goals) if keep]
        ou_bets = _binary_value_bets(
            ou_probs[valid_ou_odds],
            ou_odds[valid_ou_odds],
            y_ou[valid_goals][valid_ou_odds],
            [row for row, keep in zip(ou_info, valid_ou_odds) if keep],
            labels=("Under2.5", "Over2.5"),
        )
        invested = float(ou_bets["stake"].sum()) if not ou_bets.empty else 0.0
        profit = float(ou_bets["profit"].sum()) if not ou_bets.empty else 0.0
        _append_alt_summary(
            rows,
            run_ts=run_ts,
            experiment_name=config.experiment_name,
            split=split,
            market="over_under_2_5",
            model="poisson_total_goals",
            price_available=True,
            matches=matches,
            picks=int(len(ou_bets)),
            wins=int(ou_bets["won"].sum()) if not ou_bets.empty else 0,
            invested=invested,
            profit=profit,
            avg_odds=float(ou_bets["odds_taken"].mean()) if not ou_bets.empty else 0.0,
            note="Uses base-model total expected goals and available O/U 2.5 odds.",
        )

        market_ou_prob = np.array([
            _finite_float(row.get("ou25_over_prob"))
            for row, keep in zip(match_info, valid_goals) if keep
        ], dtype=float)
        valid_market_ou = np.isfinite(market_ou_prob)
        market_ou_pick = market_ou_prob[valid_market_ou] >= 0.5
        _append_alt_summary(
            rows,
            run_ts=run_ts,
            experiment_name=config.experiment_name,
            split=split,
            market="over_under_2_5",
            model="market_probability",
            price_available=True,
            matches=matches,
            picks=int(valid_market_ou.sum()),
            wins=int((market_ou_pick == y_ou[valid_goals][valid_market_ou].astype(bool)).sum()),
            note="Market side accuracy from O/U 2.5 implied probability; no value filter.",
        )

        ah_lines = np.array([
            _finite_float(row.get("ah_line"))
            for row, keep in zip(match_info, valid_goals) if keep
        ], dtype=float)
        _append_alt_summary(
            rows,
            run_ts=run_ts,
            experiment_name=config.experiment_name,
            split=split,
            market="asian_handicap",
            model="coverage_only",
            price_available=False,
            matches=matches,
            picks=int(np.isfinite(ah_lines).sum()),
            wins=0,
            note="Asian handicap line exists for coverage, but handicap odds are not loaded.",
        )

    append_rows_to_csv(config.final_alternative_markets_file, rows)
    if config.print_full_reports:
        print("\n" + "=" * 70)
        print("=== ALTERNATIVE MARKET AUDIT ===")
        print("=" * 70)
        print(pd.DataFrame(rows).sort_values(["split", "market", "model"]).to_string(index=False))
    return rows


def write_data_enrichment_audit(
    config: ExperimentConfig,
    run_ts: str,
    split_match_info: Mapping[str, list[dict]],
) -> list[dict]:
    rows = []
    odds_triplets = {
        "opening_1x2_odds": ("open_odds_home", "open_odds_draw", "open_odds_away"),
        "closing_1x2_odds": ("close_odds_home", "close_odds_draw", "close_odds_away"),
        "ou25_odds": ("ou25_over_odds", "ou25_under_odds"),
    }
    for split, match_info in split_match_info.items():
        for feature, cols in odds_triplets.items():
            available = 0
            for row in match_info:
                if all(np.isfinite(_finite_float(row.get(col))) for col in cols):
                    available += 1
            rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "split": split,
                "data_group": feature,
                "matches": len(match_info),
                "available_rows": available,
                "coverage": round((available / len(match_info) * 100.0) if match_info else 0.0, 4),
                "status": "available",
                "note": "Loaded from football-data columns.",
            })

        ah_available = sum(np.isfinite(_finite_float(row.get("ah_line"))) for row in match_info)
        rows.append({
            "run_ts_utc": run_ts,
            "experiment_name": config.experiment_name,
            "split": split,
            "data_group": "asian_handicap_line",
            "matches": len(match_info),
            "available_rows": int(ah_available),
            "coverage": round((ah_available / len(match_info) * 100.0) if match_info else 0.0, 4),
            "status": "partial",
            "note": "Line is loaded; handicap odds are not currently loaded.",
        })

        external_groups = {
            "confirmed_lineups": ("lineup_available", ("home_lineup_strength", "away_lineup_strength")),
            "injuries": ("team_news_available", ("home_injury_count", "away_injury_count")),
            "suspensions": ("team_news_available", ("home_suspension_count", "away_suspension_count")),
            "absences": ("team_news_available", ("home_absence_count", "away_absence_count")),
            "key_absences": ("team_news_available", ("home_key_absence_count", "away_key_absence_count")),
            "manager_changes": ("team_news_available", ("home_manager_change_recent", "away_manager_change_recent")),
            "weather": ("weather_available", ("temperature_c", "wind_kph", "precipitation_mm")),
        }
        for data_group, (flag_col, value_cols) in external_groups.items():
            available = 0
            for row in match_info:
                flag_available = flag_col in row and np.isfinite(_finite_float(row.get(flag_col))) and _finite_float(row.get(flag_col)) > 0
                values_available = any(np.isfinite(_finite_float(row.get(col))) for col in value_cols)
                if flag_available or values_available:
                    available += 1
            rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "split": split,
                "data_group": data_group,
                "matches": len(match_info),
                "available_rows": int(available),
                "coverage": round((available / len(match_info) * 100.0) if match_info else 0.0, 4),
                "status": "available" if available > 0 else "not_in_current_dataset",
                "note": "Loaded from data/external/match_context.csv." if available > 0 else "Add pre-match rows to data/external/match_context.csv.",
            })

        for missing_group in ["live_odds_snapshots"]:
            rows.append({
                "run_ts_utc": run_ts,
                "experiment_name": config.experiment_name,
                "split": split,
                "data_group": missing_group,
                "matches": len(match_info),
                "available_rows": 0,
                "coverage": 0.0,
                "status": "not_in_current_dataset",
                "note": "Needs an external source before it can be modeled honestly.",
            })

    append_rows_to_csv(config.final_data_enrichment_file, rows)
    if config.print_full_reports:
        print("\n" + "=" * 70)
        print("=== DATA ENRICHMENT AUDIT ===")
        print("=" * 70)
        print(pd.DataFrame(rows).to_string(index=False))
    return rows
