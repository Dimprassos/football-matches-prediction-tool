from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency path
    XGBClassifier = None

from src.config import DEFAULT_CONFIG
from src.data_loader import load_league_data
from src.elo import expected_score, margin_multiplier, match_result
from src.evaluation import simulate_value_betting
from src.feature_builder import market_probs_from_odds_row
from src.metrics import multiclass_brier, top_label_ece
from src.calibration import safe_logit


CLASS_LABELS = ["H", "D", "A"]
ELO_K = 40.0
ELO_HOME_ADV = 60.0
ROLLING_WINDOW = 5

PUBLIC_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_home",
    "rest_away",
    "rest_diff",
    "form_home_5",
    "form_away_5",
    "form_diff_5",
    "goals_for_home_5",
    "goals_for_away_5",
    "goals_for_diff_5",
    "goals_against_home_5",
    "goals_against_away_5",
    "goals_against_diff_5",
    "shots_for_home_5",
    "shots_for_away_5",
    "shots_for_diff_5",
    "shots_against_home_5",
    "shots_against_away_5",
    "shots_against_diff_5",
    "sot_for_home_5",
    "sot_for_away_5",
    "sot_for_diff_5",
    "sot_against_home_5",
    "sot_against_away_5",
    "sot_against_diff_5",
    "corners_for_home_5",
    "corners_for_away_5",
    "corners_for_diff_5",
    "corners_against_home_5",
    "corners_against_away_5",
    "corners_against_diff_5",
    "cards_home_5",
    "cards_away_5",
    "cards_diff_5",
    "matches_seen_home",
    "matches_seen_away",
    "matches_seen_diff",
]

OPENING_MARKET_FEATURE_COLUMNS = [
    "open_prob_home",
    "open_prob_draw",
    "open_prob_away",
    "open_logit_home",
    "open_logit_draw",
    "open_logit_away",
]


def season_window(season_start: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(f"{season_start - 1}-07-01"),
        pd.Timestamp(f"{season_start}-07-01"),
        pd.Timestamp(f"{season_start + 1}-07-01"),
    )


def _as_float(value, default: float = np.nan) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if np.isfinite(out) else float(default)


def _cards(yellows, reds) -> float:
    yellow_value = _as_float(yellows, 0.0)
    red_value = _as_float(reds, 0.0)
    return yellow_value + 2.0 * red_value


def _team_match_record(row: pd.Series, team: str) -> dict:
    is_home = row["home_team"] == team
    prefix = "home" if is_home else "away"
    opp_prefix = "away" if is_home else "home"
    goals_for = _as_float(row[f"{prefix}_goals"])
    goals_against = _as_float(row[f"{opp_prefix}_goals"])
    if goals_for > goals_against:
        points = 3.0
    elif goals_for == goals_against:
        points = 1.0
    else:
        points = 0.0

    return {
        "points": points / 3.0,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "shots_for": _as_float(row.get(f"{prefix}_shots")),
        "shots_against": _as_float(row.get(f"{opp_prefix}_shots")),
        "sot_for": _as_float(row.get(f"{prefix}_shots_target")),
        "sot_against": _as_float(row.get(f"{opp_prefix}_shots_target")),
        "corners_for": _as_float(row.get(f"{prefix}_corners")),
        "corners_against": _as_float(row.get(f"{opp_prefix}_corners")),
        "cards": _cards(row.get(f"{prefix}_yellows"), row.get(f"{prefix}_reds")),
    }


def _recent_summary(history: list[dict], window: int = ROLLING_WINDOW) -> dict:
    recent = history[-window:]
    keys = [
        "points",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "corners_for",
        "corners_against",
        "cards",
    ]
    out = {"matches_seen": float(len(history))}
    for key in keys:
        values = np.array([row.get(key, np.nan) for row in recent], dtype=float)
        finite = values[np.isfinite(values)]
        out[key] = float(finite.mean()) if len(finite) else 0.0
    return out


def _rest_days(team: str, date: pd.Timestamp, last_seen: dict[str, pd.Timestamp]) -> float:
    last_date = last_seen.get(team)
    if last_date is None:
        return 1.0
    days = min(max(0, int((date - last_date).days)), 21)
    return float(days) / 7.0


def _outcome(row: pd.Series) -> int:
    if row["home_goals"] > row["away_goals"]:
        return 0
    if row["home_goals"] == row["away_goals"]:
        return 1
    return 2


def build_feature_frame(leagues: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    for league in leagues:
        df = load_league_data(league).sort_values("date").reset_index(drop=True)
        df = df[df["is_played"] == True].copy()

        ratings: dict[str, float] = {}
        history: dict[str, list[dict]] = {}
        last_seen: dict[str, pd.Timestamp] = {}

        for _, row in df.iterrows():
            date = pd.Timestamp(row["date"])
            home = row["home_team"]
            away = row["away_team"]
            ratings.setdefault(home, 1500.0)
            ratings.setdefault(away, 1500.0)
            history.setdefault(home, [])
            history.setdefault(away, [])

            home_recent = _recent_summary(history[home])
            away_recent = _recent_summary(history[away])
            rest_home = _rest_days(home, date, last_seen)
            rest_away = _rest_days(away, date, last_seen)
            open_probs = market_probs_from_odds_row(
                row.get("open_odds_home", np.nan),
                row.get("open_odds_draw", np.nan),
                row.get("open_odds_away", np.nan),
            )
            close_probs = market_probs_from_odds_row(
                row.get("close_odds_home", np.nan),
                row.get("close_odds_draw", np.nan),
                row.get("close_odds_away", np.nan),
            )

            record = {
                "date": date,
                "league": league,
                "home_team": home,
                "away_team": away,
                "y": _outcome(row),
                "open_odds_home": _as_float(row.get("open_odds_home")),
                "open_odds_draw": _as_float(row.get("open_odds_draw")),
                "open_odds_away": _as_float(row.get("open_odds_away")),
                "close_odds_home": _as_float(row.get("close_odds_home")),
                "close_odds_draw": _as_float(row.get("close_odds_draw")),
                "close_odds_away": _as_float(row.get("close_odds_away")),
                "open_prob_home": open_probs[0],
                "open_prob_draw": open_probs[1],
                "open_prob_away": open_probs[2],
                "open_logit_home": safe_logit(open_probs[0]) if np.isfinite(open_probs[0]) else np.nan,
                "open_logit_draw": safe_logit(open_probs[1]) if np.isfinite(open_probs[1]) else np.nan,
                "open_logit_away": safe_logit(open_probs[2]) if np.isfinite(open_probs[2]) else np.nan,
                "close_prob_home": close_probs[0],
                "close_prob_draw": close_probs[1],
                "close_prob_away": close_probs[2],
                "elo_diff": (ratings[home] + ELO_HOME_ADV - ratings[away]) / 400.0,
                "rest_home": rest_home,
                "rest_away": rest_away,
                "rest_diff": rest_home - rest_away,
                "matches_seen_home": home_recent["matches_seen"],
                "matches_seen_away": away_recent["matches_seen"],
                "matches_seen_diff": home_recent["matches_seen"] - away_recent["matches_seen"],
            }

            for source_key, feature_stem in [
                ("points", "form"),
                ("goals_for", "goals_for"),
                ("goals_against", "goals_against"),
                ("shots_for", "shots_for"),
                ("shots_against", "shots_against"),
                ("sot_for", "sot_for"),
                ("sot_against", "sot_against"),
                ("corners_for", "corners_for"),
                ("corners_against", "corners_against"),
                ("cards", "cards"),
            ]:
                home_value = home_recent[source_key]
                away_value = away_recent[source_key]
                record[f"{feature_stem}_home_5"] = home_value
                record[f"{feature_stem}_away_5"] = away_value
                record[f"{feature_stem}_diff_5"] = home_value - away_value

            rows.append(record)

            r_home = ratings[home]
            r_away = ratings[away]
            exp_home = expected_score(r_home + ELO_HOME_ADV, r_away)
            score_home, score_away = match_result(int(row["home_goals"]), int(row["away_goals"]))
            mult = margin_multiplier(int(row["home_goals"]) - int(row["away_goals"]))
            ratings[home] = r_home + (ELO_K * mult) * (score_home - exp_home)
            ratings[away] = r_away + (ELO_K * mult) * (score_away - (1.0 - exp_home))

            history[home].append(_team_match_record(row, home))
            history[away].append(_team_match_record(row, away))
            last_seen[home] = date
            last_seen[away] = date

    return pd.DataFrame(rows).sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)


def _probability_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "logloss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier": float(multiclass_brier(probs, y_true)),
        "ece": float(top_label_ece(probs, y_true)),
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "home_recall": float(recall[0]),
        "draw_recall": float(recall[1]),
        "away_recall": float(recall[2]),
        "home_precision": float(precision[0]),
        "draw_precision": float(precision[1]),
        "away_precision": float(precision[2]),
        "draw_pick_rate": float((pred == 1).mean()),
    }


def _aligned_proba(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.full((len(X), 3), 1e-9, dtype=float)
    for idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, idx]
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _make_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logreg_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "rf_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    }
    if XGBClassifier is not None:
        models["xgb_default"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    return models


def run_audit(season_start: int, output_dir: Path = Path("artifacts")) -> pd.DataFrame:
    train_cut, test_cut, test_end = season_window(season_start)
    df = build_feature_frame(DEFAULT_CONFIG.leagues)
    train_mask = df["date"] < test_cut
    test_mask = (df["date"] >= test_cut) & (df["date"] < test_end)
    train = df[train_mask].copy()
    test = df[test_mask].copy()

    y_train = train["y"].to_numpy(dtype=int)
    y_test = test["y"].to_numpy(dtype=int)
    open_probs = test[["open_prob_home", "open_prob_draw", "open_prob_away"]].to_numpy(dtype=float)
    close_probs = test[["close_prob_home", "close_prob_draw", "close_prob_away"]].to_numpy(dtype=float)
    open_odds = test[["open_odds_home", "open_odds_draw", "open_odds_away"]].to_numpy(dtype=float)
    valid_market = np.isfinite(open_probs).all(axis=1) & np.isfinite(close_probs).all(axis=1)

    rows: list[dict] = []
    for model_name, probs, feature_set in [
        ("opening_market", open_probs, "opening_odds_only"),
        ("closing_market", close_probs, "closing_odds_benchmark"),
    ]:
        mask = valid_market & np.isfinite(probs).all(axis=1)
        metrics = _probability_metrics(y_test[mask], probs[mask])
        if model_name == "closing_market":
            bets, wins, profit, roi, avg_odds = 0, 0, np.nan, np.nan, np.nan
        else:
            bets, wins, profit, roi, avg_odds = simulate_value_betting(
                probs[mask],
                open_odds[mask],
                y_test[mask],
                edge_threshold=0.05,
                verbose=False,
            )
        rows.append({
            "season": f"{season_start}-{season_start + 1}",
            "model": model_name,
            "feature_set": feature_set,
            "train_matches": int(len(train)),
            "test_matches": int(mask.sum()),
            **metrics,
            "bets": int(bets),
            "hit_rate": float((wins / bets * 100.0) if bets else 0.0),
            "roi": float(roi),
            "profit": float(profit),
            "avg_odds": float(avg_odds),
            "paper_basis": (
                "paper7_closing_market_benchmark_not_bettable"
                if model_name == "closing_market"
                else "paper7_opening_market_benchmark"
            ),
        })

    feature_sets = {
        "public_pre_match": PUBLIC_FEATURE_COLUMNS,
        "opening_market_plus_public": OPENING_MARKET_FEATURE_COLUMNS + PUBLIC_FEATURE_COLUMNS,
    }
    models = _make_models(DEFAULT_CONFIG.random_state)
    for feature_set_name, columns in feature_sets.items():
        X_train = train[columns].to_numpy(dtype=float)
        X_test = test[columns].to_numpy(dtype=float)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            probs = _aligned_proba(model, X_test)
            metrics = _probability_metrics(y_test, probs)
            bets, wins, profit, roi, avg_odds = simulate_value_betting(
                probs,
                open_odds,
                y_test,
                edge_threshold=0.05,
                verbose=False,
            )
            rows.append({
                "season": f"{season_start}-{season_start + 1}",
                "model": model_name,
                "feature_set": feature_set_name,
                "train_matches": int(len(train)),
                "test_matches": int(len(test)),
                **metrics,
                "bets": int(bets),
                "hit_rate": float((wins / bets * 100.0) if bets else 0.0),
                "roi": float(roi),
                "profit": float(profit),
                "avg_odds": float(avg_odds),
                "paper_basis": "paper7_public_features_papaer1_form_fatigue",
            })

    out = pd.DataFrame(rows)
    float_cols = [
        "logloss",
        "brier",
        "ece",
        "accuracy",
        "macro_f1",
        "home_recall",
        "draw_recall",
        "away_recall",
        "home_precision",
        "draw_precision",
        "away_precision",
        "draw_pick_rate",
        "hit_rate",
        "roi",
        "profit",
        "avg_odds",
    ]
    for col in float_cols:
        out[col] = out[col].astype(float).round(6)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"literature_fast_audit_{season_start}_{season_start + 1}.csv"
    out.sort_values(["logloss", "model", "feature_set"]).to_csv(csv_path, index=False)

    notes_path = output_dir / f"literature_fast_audit_{season_start}_{season_start + 1}.md"
    best = out.sort_values("logloss").iloc[0]
    opening = out[out["model"] == "opening_market"].iloc[0]
    notes_path.write_text(
        "\n".join([
            f"# Literature Fast Audit {season_start}-{season_start + 1}",
            "",
            "Purpose: quick, non-retuned audit based on the papers before changing the main pipeline.",
            "",
            "Paper basis:",
            "- paper7: time-based 1X2 evaluation, market benchmark, draw difficulty, value betting as secondary diagnostic.",
            "- papaer1: form, fatigue/rest, momentum-style pre-match features.",
            "- paper2/papaer3: richer event/spatial features are useful, but not available in the current free dataset.",
            "",
            f"Opening market logloss: {opening['logloss']}",
            f"Best audit row: {best['model']} / {best['feature_set']} / logloss {best['logloss']}",
            "",
            "Interpretation rule: a model is useful only if it beats opening_market on logloss/Brier/calibration before ROI is considered.",
            "Closing market is reported as a benchmark only; it is not a valid opening-time betting strategy.",
        ]),
        encoding="utf-8",
    )
    print(out.sort_values("logloss").to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {notes_path}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast literature-grounded audit without full retuning.")
    parser.add_argument("--season", type=int, default=2024, help="Season start year, e.g. 2024 for 2024-2025.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(args.season)


if __name__ == "__main__":
    main()
