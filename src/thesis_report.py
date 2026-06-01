from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = "final_opening_market_pre_match"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _latest_run(df: pd.DataFrame) -> pd.DataFrame:
    if "run_ts_utc" not in df.columns or df.empty:
        return df.copy()
    latest = df["run_ts_utc"].max()
    return df[df["run_ts_utc"] == latest].copy()


def _num(value, default=np.nan) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if np.isfinite(out) else default


def _fmt(value, digits: int = 4) -> str:
    value = _num(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _fmt_pct(value, digits: int = 2) -> str:
    value = _num(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}%"


def _markdown_table(rows: list[dict], columns: Iterable[str], headers: Iterable[str] | None = None) -> str:
    columns = list(columns)
    headers = list(headers) if headers is not None else columns
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        table.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(table)


def _model_rows(model_summary: pd.DataFrame) -> list[dict]:
    rows = []
    ordered = model_summary.sort_values("logloss", key=lambda s: pd.to_numeric(s, errors="coerce"))
    for row in ordered.to_dict("records"):
        rows.append({
            "model": row["model"],
            "logloss": _fmt(row["logloss"], 4),
            "accuracy": _fmt(_num(row["accuracy"]) * 100.0, 2) + "%",
            "macro_f1": _fmt(row["macro_f1"], 4),
            "draw_recall": _fmt(_num(row["draw_recall"]) * 100.0, 2) + "%",
            "bets": int(_num(row["bets"], 0)),
            "roi": _fmt_pct(row["roi"], 2),
        })
    return rows


def _multi_season_rows(multi: pd.DataFrame) -> list[dict]:
    rows = []
    for row in multi.sort_values("season").to_dict("records"):
        rows.append({
            "season": row["season"],
            "opening_ll": _fmt(row["opening_logloss"], 4),
            "best_ml": f"{row['best_ml_model']} / {row['best_ml_feature_set']}",
            "best_ml_ll": _fmt(row["best_ml_logloss"], 4),
            "delta": f"{_num(row['best_ml_minus_opening_logloss']):+.4f}",
            "draw_recall": _fmt(_num(row["best_ml_draw_recall"]) * 100.0, 2) + "%",
            "roi": _fmt_pct(row["best_ml_roi"], 2),
        })
    return rows


def _draw_tradeoff_rows(model_summary: pd.DataFrame) -> list[dict]:
    preferred = ["market", "logreg", "meta", "mlp"]
    rows = []
    df = model_summary[model_summary["model"].isin(preferred)].copy()
    df["_order"] = df["model"].map({name: idx for idx, name in enumerate(preferred)})
    for row in df.sort_values("_order").to_dict("records"):
        rows.append({
            "model": row["model"],
            "logloss": _fmt(row["logloss"], 4),
            "macro_f1": _fmt(row["macro_f1"], 4),
            "draw_recall": _fmt(_num(row["draw_recall"]) * 100.0, 2) + "%",
            "draw_precision": _fmt(_num(row["draw_precision"]) * 100.0, 2) + "%",
            "roi": _fmt_pct(row["roi"], 2),
        })
    return rows


def _ablation_rows(ablation: pd.DataFrame) -> list[dict]:
    useful_sets = {
        "market_only",
        "market_plus_context",
        "no_market",
        "core_18",
        "no_understat_xg",
        "no_external_context",
    }
    df = ablation[ablation["feature_set"].isin(useful_sets)].copy()
    rows = []
    for row in df.sort_values(["model", "logloss"], key=lambda s: s if s.name == "model" else pd.to_numeric(s, errors="coerce")).to_dict("records"):
        rows.append({
            "model": row["model"],
            "feature_set": row["feature_set"],
            "logloss": _fmt(row["logloss"], 4),
            "accuracy": _fmt(_num(row["accuracy"]) * 100.0, 2) + "%",
            "macro_f1": _fmt(row["macro_f1"], 4),
            "draw_recall": _fmt(_num(row["draw_recall"]) * 100.0, 2) + "%",
        })
    return rows


def _coverage_rows(data_audit: pd.DataFrame) -> list[dict]:
    groups = [
        "opening_1x2_odds",
        "closing_1x2_odds",
        "confirmed_lineups",
        "injuries",
        "suspensions",
        "manager_changes",
        "weather",
        "live_odds_snapshots",
    ]
    test = data_audit[data_audit["split"] == "test"].copy()
    rows = []
    for group in groups:
        item = test[test["data_group"] == group]
        if item.empty:
            continue
        row = item.iloc[0].to_dict()
        rows.append({
            "data_group": group,
            "coverage": f"{int(_num(row['available_rows'], 0))}/{int(_num(row['matches'], 0))} ({_fmt_pct(row['coverage'], 1)})",
            "status": row["status"],
        })
    return rows


def _robustness_rows(robustness: pd.DataFrame) -> list[dict]:
    overall = robustness[robustness["group_type"] == "all"].copy()
    rows = []
    for row in overall.sort_values("roi", key=lambda s: pd.to_numeric(s, errors="coerce"), ascending=False).to_dict("records"):
        rows.append({
            "model": row["model"],
            "bets": int(_num(row["bets"], 0)),
            "hit_rate": _fmt_pct(row["hit_rate"], 2),
            "profit": _fmt(row["profit"], 4),
            "roi": _fmt_pct(row["roi"], 2),
        })
    return rows


def build_report(artifacts_dir: Path, experiment_name: str) -> tuple[str, pd.DataFrame]:
    model_summary = _latest_run(_read_csv(artifacts_dir / f"final_model_summary_{experiment_name}.csv"))
    probability_quality = _latest_run(_read_csv(artifacts_dir / f"final_probability_quality_{experiment_name}.csv"))
    ablation = _latest_run(_read_csv(artifacts_dir / f"final_ablation_summary_{experiment_name}.csv"))
    robustness = _latest_run(_read_csv(artifacts_dir / f"final_betting_robustness_{experiment_name}.csv"))
    data_audit = _latest_run(_read_csv(artifacts_dir / f"final_data_enrichment_audit_{experiment_name}.csv"))
    multi = _read_csv(artifacts_dir / "literature_multi_season_summary_2021_2024.csv")

    model_summary["logloss"] = pd.to_numeric(model_summary["logloss"], errors="coerce")
    market = model_summary[model_summary["model"] == "market"].iloc[0]
    best_model = model_summary.sort_values("logloss").iloc[0]
    multi_beats = int((pd.to_numeric(multi["best_ml_minus_opening_logloss"], errors="coerce") < 0).sum())
    avg_delta = float(pd.to_numeric(multi["best_ml_minus_opening_logloss"], errors="coerce").mean())

    key_findings = pd.DataFrame([
        {
            "finding": "multi_season_best_ml_beats_opening_market",
            "value": f"{multi_beats}/{len(multi)} seasons",
        },
        {
            "finding": "multi_season_avg_best_ml_minus_opening_logloss",
            "value": f"{avg_delta:+.6f}",
        },
        {
            "finding": "main_best_probability_model",
            "value": f"{best_model['model']} logloss={best_model['logloss']:.6f}",
        },
        {
            "finding": "main_opening_market_logloss",
            "value": f"{_num(market['logloss']):.6f}",
        },
        {
            "finding": "recommended_betting_interpretation",
            "value": "no_bet; ML strategies do not show robust positive edge",
        },
    ])

    generated = datetime.now(UTC).isoformat()
    lines = [
        f"# Thesis Results Summary: {experiment_name}",
        "",
        f"Generated: {generated}",
        "",
        "## Methodological Frame",
        "",
        "This summary follows the literature-driven setup used in the project:",
        "",
        "- paper7: time-based 1X2 evaluation, market benchmark, draw difficulty, and ROI as a secondary diagnostic.",
        "- papaer1: public pre-match form, rest/fatigue, momentum, and contextual features.",
        "- paper2 and papaer3: richer event/spatial/player data can improve performance, but those data are not present in the current football-data dataset.",
        "- paper8: very high reported accuracy is treated as non-comparable unless leakage/target/split assumptions are clear.",
        "",
        "The deployable betting benchmark is the opening market. Closing market is used only as a non-bettable reference.",
        "",
        "## Key Findings",
        "",
        f"- Across the completed-season fast audit, best ML beat the opening market in **{multi_beats}/{len(multi)}** seasons.",
        f"- Average best-ML minus opening-market logloss was **{avg_delta:+.6f}**; positive means worse than market.",
        f"- In the main pipeline, the best probability model was **{best_model['model']}** with logloss **{best_model['logloss']:.4f}**.",
        f"- Opening market logloss in the main pipeline was **{_num(market['logloss']):.4f}**.",
        "- Betting recommendation remains **no bet** because positive ROI is not robust after probability-quality checks.",
        "",
        "## Multi-Season Opening-Market Audit",
        "",
        _markdown_table(
            _multi_season_rows(multi),
            ["season", "opening_ll", "best_ml", "best_ml_ll", "delta", "draw_recall", "roi"],
            ["Season", "Opening LL", "Best ML", "Best ML LL", "Delta", "Draw Recall", "ROI"],
        ),
        "",
        "## Main Pipeline Model Quality",
        "",
        _markdown_table(
            _model_rows(model_summary),
            ["model", "logloss", "accuracy", "macro_f1", "draw_recall", "bets", "roi"],
            ["Model", "LogLoss", "Accuracy", "Macro F1", "Draw Recall", "Bets", "ROI"],
        ),
        "",
        "## Draw Tradeoff",
        "",
        "Balanced models can predict more draws, but this comes with worse logloss/calibration. This matches the draw difficulty discussed in paper7.",
        "",
        _markdown_table(
            _draw_tradeoff_rows(model_summary),
            ["model", "logloss", "macro_f1", "draw_recall", "draw_precision", "roi"],
            ["Model", "LogLoss", "Macro F1", "Draw Recall", "Draw Precision", "ROI"],
        ),
        "",
        "## Feature Ablation Snapshot",
        "",
        _markdown_table(
            _ablation_rows(ablation),
            ["model", "feature_set", "logloss", "accuracy", "macro_f1", "draw_recall"],
            ["Model", "Feature Set", "LogLoss", "Accuracy", "Macro F1", "Draw Recall"],
        ),
        "",
        "## Betting Robustness",
        "",
        _markdown_table(
            _robustness_rows(robustness),
            ["model", "bets", "hit_rate", "profit", "roi"],
            ["Model", "Bets", "Hit Rate", "Profit", "ROI"],
        ),
        "",
        "## Data Coverage",
        "",
        _markdown_table(
            _coverage_rows(data_audit),
            ["data_group", "coverage", "status"],
            ["Data Group", "Test Coverage", "Status"],
        ),
        "",
        "## Thesis Interpretation",
        "",
        "The current results support a realistic negative finding rather than a profitable-bot claim. With public pre-match data and opening odds, ML models mostly learn the market and do not consistently improve logloss/Brier/ROI. Rolling defensive features and class balancing can change class behavior, especially draw recall, but they do not create a stable probability edge. Stronger performance likely requires richer event, spatial, lineup, or player-level data, consistent with paper2 and papaer3.",
        "",
    ]

    return "\n".join(lines), key_findings


def write_report(artifacts_dir: Path, experiment_name: str) -> tuple[Path, Path]:
    text, key_findings = build_report(artifacts_dir, experiment_name)
    md_path = artifacts_dir / f"thesis_results_summary_{experiment_name}.md"
    csv_path = artifacts_dir / f"thesis_key_findings_{experiment_name}.csv"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    key_findings.to_csv(csv_path, index=False)
    return md_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis-ready result summary from generated artifacts.")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_path, csv_path = write_report(args.artifacts_dir, args.experiment_name)
    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
