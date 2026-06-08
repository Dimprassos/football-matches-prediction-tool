"""Streamlit tool for predicting football match results.

Thin UI layer over the trained pipeline. It loads the cached artifacts of the
`final_opening_market_pre_match` experiment (opening odds, leakage-safe) and
exposes two things the assignment asks for:

  * an interactive 1X2 predictor where the user picks the league, the two teams,
    the odds and *which model* to trust (configurability + confidence level);
  * a model-selection screen that shows the stored evaluation metrics so the
    user can decide which model to use.

The interface is bilingual (Greek / English); the language is chosen in the
sidebar and defaults to Greek. All user-facing strings live in the translation
tables below so the two languages stay in sync.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CONTEXT_AWARE_CONFIG, FINAL_CONFIG
from src.predictor import (
    get_league_runtime_state,
    load_runtime_artifacts,
    predict_custom_match,
)

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
USER_FILE = "zz_user_added.csv"  # one per league; appended rows the loader auto-merges
CANONICAL_EXP = FINAL_CONFIG.experiment_name

LEAGUES = ["england", "spain", "italy", "germany", "france"]
OUTCOME_SHORT = ["1", "X", "2"]

# --------------------------------------------------------------------------- #
# i18n: language code -> display name, and the current-language helper         #
# --------------------------------------------------------------------------- #
LANGUAGES = {"el": "Ελληνικά", "en": "English"}
DEFAULT_LANG = "el"


def _lang() -> str:
    """Current UI language code (set by the sidebar selector; defaults to Greek)."""
    return st.session_state.get("lang", DEFAULT_LANG)


# Language-dependent display labels (keys are stable; only the text is translated).
LEAGUE_LABELS = {
    "el": {
        "england": "Αγγλία (Premier League)",
        "spain": "Ισπανία (La Liga)",
        "italy": "Ιταλία (Serie A)",
        "germany": "Γερμανία (Bundesliga)",
        "france": "Γαλλία (Ligue 1)",
    },
    "en": {
        "england": "England (Premier League)",
        "spain": "Spain (La Liga)",
        "italy": "Italy (Serie A)",
        "germany": "Germany (Bundesliga)",
        "france": "France (Ligue 1)",
    },
}

# user-facing label -> key returned by predict_custom_match / used in eval CSVs
MODEL_LABELS = {
    "el": {
        "base": "Βάση — Elo + Poisson/Dixon-Coles",
        "market": "Αγορά — opening odds",
        "market_corr": "Διορθωμένη Αγορά — market + model",
        "meta": "XGBoost",
        "logreg": "Logistic Regression",
        "mlp": "Νευρωνικό Δίκτυο (MLP)",
        "ensemble": "Ensemble (blend)",
    },
    "en": {
        "base": "Base — Elo + Poisson/Dixon-Coles",
        "market": "Market — opening odds",
        "market_corr": "Corrected market — market + model",
        "meta": "XGBoost",
        "logreg": "Logistic Regression",
        "mlp": "Neural network (MLP)",
        "ensemble": "Ensemble (blend)",
    },
}

OUTCOME_LABELS = {
    "el": ["Νίκη Γηπεδούχου (1)", "Ισοπαλία (X)", "Νίκη Φιλοξενούμενης (2)"],
    "en": ["Home win (1)", "Draw (X)", "Away win (2)"],
}

CONFIDENCE_LABELS = {
    "el": {"high": "Υψηλή", "medium": "Μέτρια", "low": "Χαμηλή"},
    "en": {"high": "High", "medium": "Medium", "low": "Low"},
}

# Prediction experiments the user can pick between. The canonical model is
# market-dominated; the context-aware variant is forced to use understat/form
# features so the reconstructed pre-match context actually affects the prediction.
PREDICT_EXPERIMENTS = {
    FINAL_CONFIG.experiment_name: FINAL_CONFIG,
    CONTEXT_AWARE_CONFIG.experiment_name: CONTEXT_AWARE_CONFIG,
}
EXPERIMENT_LABELS = {
    "el": {
        FINAL_CONFIG.experiment_name: "Αγορά (canonical)",
        CONTEXT_AWARE_CONFIG.experiment_name: "Πλούσιο σε features (understat/form)",
    },
    "en": {
        FINAL_CONFIG.experiment_name: "Market (canonical)",
        CONTEXT_AWARE_CONFIG.experiment_name: "Feature-rich (understat/form)",
    },
}

# Evaluation-report column headers (raw CSV column -> translated header).
EVAL_COLUMNS = {
    "el": {
        "model": "Μοντέλο", "matches": "Αγώνες", "logloss": "Log loss",
        "brier": "Brier", "ece": "ECE", "accuracy": "Accuracy",
        "macro_f1": "Macro F1", "draw_recall": "Draw recall",
        "draw_pick_rate": "Draw pick rate",
    },
    "en": {
        "model": "Model", "matches": "Matches", "logloss": "Log loss",
        "brier": "Brier", "ece": "ECE", "accuracy": "Accuracy",
        "macro_f1": "Macro F1", "draw_recall": "Draw recall",
        "draw_pick_rate": "Draw pick rate",
    },
}

# Dataset-summary table headers.
DATASET_COLUMNS = {
    "el": {"league": "Πρωτάθλημα", "played": "Αγώνες (παιγμένοι)",
           "from": "Από", "to": "Έως", "user_rows": "Γραμμές χρήστη"},
    "en": {"league": "League", "played": "Matches (played)",
           "from": "From", "to": "To", "user_rows": "User rows"},
}

# All remaining UI strings. Values may contain {placeholders} filled via t(...).
TEXT = {
    "el": {
        "app_title": "⚽ Εργαλείο Πρόβλεψης Αποτελεσμάτων Ποδοσφαιρικών Αγώνων",
        "page_title": "Πρόβλεψη Ποδοσφαιρικών Αγώνων",
        "language": "Γλώσσα / Language",
        "nav_header": "### Πλοήγηση",
        "nav_predict": "Πρόβλεψη Αγώνα",
        "nav_evaluation": "Αξιολόγηση Μοντέλων",
        "nav_train": "Εκπαίδευση / Δεδομένα",
        "sidebar_info": ("Πείραμα: `{exp}`\n\nOpening odds, leakage-safe χρονικά splits. "
                         "5 πρωταθλήματα (Αγγλία, Ισπανία, Ιταλία, Γερμανία, Γαλλία)."),
        # predict page
        "predict_header": "Πρόβλεψη Αγώνα",
        "predict_caption": ("Επίλεξε πρωτάθλημα, ομάδες και (προαιρετικά) αποδόσεις. "
                            "Διάλεξε ποιο μοντέλο εμπιστεύεσαι — το εργαλείο δίνει "
                            "πιθανότητες 1/X/2 και επίπεδο εμπιστοσύνης."),
        "league": "Πρωτάθλημα",
        "experiment_featureset": "Πείραμα / σετ features",
        "experiment_help": ("Το «Αγορά» μοντέλο βασίζεται κυρίως στις αποδόσεις. Το «Πλούσιο σε "
                            "features» χρησιμοποιεί understat/φόρμα από το ιστορικό των ομάδων, "
                            "οπότε δίνει διαφορετική πρόβλεψη ακόμη και χωρίς αποδόσεις."),
        "current_season_teams": "Ομάδες τρέχουσας σεζόν ({n})",
        "home_team": "Γηπεδούχος",
        "away_team": "Φιλοξενούμενη",
        "market_odds_header": ("**Αποδόσεις αγοράς** (προαιρετικό — χρειάζονται για τα "
                               "μοντέλα *Αγορά / Διορθωμένη Αγορά / Ensemble*)"),
        "odds_home": "Απόδοση 1 (Γηπεδούχος)",
        "odds_draw": "Απόδοση X (Ισοπαλία)",
        "odds_away": "Απόδοση 2 (Φιλοξενούμενη)",
        "prediction_model": "Μοντέλο πρόβλεψης",
        "prediction_model_help": ("Το backend υπολογίζει όλα τα μοντέλα· εδώ επιλέγεις "
                                  "ποιο θα προβληθεί ως κύρια πρόβλεψη."),
        "same_team_warning": "Ο γηπεδούχος και η φιλοξενούμενη δεν μπορεί να είναι η ίδια ομάδα.",
        "predict_button": "Πρόβλεψη",
        "prediction_error": "Σφάλμα πρόβλεψης: {exc}",
        "no_odds_info": ("Δεν δόθηκαν έγκυρες αποδόσεις, οπότε τα μοντέλα που βασίζονται "
                         "στην αγορά πέφτουν πίσω στις πιθανότητες του στατιστικού μοντέλου."),
        "model_label": "**Μοντέλο:** {model}",
        "prediction_result": "Πρόβλεψη: **{pick}** — Εμπιστοσύνη: **{band}** ({pct:.1f}%)",
        "elo_home": "Elo Γηπεδούχου",
        "elo_away": "Elo Φιλοξενούμενης",
        "xg_home": "Αναμενόμενα γκολ (Γηπ.)",
        "xg_away": "Αναμενόμενα γκολ (Φιλ.)",
        "most_likely_scores": "**Πιθανότερα σκορ**",
        "score": "Σκορ",
        "probability_pct": "Πιθανότητα (%)",
        "compare_models": "Σύγκριση όλων των μοντέλων",
        "col_model": "Μοντέλο",
        "col_pick": "Πρόβλεψη",
        # evaluation page
        "eval_header": "Αξιολόγηση & Επιλογή Μοντέλου",
        "eval_caption": ("Αποθηκευμένα αποτελέσματα αξιολόγησης από το backtest. "
                         "Χαμηλότερο log loss / Brier / ECE = καλύτερο μοντέλο. "
                         "Χρησιμοποίησέ τα για να επιλέξεις μοντέλο στην καρτέλα Πρόβλεψη."),
        "experiment": "Πείραμα",
        "experiment_eval_help": ("Το canonical είναι το πείραμα που αναφέρει η εργασία· "
                                 "το user_retrain παράγεται από την καρτέλα Εκπαίδευσης."),
        "dataset": "Σύνολο",
        "eval_not_found": "Δεν βρέθηκαν αποτελέσματα αξιολόγησης: {exc}",
        "eval_missing_columns": "Το αρχείο αξιολόγησης δεν έχει τις βασικές στήλες (model/logloss).",
        "lowest_logloss": "Χαμηλότερο log loss: **{model}** ({ll:.4f})",
        "eval_note": ("Σημείωση: Log loss/Brier/ECE μετρούν την ποιότητα των *πιθανοτήτων* "
                      "(βαθμονόμηση), όχι μόνο το ποσοστό επιτυχίας. Το draw pick rate δείχνει "
                      "πόσο συχνά ένα μοντέλο επιλέγει «ισοπαλία» ως πιο πιθανή έκβαση."),
        "canonical_suffix": "{exp} (canonical)",
        # train page
        "train_header": "Εκπαίδευση / Επέκταση Δεδομένων",
        "dataset_subheader": "1) Σύνολο δεδομένων",
        "add_match_header": ("**Προσθήκη αγώνα στο dataset** (γράφεται σε ξεχωριστό αρχείο "
                             "`zz_user_added.csv` — τα αρχικά δεδομένα δεν αλλάζουν)"),
        "date": "Ημερομηνία",
        "home_goals": "Γκολ γηπεδούχου",
        "away_goals": "Γκολ φιλοξενούμενης",
        "add_row": "Προσθήκη γραμμής",
        "match_added": "Προστέθηκε: {home} {fthg}–{ftag} {away} ({date}) → {file}",
        "retrain_subheader": "2) (Επαν)εκπαίδευση μοντέλων",
        "retrain_caption": ("Τρέχει ολόκληρο το pipeline σε **ξεχωριστό** πείραμα «user_retrain», "
                            "ώστε τα canonical αποτελέσματα που αναφέρει η εργασία να μην αλλάζουν. "
                            "Επαναχρησιμοποιεί τις βελτιστοποιημένες υπερπαραμέτρους και ξανα-εκπαιδεύει "
                            "τα μοντέλα στα τρέχοντα (+ προστιθέμενα) δεδομένα. Αναμένεται να διαρκέσει "
                            "μερικά λεπτά."),
        "retrain_button": "Εκπαίδευση (retrain)",
        "retrain_spinner": "Εκπαίδευση σε εξέλιξη... (μην κλείσεις τη σελίδα)",
        "retrain_done": ("Η εκπαίδευση ολοκληρώθηκε. Δες τα αποτελέσματα στην καρτέλα "
                         "«Αξιολόγηση Μοντέλων» → πείραμα **user_retrain**."),
        "retrain_failed": "Η εκπαίδευση απέτυχε — δες το log παραπάνω.",
    },
    "en": {
        "app_title": "⚽ Football Match Result Prediction Tool",
        "page_title": "Football Match Prediction",
        "language": "Language / Γλώσσα",
        "nav_header": "### Navigation",
        "nav_predict": "Match Prediction",
        "nav_evaluation": "Model Evaluation",
        "nav_train": "Training / Data",
        "sidebar_info": ("Experiment: `{exp}`\n\nOpening odds, leakage-safe temporal splits. "
                         "5 leagues (England, Spain, Italy, Germany, France)."),
        # predict page
        "predict_header": "Match Prediction",
        "predict_caption": ("Choose a league, the teams and (optionally) the odds. "
                            "Pick which model to trust — the tool returns 1/X/2 "
                            "probabilities and a confidence level."),
        "league": "League",
        "experiment_featureset": "Experiment / feature set",
        "experiment_help": ("The \"Market\" model relies mostly on the odds. The "
                            "\"Feature-rich\" model uses understat/form from the teams' "
                            "history, so it gives a different prediction even without odds."),
        "current_season_teams": "Current-season teams ({n})",
        "home_team": "Home team",
        "away_team": "Away team",
        "market_odds_header": ("**Market odds** (optional — needed for the "
                               "*Market / Corrected market / Ensemble* models)"),
        "odds_home": "Odds 1 (Home)",
        "odds_draw": "Odds X (Draw)",
        "odds_away": "Odds 2 (Away)",
        "prediction_model": "Prediction model",
        "prediction_model_help": ("The backend computes every model; here you choose "
                                  "which one is shown as the main prediction."),
        "same_team_warning": "The home and away team cannot be the same.",
        "predict_button": "Predict",
        "prediction_error": "Prediction error: {exc}",
        "no_odds_info": ("No valid odds were given, so the market-based models fall back "
                         "to the statistical model's probabilities."),
        "model_label": "**Model:** {model}",
        "prediction_result": "Prediction: **{pick}** — Confidence: **{band}** ({pct:.1f}%)",
        "elo_home": "Home Elo",
        "elo_away": "Away Elo",
        "xg_home": "Expected goals (Home)",
        "xg_away": "Expected goals (Away)",
        "most_likely_scores": "**Most likely scorelines**",
        "score": "Score",
        "probability_pct": "Probability (%)",
        "compare_models": "Compare all models",
        "col_model": "Model",
        "col_pick": "Pick",
        # evaluation page
        "eval_header": "Evaluation & Model Selection",
        "eval_caption": ("Stored evaluation results from the backtest. "
                         "Lower log loss / Brier / ECE = better model. "
                         "Use them to choose a model on the Prediction page."),
        "experiment": "Experiment",
        "experiment_eval_help": ("The canonical experiment is the one cited by the thesis; "
                                 "user_retrain is produced from the Training page."),
        "dataset": "Dataset",
        "eval_not_found": "No evaluation results found: {exc}",
        "eval_missing_columns": "The evaluation file is missing the core columns (model/logloss).",
        "lowest_logloss": "Lowest log loss: **{model}** ({ll:.4f})",
        "eval_note": ("Note: log loss/Brier/ECE measure the quality of the *probabilities* "
                      "(calibration), not just the hit rate. Draw pick rate shows how often a "
                      "model picks \"draw\" as the most likely outcome."),
        "canonical_suffix": "{exp} (canonical)",
        # train page
        "train_header": "Training / Data Extension",
        "dataset_subheader": "1) Dataset",
        "add_match_header": ("**Add a match to the dataset** (written to a separate "
                             "`zz_user_added.csv` file — the original data is unchanged)"),
        "date": "Date",
        "home_goals": "Home goals",
        "away_goals": "Away goals",
        "add_row": "Add row",
        "match_added": "Added: {home} {fthg}–{ftag} {away} ({date}) → {file}",
        "retrain_subheader": "2) (Re)train the models",
        "retrain_caption": ("Runs the whole pipeline in a **separate** \"user_retrain\" experiment, "
                            "so the canonical results cited by the thesis stay unchanged. It reuses "
                            "the tuned hyper-parameters and refits the models on the current "
                            "(+ added) data. Expect it to take a few minutes."),
        "retrain_button": "Train (retrain)",
        "retrain_spinner": "Training in progress... (do not close the page)",
        "retrain_done": ("Training finished. See the results on the "
                         "\"Model Evaluation\" page → experiment **user_retrain**."),
        "retrain_failed": "Training failed — see the log above.",
    },
}


def t(key: str, **fmt) -> str:
    """Return the current language's text for ``key``, formatted with ``fmt`` if given."""
    text = TEXT[_lang()].get(key, key)
    return text.format(**fmt) if fmt else text


def _predict_config(exp_name: str):
    return PREDICT_EXPERIMENTS[exp_name]


def available_predict_experiments() -> list[str]:
    """Experiments whose runtime artifacts exist on disk (so they're loadable)."""
    return [
        name for name, cfg in PREDICT_EXPERIMENTS.items()
        if cfg.params_file.exists() and cfg.model_file.exists()
    ]


# --------------------------------------------------------------------------- #
# cached loaders                                                              #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=True)
def load_artifacts(exp_name: str = CANONICAL_EXP):
    return load_runtime_artifacts(_predict_config(exp_name))


@st.cache_resource(show_spinner=True)
def load_state(league: str, exp_name: str = CANONICAL_EXP):
    params = load_artifacts(exp_name)[0]
    return get_league_runtime_state(league, params)


def eval_file(exp: str) -> Path:
    """Path to an experiment's saved probability-quality evaluation CSV."""
    return FINAL_CONFIG.artifacts_dir / f"final_probability_quality_{exp}.csv"


def list_eval_experiments() -> list[str]:
    """Experiments relevant to the tool that have a stored evaluation report:
    the canonical experiment the thesis cites, plus the user's own retrain."""
    return [
        e for e in (CANONICAL_EXP, CONTEXT_AWARE_CONFIG.experiment_name, "user_retrain")
        if eval_file(e).exists()
    ]


@st.cache_data(show_spinner=False)
def load_latest_eval(exp: str, split: str) -> pd.DataFrame:
    """Latest run of the per-split probability-quality report, one row per model."""
    df = pd.read_csv(eval_file(exp))
    if "split" in df.columns:
        df = df[df["split"] == split]
    if "run_ts_utc" in df.columns and len(df):
        df = df[df["run_ts_utc"] == df["run_ts_utc"].max()]
    return df.reset_index(drop=True)


# --- dataset extension (FR11) ---------------------------------------------- #
def user_file(league: str) -> Path:
    return RAW_DIR / league / USER_FILE


def _dataset_token() -> str:
    """Changes whenever a user-added file changes, to bust the summary cache."""
    return "|".join(
        f"{lg}:{user_file(lg).stat().st_mtime}" if user_file(lg).exists() else f"{lg}:0"
        for lg in LEAGUES
    )


@st.cache_data(show_spinner=False)
def dataset_summary(token: str, lang: str) -> pd.DataFrame:
    """Per-league dataset overview (played matches, date range, user-added rows)."""
    from src.data_processing import load_league_data

    cols = DATASET_COLUMNS[lang]
    rows = []
    for lg in LEAGUES:
        df = load_league_data(lg)
        dates = pd.to_datetime(df["date"])
        uf = user_file(lg)
        user_rows = max(0, sum(1 for _ in uf.open(encoding="utf-8")) - 1) if uf.exists() else 0
        rows.append({
            cols["league"]: LEAGUE_LABELS[lang].get(lg, lg),
            cols["played"]: int(df["is_played"].sum()),
            cols["from"]: dates.min().date().isoformat(),
            cols["to"]: dates.max().date().isoformat(),
            cols["user_rows"]: user_rows,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def league_teams(league: str, token: str, current_only: bool = True) -> list[str]:
    """Team names for a league. By default only the *current season's* clubs
    (the raw CSVs span ~2012-now, so the full set includes long-relegated teams)."""
    from src.data_processing import load_league_data

    df = load_league_data(league)
    if current_only:
        d = pd.to_datetime(df["date"])
        last = d.max()
        # a European season runs Aug->May; start it on 1 July of the season's first year
        season_start = pd.Timestamp(year=last.year if last.month >= 7 else last.year - 1, month=7, day=1)
        recent = df[d >= season_start]
        teams = sorted(set(recent["home_team"]) | set(recent["away_team"]))
        if len(teams) >= 6:  # guard against an empty/partial latest season
            return teams
    return sorted(set(df["home_team"]) | set(df["away_team"]))


def append_user_match(league: str, d: date, home: str, away: str, fthg: int, ftag: int) -> Path:
    """Append one match to the league's user CSV (football-data schema)."""
    path = user_file(league)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "Date": d.strftime("%d/%m/%Y"),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": int(fthg),
            "FTAG": int(ftag),
        })
    return path


def run_prediction(league: str, home: str, away: str, oh: float, od: float, oa: float,
                   exp_name: str = CANONICAL_EXP):
    """Load the chosen experiment's state + artifacts and predict one fixture."""
    state = load_state(league, exp_name)
    (_params, meta_model, meta_cfg, mlp_model, mlp_meta,
     logreg_model, logreg_meta, blend_cfg) = load_artifacts(exp_name)
    return predict_custom_match(
        home, away, oh, od, oa, state,
        meta_model, meta_cfg, mlp_model, mlp_meta, blend_cfg,
        logreg_model=logreg_model, logreg_meta=logreg_meta,
    )


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def confidence_band(probs: np.ndarray) -> tuple[str, float]:
    """Map the top outcome probability to a confidence level key and its value.

    Returns ``(level_key, top)`` where ``level_key`` is one of high/medium/low;
    the caller looks the key up in :data:`CONFIDENCE_LABELS` for the active language.
    """
    p = np.asarray(probs, dtype=float)
    top = float(p.max())
    if top >= 0.55:
        return "high", top
    if top >= 0.45:
        return "medium", top
    return "low", top


def probs_table(res: dict) -> pd.DataFrame:
    """Build the side-by-side comparison table of every model's 1/X/2 probabilities."""
    lang = _lang()
    rows = []
    for key, label in MODEL_LABELS[lang].items():
        if key not in res:
            continue
        p = np.asarray(res[key], dtype=float)
        rows.append({
            t("col_model"): label,
            "1 (%)": round(100 * p[0], 1),
            "X (%)": round(100 * p[1], 1),
            "2 (%)": round(100 * p[2], 1),
            t("col_pick"): OUTCOME_SHORT[int(p.argmax())],
            "Confidence (%)": round(100 * float(p.max()), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# pages                                                                       #
# --------------------------------------------------------------------------- #
def page_predict():
    """Interactive 1X2 prediction page: pick league/teams/odds/experiment/model."""
    lang = _lang()
    st.header(t("predict_header"))
    st.caption(t("predict_caption"))

    predict_exps = available_predict_experiments()
    col_l, col_m = st.columns([1, 1])
    with col_l:
        league = st.selectbox(
            t("league"), LEAGUES,
            format_func=lambda x: LEAGUE_LABELS[lang].get(x, x),
        )
    with col_m:
        exp_name = st.selectbox(
            t("experiment_featureset"), predict_exps,
            format_func=lambda e: EXPERIMENT_LABELS[lang].get(e, e),
            help=t("experiment_help"),
        )
    teams = league_teams(league, _dataset_token())
    st.caption(t("current_season_teams", n=len(teams)))

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox(t("home_team"), teams, key="home")
    with c2:
        default_away = 1 if len(teams) > 1 else 0
        away = st.selectbox(t("away_team"), teams, index=default_away, key="away")

    st.markdown(t("market_odds_header"))
    o1, o2, o3 = st.columns(3)
    with o1:
        oh = st.number_input(t("odds_home"), min_value=0.0, value=0.0, step=0.05, format="%.2f")
    with o2:
        od = st.number_input(t("odds_draw"), min_value=0.0, value=0.0, step=0.05, format="%.2f")
    with o3:
        oa = st.number_input(t("odds_away"), min_value=0.0, value=0.0, step=0.05, format="%.2f")

    model_keys = list(MODEL_LABELS[lang].keys())
    model_key = st.selectbox(
        t("prediction_model"),
        model_keys,
        index=model_keys.index("ensemble"),
        format_func=lambda k: MODEL_LABELS[lang][k],
        help=t("prediction_model_help"),
    )

    if home == away:
        st.warning(t("same_team_warning"))
        return

    if not st.button(t("predict_button"), type="primary"):
        return

    try:
        res = run_prediction(league, home, away, oh, od, oa, exp_name)
    except Exception as exc:  # noqa: BLE001 - surface any backend error to the UI
        st.error(t("prediction_error", exc=exc))
        return

    used_market = oh > 1.0 and od > 1.0 and oa > 1.0
    if model_key in {"market", "market_corr", "ensemble"} and not used_market:
        st.info(t("no_odds_info"))

    probs = np.asarray(res[model_key], dtype=float)
    pick_idx = int(probs.argmax())
    band_key, top = confidence_band(probs)
    outcomes = OUTCOME_LABELS[lang]

    st.subheader(f"{home} vs {away}")
    st.markdown(t("model_label", model=MODEL_LABELS[lang][model_key]))

    m1, m2, m3 = st.columns(3)
    m1.metric(outcomes[0], f"{100 * probs[0]:.1f}%")
    m2.metric(outcomes[1], f"{100 * probs[1]:.1f}%")
    m3.metric(outcomes[2], f"{100 * probs[2]:.1f}%")

    st.success(t("prediction_result", pick=outcomes[pick_idx],
                 band=CONFIDENCE_LABELS[lang][band_key], pct=100 * top))

    e1, e2 = st.columns(2)
    e1.metric(t("elo_home"), f"{res['elo'][0]:.0f}")
    e2.metric(t("elo_away"), f"{res['elo'][1]:.0f}")
    e1.metric(t("xg_home"), f"{res['xg'][0]:.2f}")
    e2.metric(t("xg_away"), f"{res['xg'][1]:.2f}")

    st.markdown(t("most_likely_scores"))
    st.table(pd.DataFrame(
        [{t("score"): f"{hg}-{ag}", t("probability_pct"): round(100 * psc, 1)}
         for (hg, ag), psc in res["scores"]]
    ))

    with st.expander(t("compare_models")):
        st.dataframe(probs_table(res), hide_index=True, width="stretch")


def page_evaluation():
    """Model-selection page: show stored evaluation metrics so the user can compare models."""
    lang = _lang()
    st.header(t("eval_header"))
    st.caption(t("eval_caption"))

    experiments = list_eval_experiments()
    c1, c2 = st.columns([2, 1])
    with c1:
        exp = st.selectbox(
            t("experiment"), experiments,
            format_func=lambda e: t("canonical_suffix", exp=e) if e == CANONICAL_EXP else e,
            help=t("experiment_eval_help"),
        )
    with c2:
        split = st.radio(
            t("dataset"), ["test", "validation"], horizontal=True,
            format_func=lambda s: "Test" if s == "test" else "Validation",
        )

    try:
        df = load_latest_eval(exp, split)
    except Exception as exc:  # noqa: BLE001
        st.error(t("eval_not_found", exc=exc))
        return

    # Build the table from whatever columns this experiment's report actually has,
    # so older reports with a leaner schema still render instead of crashing.
    col_labels = EVAL_COLUMNS[lang]
    model_col = col_labels["model"]
    logloss_col = col_labels["logloss"]
    present = [c for c in col_labels if c in df.columns]
    if "model" not in present or "logloss" not in present:
        st.error(t("eval_missing_columns"))
        return

    show = df[present].copy()
    show["model"] = show["model"].map(lambda m: MODEL_LABELS[lang].get(m, m))
    show = show.rename(columns=col_labels).sort_values(logloss_col).reset_index(drop=True)

    best_model = show.iloc[0][model_col]
    best_ll = show.iloc[0][logloss_col]
    st.success(t("lowest_logloss", model=best_model, ll=best_ll))

    numeric_labels = [col_labels[k] for k in
                      ("logloss", "brier", "ece", "accuracy", "macro_f1", "draw_recall", "draw_pick_rate")]
    fmt = {label: "{:.4f}" if label in (col_labels["logloss"], col_labels["brier"], col_labels["ece"]) else "{:.3f}"
           for label in numeric_labels if label in show.columns}
    highlight = [col_labels[k] for k in ("logloss", "brier", "ece") if col_labels[k] in show.columns]
    styler = show.style.format(fmt)
    if highlight:
        styler = styler.highlight_min(subset=highlight, color="#1b5e20")
    st.dataframe(styler, hide_index=True, width="stretch")

    st.caption(t("eval_note"))


def page_train():
    """Dataset-extension + retrain page: add matches and retrain into the user_retrain experiment."""
    lang = _lang()
    st.header(t("train_header"))

    if st.session_state.get("last_add"):
        st.success(st.session_state.pop("last_add"))

    # --- 1) dataset (FR11) ------------------------------------------------- #
    st.subheader(t("dataset_subheader"))
    st.dataframe(dataset_summary(_dataset_token(), lang), hide_index=True, width="stretch")

    st.markdown(t("add_match_header"))
    league = st.selectbox(
        t("league"), LEAGUES,
        format_func=lambda x: LEAGUE_LABELS[lang].get(x, x), key="train_league",
    )
    teams = league_teams(league, _dataset_token())
    with st.form("add_match", clear_on_submit=True):
        a, b = st.columns(2)
        home = a.selectbox(t("home_team"), teams, key="tr_home")
        away = b.selectbox(t("away_team"), teams, index=min(1, len(teams) - 1), key="tr_away")
        c, d_, e = st.columns(3)
        match_date = c.date_input(t("date"), value=date.today())
        fthg = d_.number_input(t("home_goals"), min_value=0, max_value=30, value=0, step=1)
        ftag = e.number_input(t("away_goals"), min_value=0, max_value=30, value=0, step=1)
        submitted = st.form_submit_button(t("add_row"))
    if submitted:
        if home == away:
            st.warning(t("same_team_warning"))
        else:
            path = append_user_match(league, match_date, home, away, int(fthg), int(ftag))
            st.session_state["last_add"] = t(
                "match_added", home=home, fthg=int(fthg), ftag=int(ftag),
                away=away, date=match_date.isoformat(), file=path.name,
            )
            st.rerun()

    st.divider()

    # --- 2) retrain (FR10) ------------------------------------------------- #
    st.subheader(t("retrain_subheader"))
    st.caption(t("retrain_caption"))
    if st.button(t("retrain_button"), type="primary"):
        log_box = st.empty()
        lines: list[str] = []
        cmd = [sys.executable, str(ROOT / "scripts" / "retrain_runner.py"),
               "--experiment", "user_retrain", "--reset"]
        with st.spinner(t("retrain_spinner")):
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            for line in proc.stdout:  # live stream into the UI
                lines.append(line.rstrip("\n"))
                log_box.code("\n".join(lines[-25:]))
            proc.wait()
        if proc.returncode == 0 and any("RETRAIN_DONE" in ln for ln in lines):
            st.cache_data.clear()
            st.success(t("retrain_done"))
        else:
            st.error(t("retrain_failed"))


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    """Streamlit entry point: language selector + sidebar navigation between the three pages."""
    st.set_page_config(page_title=t("page_title"), page_icon="⚽", layout="wide")
    st.title(t("app_title"))

    with st.sidebar:
        st.radio(
            TEXT[DEFAULT_LANG]["language"],  # label shown in both languages
            list(LANGUAGES.keys()),
            format_func=lambda c: LANGUAGES[c],
            horizontal=True,
            key="lang",
        )
        st.divider()
        st.markdown(t("nav_header"))
        pages = ["predict", "evaluation", "train"]
        page = st.radio(
            t("nav_header"),
            pages,
            format_func=lambda p: t(f"nav_{p}"),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(t("sidebar_info", exp=FINAL_CONFIG.experiment_name))

    if page == "predict":
        page_predict()
    elif page == "evaluation":
        page_evaluation()
    else:
        page_train()


if __name__ == "__main__":
    main()
