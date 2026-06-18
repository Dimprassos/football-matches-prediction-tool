"""Streamlit tool for predicting football match results.

Thin UI layer over the trained pipeline. It loads the cached artifacts of the
configured experiments (the canonical `final_opening_market_pre_match` uses
opening odds and is leakage-safe) and offers four pages:

  * **Match prediction** — an interactive 1X2 predictor where the user picks the
    experiment, the league, the two teams, the odds and *which model* to trust
    (configurability + confidence level). Models range from the base statistical
    model and the market to XGBoost / MLP / LogReg, their ensemble, and — for the
    deep-learning experiment — FootyNet (LSTM) and the FootyNet+market stack.
  * **Model evaluation** — shows the stored evaluation metrics per model so the
    user can decide which one to use.
  * **Training & data** — add matches to the dataset and retrain on the new data.
  * **About / Methodology** — explains the pipeline, data, leakage controls,
    metrics, and the market benchmark.

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

from src.config import CONTEXT_AWARE_CONFIG, FINAL_CONFIG, FOOTYNET_CONFIG, PLAYER_CONTEXT_CONFIG
from src.player_context import (
    build_player_strength_index,
    build_runtime_player_context,
    load_player_context_tables,
)
from src.predictor import (
    get_league_runtime_state,
    load_runtime_artifacts,
    predict_custom_match,
)
from src.team_names import normalize_team_name

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
USER_FILE = "zz_user_added.csv"  # one per league; appended rows the loader auto-merges
CANONICAL_EXP = FINAL_CONFIG.experiment_name
FOOTYNET_CKPT = ROOT / "artifacts" / "footynet_deep.pt"  # trained by scripts/train_footynet.py

LEAGUES = ["england", "spain", "italy", "germany", "france"]
OUTCOME_SHORT = ["1", "X", "2"]
PLAYER_CONTEXT_FILES = [
    "player_registry.csv",
    "player_match_stats.csv",
    "match_lineups.csv",
    "match_absences.csv",
]

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
        "footynet": "FootyNet (LSTM)",
        "footynet_stack": "FootyNet + Αγορά (stack)",
    },
    "en": {
        "base": "Base — Elo + Poisson/Dixon-Coles",
        "market": "Market — opening odds",
        "market_corr": "Corrected market — market + model",
        "meta": "XGBoost",
        "logreg": "Logistic Regression",
        "mlp": "Neural network (MLP)",
        "ensemble": "Ensemble (blend)",
        "footynet": "FootyNet (LSTM)",
        "footynet_stack": "FootyNet + Market (stack)",
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
    PLAYER_CONTEXT_CONFIG.experiment_name: PLAYER_CONTEXT_CONFIG,
    FOOTYNET_CONFIG.experiment_name: FOOTYNET_CONFIG,
}
EXPERIMENT_LABELS = {
    "el": {
        FINAL_CONFIG.experiment_name: "Αγορά (canonical)",
        CONTEXT_AWARE_CONFIG.experiment_name: "Πλούσιο σε features (understat/form)",
        PLAYER_CONTEXT_CONFIG.experiment_name: "Με εντεκάδες (lineup strength)",
        FOOTYNET_CONFIG.experiment_name: "Βαθιά μάθηση (LSTM / FootyNet)",
    },
    "en": {
        FINAL_CONFIG.experiment_name: "Market (canonical)",
        CONTEXT_AWARE_CONFIG.experiment_name: "Feature-rich (understat/form)",
        PLAYER_CONTEXT_CONFIG.experiment_name: "Lineup-aware (lineup strength)",
        FOOTYNET_CONFIG.experiment_name: "Deep learning (LSTM / FootyNet)",
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
        "app_title": "Εργαλείο Πρόβλεψης Αποτελεσμάτων Ποδοσφαιρικών Αγώνων",
        "page_title": "Πρόβλεψη Ποδοσφαιρικών Αγώνων",
        "language": "Γλώσσα / Language",
        "nav_header": "### Πλοήγηση",
        "nav_predict": "Πρόβλεψη Αγώνα",
        "nav_evaluation": "Αξιολόγηση Μοντέλων",
        "nav_train": "Εκπαίδευση / Δεδομένα",
        "nav_about": "Μεθοδολογία / About",
        "sidebar_selected_exp": "**Επιλεγμένο πείραμα**\n\n{exp}",
        "sidebar_info": ("Opening odds, leakage-safe χρονικά splits. "
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
        "player_context_header": "Player context / ενδεκάδες και απουσίες",
        "player_context_enable": "Χρήση χειροκίνητων ενδεκάδων/απουσιών",
        "player_context_match_date": "Ημερομηνία αγώνα για rolling player strength",
        "player_context_no_registry": ("Δεν υπάρχουν παίκτες στο `player_registry.csv` για μία ή και τις δύο "
                                       "ομάδες. Μπορείς να δώσεις `player_id` χειροκίνητα."),
        "home_starters": "Βασικοί γηπεδούχου",
        "away_starters": "Βασικοί φιλοξενούμενης",
        "home_absences": "Απουσίες γηπεδούχου",
        "away_absences": "Απουσίες φιλοξενούμενης",
        "absence_type": "Τύπος απουσίας",
        "absence_status": "Κατάσταση",
        "player_ids_placeholder": "player_id ανά γραμμή ή με κόμμα",
        "player_context_empty": "Δεν δόθηκαν παίκτες. Η πρόβλεψη μένει χωρίς player context.",
        "player_context_neutral": ("Δόθηκαν παίκτες, αλλά δεν βρέθηκαν rolling stats ή importance tier. "
                                   "Θα χρησιμοποιηθεί neutral fallback."),
        "player_context_ready": ("Player context ενεργό: home={home:.0f}, away={away:.0f}, "
                                 "absence diff={diff:.2f}"),
        "player_context_summary": "Σύνοψη player context",
        "player_context_load_error": "Σφάλμα φόρτωσης player context: {exc}",
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
        "prob_chart_header": "**Κατανομή πιθανοτήτων**",
        "compare_chart_header": "**Σύγκριση μοντέλων ανά έκβαση (%)**",
        "vs_market_caption": ("Τα βελάκια δείχνουν τη διαφορά του μοντέλου από την αγορά "
                              "σε ποσοστιαίες μονάδες (περιγραφικά, όχι σύσταση στοιχήματος)."),
        "glossary_header": "Τι σημαίνει κάθε μοντέλο;",
        "glossary_body": (
            "- **Βάση (base)** — στατιστικό μοντέλο Elo + Poisson/Dixon-Coles (δικό μας).\n"
            "- **Αγορά (market)** — οι opening αποδόσεις μετατρεμμένες σε πιθανότητες "
            "(εξωτερική αναφορά, όχι μοντέλο μας· σχεδόν-βέλτιστο benchmark).\n"
            "- **Διορθωμένη Αγορά** — η αγορά διορθωμένη ελαφρώς από το μοντέλο μας.\n"
            "- **XGBoost / Logistic Regression / MLP** — εκπαιδευμένα meta μοντέλα (δικά μας).\n"
            "- **Ensemble** — σταθμισμένο blend των παραπάνω.\n"
            "- **FootyNet (LSTM)** — μοντέλο βαθιάς μάθησης: 2 LSTM encoders στα τελευταία K "
            "ματς κάθε ομάδας + static branch.\n"
            "- **FootyNet + Αγορά (stack)** — convex blend FootyNet × αγοράς, με βάρη "
            "μαθημένα στο validation (leakage-safe)."
        ),
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
        # about / methodology page
        "about_header": "Μεθοδολογία & Επεξήγηση",
        "about_pipeline_h": "Pipeline πρόβλεψης",
        "about_pipeline": (
            "Για κάθε αγώνα υπολογίζεται ένα **στατιστικό base** (Elo + Poisson/Dixon-Coles), "
            "από το οποίο και από τις **αποδόσεις της αγοράς** χτίζονται features για τα meta "
            "μοντέλα (XGBoost, Logistic Regression, MLP) και ένα **ensemble** blend. Παράλληλα "
            "εκπαιδεύτηκε ένα μοντέλο **βαθιάς μάθησης (FootyNet)** πάνω σε ακολουθίες των "
            "τελευταίων αγώνων κάθε ομάδας. Όλες οι πιθανότητες βαθμονομούνται (temperature "
            "scaling) ώστε να είναι αξιόπιστες."
        ),
        "about_data_h": "Πηγές δεδομένων",
        "about_data": (
            "Ιστορικά αποτελέσματα & αποδόσεις από football-data.co.uk (5 λίγκες: Αγγλία, "
            "Ισπανία, Ιταλία, Γερμανία, Γαλλία), συν understat xG. Ως «αγορά» χρησιμοποιούνται "
            "οι **opening** αποδόσεις (μέσος όρος μπουκμέικερ), μετατρεμμένες σε πιθανότητες "
            "αφαιρώντας το περιθώριο (vig)."
        ),
        "about_leakage_h": "Πολιτική κατά του leakage",
        "about_leakage": (
            "- **Χρονικά splits:** train < 2024-07, validation 2024-25, test ≥ 2025-07.\n"
            "- **Κανόνας `date < D`:** ένας αγώνας στην ημερομηνία D χρησιμοποιεί μόνο "
            "δεδομένα από προηγούμενους αγώνες — ποτέ μελλοντικά.\n"
            "- Οι **opening αποδόσεις** είναι γνωστές πριν το παιχνίδι· το τελικό σκορ δεν "
            "μπαίνει ποτέ στα features.\n"
            "- Τα βάρη του stacking μαθαίνονται **μόνο στο validation**, αξιολογούνται στο test."
        ),
        "about_metrics_h": "Τι σημαίνει κάθε μετρική",
        "about_metrics": (
            "- **Log loss / Brier / ECE** — ποιότητα & βαθμονόμηση των *πιθανοτήτων* "
            "(χαμηλότερο = καλύτερο).\n"
            "- **Accuracy** — ποσοστό σωστών κορυφαίων προβλέψεων.\n"
            "- **Macro F1** — ισορροπία ανά κλάση (1/X/2).\n"
            "- **Draw recall** — πόσες πραγματικές ισοπαλίες πιάνει το μοντέλο."
        ),
        "about_benchmark_h": "Η αγορά ως benchmark",
        "about_benchmark": (
            "Η αγορά είναι σχεδόν-βέλτιστη: ενσωματώνει πληροφορία που δεν υπάρχει στα "
            "ιστορικά δεδομένα. Στόχος του εργαλείου **δεν** είναι να τη νικήσει, αλλά να "
            "δείξει πόσο κοντά φτάνουν τα μοντέλα και να επιτρέψει συγκρίσεις με διάφορες "
            "μετρικές. Το «πλησιάζουμε αλλά δεν ξεπερνάμε την αγορά» είναι έγκυρο εύρημα."
        ),
    },
    "en": {
        "app_title": "Football Match Result Prediction Tool",
        "page_title": "Football Match Prediction",
        "language": "Language / Γλώσσα",
        "nav_header": "### Navigation",
        "nav_predict": "Match Prediction",
        "nav_evaluation": "Model Evaluation",
        "nav_train": "Training / Data",
        "nav_about": "Methodology / About",
        "sidebar_selected_exp": "**Selected experiment**\n\n{exp}",
        "sidebar_info": ("Opening odds, leakage-safe temporal splits. "
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
        "player_context_header": "Player context / lineups and absences",
        "player_context_enable": "Use manual lineups/absences",
        "player_context_match_date": "Match date for rolling player strength",
        "player_context_no_registry": ("No `player_registry.csv` players were found for one or both teams. "
                                       "You can enter `player_id` values manually."),
        "home_starters": "Home starters",
        "away_starters": "Away starters",
        "home_absences": "Home absences",
        "away_absences": "Away absences",
        "absence_type": "Absence type",
        "absence_status": "Status",
        "player_ids_placeholder": "one player_id per line or comma separated",
        "player_context_empty": "No players were provided. Prediction stays without player context.",
        "player_context_neutral": ("Players were provided, but no rolling stats or importance tier were found. "
                                   "The neutral fallback will be used."),
        "player_context_ready": "Player context active: home={home:.0f}, away={away:.0f}, absence diff={diff:.2f}",
        "player_context_summary": "Player context summary",
        "player_context_load_error": "Player context load error: {exc}",
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
        "prob_chart_header": "**Probability distribution**",
        "compare_chart_header": "**Model comparison per outcome (%)**",
        "vs_market_caption": ("Arrows show the model's difference from the market in "
                              "percentage points (descriptive, not a betting recommendation)."),
        "glossary_header": "What does each model mean?",
        "glossary_body": (
            "- **Base** — statistical Elo + Poisson/Dixon-Coles model (ours).\n"
            "- **Market** — the opening odds converted to probabilities (external reference, "
            "not our model; a near-optimal benchmark).\n"
            "- **Corrected market** — the market lightly corrected by our model.\n"
            "- **XGBoost / Logistic Regression / MLP** — trained meta models (ours).\n"
            "- **Ensemble** — weighted blend of the above.\n"
            "- **FootyNet (LSTM)** — deep-learning model: 2 LSTM encoders over each team's "
            "last K matches + a static branch.\n"
            "- **FootyNet + Market (stack)** — convex blend of FootyNet x market, weights "
            "learned on validation (leakage-safe)."
        ),
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
        # about / methodology page
        "about_header": "Methodology & Explanation",
        "about_pipeline_h": "Prediction pipeline",
        "about_pipeline": (
            "For each match a **statistical base** (Elo + Poisson/Dixon-Coles) is computed; "
            "from it and the **market odds** we build features for the meta models (XGBoost, "
            "Logistic Regression, MLP) and an **ensemble** blend. In parallel a **deep-learning "
            "model (FootyNet)** was trained on each team's recent-match sequences. All "
            "probabilities are calibrated (temperature scaling) to stay reliable."
        ),
        "about_data_h": "Data sources",
        "about_data": (
            "Historical results & odds from football-data.co.uk (5 leagues: England, Spain, "
            "Italy, Germany, France), plus understat xG. The \"market\" uses the **opening** "
            "odds (bookmaker average), converted to probabilities by removing the margin (vig)."
        ),
        "about_leakage_h": "Anti-leakage policy",
        "about_leakage": (
            "- **Temporal splits:** train < 2024-07, validation 2024-25, test >= 2025-07.\n"
            "- **`date < D` rule:** a match on date D uses only data from earlier matches — "
            "never future ones.\n"
            "- **Opening odds** are known before kickoff; the final score never enters the "
            "features.\n"
            "- Stacking weights are learned **only on validation**, evaluated on test."
        ),
        "about_metrics_h": "What each metric means",
        "about_metrics": (
            "- **Log loss / Brier / ECE** — quality & calibration of the *probabilities* "
            "(lower = better).\n"
            "- **Accuracy** — share of correct top picks.\n"
            "- **Macro F1** — per-class balance (1/X/2).\n"
            "- **Draw recall** — how many real draws the model catches."
        ),
        "about_benchmark_h": "The market as a benchmark",
        "about_benchmark": (
            "The market is near-optimal: it embeds information absent from historical data. "
            "The tool's goal is **not** to beat it, but to show how close the models get and "
            "to allow comparisons across metrics. \"We approach but do not surpass the market\" "
            "is a valid finding."
        ),
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
    names = [
        name for name, cfg in PREDICT_EXPERIMENTS.items()
        if cfg.params_file.exists() and cfg.model_file.exists()
    ]
    # FootyNet ships a torch checkpoint instead of the meta JSON/model pair.
    foot = FOOTYNET_CONFIG.experiment_name
    if foot in PREDICT_EXPERIMENTS and FOOTYNET_CKPT.exists() and foot not in names:
        names.append(foot)
    return names


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


@st.cache_resource(show_spinner=True)
def load_footynet_model():
    """Load the trained FootyNet checkpoint once (torch model + metadata)."""
    from src.footynet_serve import load_footynet

    return load_footynet(str(FOOTYNET_CKPT))


@st.cache_resource(show_spinner=True)
def load_footynet_sequences(league: str, token: str):
    """Per-team last-K match sequences for one league (rebuilt when the dataset changes)."""
    from src.sequence_data import build_team_sequences

    return build_team_sequences(load_state(league, CANONICAL_EXP).played_df)


def _player_context_token() -> str:
    """Changes when any player-context CSV changes, to bust the Streamlit cache."""
    parts = []
    for name in PLAYER_CONTEXT_FILES:
        path = EXTERNAL_DIR / name
        parts.append(f"{name}:{path.stat().st_mtime}" if path.exists() else f"{name}:0")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def load_player_tables(token: str):
    """Load optional player-context CSVs from data/external."""
    return load_player_context_tables(EXTERNAL_DIR)


@st.cache_resource(show_spinner=True)
def load_player_strength_index(token: str):
    """Build and cache the player-strength index (expensive; rebuilt only when CSVs change).

    The tables came through the strict loader, which already normalizes names/dates, so
    the index can skip re-normalization and just group.
    """
    tables = load_player_context_tables(EXTERNAL_DIR)
    return build_player_strength_index(tables.match_stats, tables.registry, assume_normalized=True)


def eval_file(exp: str) -> Path:
    """Path to an experiment's saved probability-quality evaluation CSV."""
    return FINAL_CONFIG.artifacts_dir / f"final_probability_quality_{exp}.csv"


def list_eval_experiments() -> list[str]:
    """Experiments relevant to the tool that have a stored evaluation report:
    the canonical experiment the thesis cites, plus the user's own retrain."""
    return [
        e for e in (
            CANONICAL_EXP,
            CONTEXT_AWARE_CONFIG.experiment_name,
            PLAYER_CONTEXT_CONFIG.experiment_name,
            FOOTYNET_CONFIG.experiment_name,
            "user_retrain",
        )
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
                   exp_name: str = CANONICAL_EXP, context: dict | None = None):
    """Load the chosen experiment's state + artifacts and predict one fixture."""
    if exp_name == FOOTYNET_CONFIG.experiment_name:
        from src.footynet_serve import predict_footynet_fixture

        model, ckpt = load_footynet_model()
        state = load_state(league, CANONICAL_EXP)
        return predict_footynet_fixture(
            home, away, oh, od, oa, state=state, model=model, ckpt=ckpt,
            team_sequences=load_footynet_sequences(league, _dataset_token()),
        )
    state = load_state(league, exp_name)
    (_params, meta_model, meta_cfg, mlp_model, mlp_meta,
     logreg_model, logreg_meta, blend_cfg) = load_artifacts(exp_name)
    return predict_custom_match(
        home, away, oh, od, oa, state,
        meta_model, meta_cfg, mlp_model, mlp_meta, blend_cfg,
        logreg_model=logreg_model, logreg_meta=logreg_meta,
        context=context,
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


def _team_registry_rows(registry: pd.DataFrame, league: str, team: str, match_date: date) -> pd.DataFrame:
    """Registry rows active for ``team`` at the chosen runtime prediction date."""
    if registry is None or registry.empty:
        return pd.DataFrame()

    league_norm = str(league).strip().lower()
    team_norm = normalize_team_name(team, league_norm)
    as_of = pd.Timestamp(match_date)
    rows = registry.copy()
    date_from = pd.to_datetime(rows.get("date_from", pd.Series(pd.NaT, index=rows.index)), errors="coerce")
    date_to = pd.to_datetime(rows.get("date_to", pd.Series(pd.NaT, index=rows.index)), errors="coerce")
    mask = (
        (rows["league"].astype(str) == league_norm)
        & (rows["team"].astype(str) == team_norm)
        & (date_from.isna() | (date_from <= as_of))
        & (date_to.isna() | (date_to >= as_of))
    )
    rows = rows.loc[mask].copy()
    if rows.empty:
        return rows
    rows["player_id"] = rows["player_id"].astype(str)
    rows = rows.sort_values(["player_name", "player_id"])
    return rows.drop_duplicates("player_id", keep="last").reset_index(drop=True)


def _player_label_map(rows: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for _, row in rows.iterrows():
        player_id = str(row.get("player_id", "")).strip()
        if not player_id:
            continue
        name = str(row.get("player_name", player_id)).strip() or player_id
        meta = [
            str(row.get(col, "")).strip()
            for col in ("position", "importance_tier")
            if str(row.get(col, "")).strip() and str(row.get(col, "")).strip() != "unknown"
        ]
        labels[player_id] = f"{name} ({', '.join(meta)})" if meta else name
    return labels


def _parse_player_ids(text: str) -> list[str]:
    seen: set[str] = set()
    player_ids: list[str] = []
    for part in str(text or "").replace(",", "\n").splitlines():
        player_id = part.strip()
        if player_id and player_id not in seen:
            player_ids.append(player_id)
            seen.add(player_id)
    return player_ids


def _player_ids_input(label_key: str, rows: pd.DataFrame, key: str,
                      default: list[str] | None = None) -> list[str]:
    default = [str(p) for p in (default or [])]
    if rows is None or rows.empty:
        value = st.text_area(
            t(label_key),
            value="\n".join(default),
            key=f"{key}_text",
            placeholder=t("player_ids_placeholder"),
            height=96,
        )
        return _parse_player_ids(value)

    labels = _player_label_map(rows)
    options = list(dict.fromkeys(rows["player_id"].astype(str).tolist()))
    preselect = [p for p in default if p in options]
    return st.multiselect(
        t(label_key),
        options,
        default=preselect,
        format_func=lambda player_id: labels.get(player_id, player_id),
        key=f"{key}_select",
    )


def _absence_items_input(label_key: str, rows: pd.DataFrame, key: str) -> list[dict]:
    player_ids = _player_ids_input(label_key, rows, key)
    c1, c2 = st.columns(2)
    absence_type = c1.selectbox(t("absence_type"), ["injury", "suspension", "rest", "other"], key=f"{key}_type")
    status = c2.selectbox(t("absence_status"), ["out", "doubtful", "available"], key=f"{key}_status")
    return [
        {"player_id": player_id, "absence_type": absence_type, "status": status}
        for player_id in player_ids
    ]


def _render_player_context_preview(context: dict | None, diagnostics: pd.DataFrame) -> None:
    context = context or {}
    if not context and (diagnostics is None or diagnostics.empty):
        st.caption(t("player_context_empty"))
        return

    home_available = float(context.get("home_player_context_available", 0.0))
    away_available = float(context.get("away_player_context_available", 0.0))
    diff = float(context.get("absence_strength_loss_diff", 0.0))
    has_strength = diagnostics is not None and not diagnostics.empty and bool(diagnostics["context_available"].astype(bool).any())
    if has_strength:
        st.success(t("player_context_ready", home=home_available, away=away_available, diff=diff))
    else:
        st.info(t("player_context_neutral"))

    if diagnostics is None or diagnostics.empty:
        return
    show = diagnostics.copy()
    show["player_name"] = show["player_name"].where(show["player_name"].astype(str).str.strip() != "", show["player_id"])
    show["player_strength"] = pd.to_numeric(show["player_strength"], errors="coerce").round(3)
    st.markdown(t("player_context_summary"))
    st.dataframe(
        show[[
            "side",
            "role",
            "player_name",
            "absence_type",
            "status",
            "player_strength",
            "source",
            "context_available",
            "history_matches",
        ]],
        hide_index=True,
        width="stretch",
    )


def player_context_controls(league: str, home: str, away: str) -> dict | None:
    """Optional Streamlit controls that build runtime player context for prediction."""
    context: dict | None = None
    with st.expander(t("player_context_header")):
        enabled = st.checkbox(t("player_context_enable"), value=False)
        if not enabled:
            return None

        match_date = st.date_input(t("player_context_match_date"), value=date.today())
        try:
            tables = load_player_tables(_player_context_token())
        except Exception as exc:  # noqa: BLE001 - show validation errors in the UI
            st.error(t("player_context_load_error", exc=exc))
            return None

        home_rows = _team_registry_rows(tables.registry, league, home, match_date)
        away_rows = _team_registry_rows(tables.registry, league, away, match_date)
        if home_rows.empty or away_rows.empty:
            st.info(t("player_context_no_registry"))

        # Pre-fill each side with its likely XI (most-used players recently) so the user
        # can tweak rather than pick 11 from scratch.
        index = load_player_strength_index(_player_context_token())
        home_xi = index.likely_xi(home, league, match_date)
        away_xi = index.likely_xi(away, league, match_date)

        h_col, a_col = st.columns(2)
        with h_col:
            st.markdown(f"**{home}**")
            home_starters = _player_ids_input("home_starters", home_rows, "home_starters", default=home_xi)
            home_absences = _absence_items_input("home_absences", home_rows, "home_absences")
        with a_col:
            st.markdown(f"**{away}**")
            away_starters = _player_ids_input("away_starters", away_rows, "away_starters", default=away_xi)
            away_absences = _absence_items_input("away_absences", away_rows, "away_absences")

        try:
            context, diagnostics = build_runtime_player_context(
                league=league,
                home_team=home,
                away_team=away,
                match_date=match_date,
                registry=tables.registry,
                match_stats=tables.match_stats,
                home_starters=home_starters,
                away_starters=away_starters,
                home_absences=home_absences,
                away_absences=away_absences,
                index=load_player_strength_index(_player_context_token()),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(t("player_context_load_error", exc=exc))
            return None
        _render_player_context_preview(context, diagnostics)
    return context


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
            key="predict_exp",
        )
    with st.expander(t("glossary_header")):
        st.markdown(t("glossary_body"))
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

    if exp_name == FOOTYNET_CONFIG.experiment_name:
        # FootyNet serves the stacking blend (default), the raw net, the base model and the market.
        model_keys = ["footynet_stack", "footynet", "base", "market"]
        default_model_idx = 0
    else:
        # the FootyNet-only outputs are not produced by the other experiments
        model_keys = [k for k in MODEL_LABELS[lang].keys() if k not in ("footynet", "footynet_stack")]
        default_model_idx = model_keys.index("ensemble")
    model_key = st.selectbox(
        t("prediction_model"),
        model_keys,
        index=default_model_idx,
        format_func=lambda k: MODEL_LABELS[lang][k],
        help=t("prediction_model_help"),
    )

    if exp_name == FOOTYNET_CONFIG.experiment_name:
        st.info({
            "el": "Μοντέλο βαθιάς μάθησης (recurrent late-fusion): δύο LSTM encoders για τα "
                  "τελευταία K ματς κάθε ομάδας + static branch (base+market). Σύγκρινε το "
                  "**FootyNet** με το **base** (στατιστικό μοντέλο) και την **αγορά**. "
                  "Στην αξιολόγηση πετυχαίνει το καλύτερο logloss ανάμεσα στα καθαρά ML μοντέλα.",
            "en": "Deep-learning model (recurrent late-fusion): two LSTM encoders over each "
                  "team's last K matches + a static branch (base+market). Compare **FootyNet** "
                  "against the **base** model and the **market**. In evaluation it reaches the "
                  "best logloss among the pure-ML models.",
        }[lang])

    if exp_name == PLAYER_CONTEXT_CONFIG.experiment_name:
        st.info({
            "el": "Αυτό το πείραμα χρησιμοποιεί τη δύναμη εντεκάδας. Διάλεξε μοντέλο "
                  "**XGBoost** για να δεις την εντεκάδα να αλλάζει την πρόβλεψη — το "
                  "**ensemble** είναι market-dominated και δεν ανταποκρίνεται. (Το ablation "
                  "έδειξε ότι τα lineup features δεν βελτιώνουν το logloss έναντι της αγοράς.)",
            "en": "This experiment uses lineup strength. Pick the **XGBoost** model to see "
                  "the lineup change the prediction — the **ensemble** is market-dominated and "
                  "won't respond. (Ablation found lineup features do not improve logloss "
                  "over the market.)",
        }[lang])

    if home == away:
        st.warning(t("same_team_warning"))
        return

    player_context = (
        None if exp_name == FOOTYNET_CONFIG.experiment_name
        else player_context_controls(league, home, away)
    )

    if not st.button(t("predict_button"), type="primary"):
        return

    try:
        res = run_prediction(league, home, away, oh, od, oa, exp_name, context=player_context)
    except Exception as exc:  # noqa: BLE001 - surface any backend error to the UI
        st.error(t("prediction_error", exc=exc))
        return

    used_market = oh > 1.0 and od > 1.0 and oa > 1.0
    if model_key in {"market", "market_corr", "ensemble"} and not used_market:
        st.info(t("no_odds_info"))

    # guard against a transient experiment/model mismatch during a rerun
    if model_key not in res:
        model_key = next(iter(res))
    probs = np.asarray(res[model_key], dtype=float)
    pick_idx = int(probs.argmax())
    band_key, top = confidence_band(probs)
    outcomes = OUTCOME_LABELS[lang]

    st.subheader(f"{home} vs {away}")
    st.markdown(t("model_label", model=MODEL_LABELS[lang][model_key]))

    # show the model's gap to the market (descriptive) only when real odds were given
    market_ref = (np.asarray(res["market"], dtype=float)
                  if ("market" in res and used_market and model_key != "market") else None)
    for col, i in zip(st.columns(3), range(3)):
        if market_ref is not None:
            col.metric(outcomes[i], f"{100 * probs[i]:.1f}%",
                       delta=f"{100 * (probs[i] - market_ref[i]):+.1f} pp", delta_color="off")
        else:
            col.metric(outcomes[i], f"{100 * probs[i]:.1f}%")

    st.success(t("prediction_result", pick=outcomes[pick_idx],
                 band=CONFIDENCE_LABELS[lang][band_key], pct=100 * top))
    if market_ref is not None:
        st.caption(t("vs_market_caption"))

    st.markdown(t("prob_chart_header"))
    st.bar_chart(pd.DataFrame({MODEL_LABELS[lang][model_key]: 100 * probs}, index=OUTCOME_SHORT))

    # FootyNet serves only probability arrays; the Elo/xG/score-grid diagnostics
    # come from the statistical base pipeline and are absent for that experiment.
    if "elo" in res:
        e1, e2 = st.columns(2)
        e1.metric(t("elo_home"), f"{res['elo'][0]:.0f}")
        e2.metric(t("elo_away"), f"{res['elo'][1]:.0f}")
        e1.metric(t("xg_home"), f"{res['xg'][0]:.2f}")
        e2.metric(t("xg_away"), f"{res['xg'][1]:.2f}")

    if "scores" in res:
        st.markdown(t("most_likely_scores"))
        st.table(pd.DataFrame(
            [{t("score"): f"{hg}-{ag}", t("probability_pct"): round(100 * psc, 1)}
             for (hg, ag), psc in res["scores"]]
        ))

    with st.expander(t("compare_models")):
        st.markdown(t("compare_chart_header"))
        comparison = {MODEL_LABELS[lang][k]: 100 * np.asarray(res[k], dtype=float)
                      for k in MODEL_LABELS[lang] if k in res}
        st.bar_chart(pd.DataFrame(comparison, index=OUTCOME_SHORT))
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


def page_about():
    """Methodology / About page: pipeline, data sources, leakage policy, metrics, benchmark."""
    st.header(t("about_header"))
    for head_key, body_key in (
        ("about_pipeline_h", "about_pipeline"),
        ("about_data_h", "about_data"),
        ("about_leakage_h", "about_leakage"),
        ("about_metrics_h", "about_metrics"),
        ("about_benchmark_h", "about_benchmark"),
    ):
        st.subheader(t(head_key))
        st.markdown(t(body_key))


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    """Streamlit entry point: language selector + sidebar navigation between the three pages."""
    st.set_page_config(page_title=t("page_title"), layout="wide")
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
        pages = ["predict", "evaluation", "train", "about"]
        page = st.radio(
            t("nav_header"),
            pages,
            format_func=lambda p: t(f"nav_{p}"),
            label_visibility="collapsed",
        )
        st.divider()
        lang = _lang()
        selected = st.session_state.get("predict_exp", CANONICAL_EXP)
        st.info(t("sidebar_selected_exp", exp=EXPERIMENT_LABELS[lang].get(selected, selected)))
        st.caption(t("sidebar_info"))

    if page == "predict":
        page_predict()
    elif page == "evaluation":
        page_evaluation()
    elif page == "train":
        page_train()
    else:
        page_about()


if __name__ == "__main__":
    main()
