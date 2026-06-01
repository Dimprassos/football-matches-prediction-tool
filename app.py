"""Streamlit tool for predicting football match results.

Thin UI layer over the trained pipeline. It loads the cached artifacts of the
`final_opening_market_pre_match` experiment (opening odds, leakage-safe) and
exposes two things the assignment asks for:

  * an interactive 1X2 predictor where the user picks the league, the two teams,
    the odds and *which model* to trust (configurability + confidence level);
  * a model-selection screen that shows the stored evaluation metrics so the
    user can decide which model to use.

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

from src.config import FINAL_CONFIG
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
LEAGUE_LABELS = {
    "england": "Αγγλία (Premier League)",
    "spain": "Ισπανία (La Liga)",
    "italy": "Ιταλία (Serie A)",
    "germany": "Γερμανία (Bundesliga)",
    "france": "Γαλλία (Ligue 1)",
}

# user-facing label -> key returned by predict_custom_match / used in eval CSVs
MODEL_LABELS = {
    "base": "Βάση — Elo + Poisson/Dixon-Coles",
    "market": "Αγορά — opening odds",
    "market_corr": "Διορθωμένη Αγορά — market + model",
    "meta": "XGBoost",
    "logreg": "Logistic Regression",
    "mlp": "Νευρωνικό Δίκτυο (MLP)",
    "ensemble": "Ensemble (blend)",
}
OUTCOME_LABELS = ["Νίκη Γηπεδούχου (1)", "Ισοπαλία (X)", "Νίκη Φιλοξενούμενης (2)"]
OUTCOME_SHORT = ["1", "X", "2"]


# --------------------------------------------------------------------------- #
# cached loaders                                                              #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Φόρτωση εκπαιδευμένων μοντέλων...")
def load_artifacts():
    return load_runtime_artifacts(FINAL_CONFIG)


@st.cache_resource(show_spinner="Φόρτωση δεδομένων πρωταθλήματος...")
def load_state(league: str):
    params = load_artifacts()[0]
    return get_league_runtime_state(league, params)


def eval_file(exp: str) -> Path:
    return FINAL_CONFIG.artifacts_dir / f"final_probability_quality_{exp}.csv"


def list_eval_experiments() -> list[str]:
    """Experiments relevant to the tool that have a stored evaluation report:
    the canonical experiment the thesis cites, plus the user's own retrain."""
    return [e for e in (CANONICAL_EXP, "user_retrain") if eval_file(e).exists()]


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
def dataset_summary(token: str) -> pd.DataFrame:
    from src.data_processing import load_league_data

    rows = []
    for lg in LEAGUES:
        df = load_league_data(lg)
        dates = pd.to_datetime(df["date"])
        uf = user_file(lg)
        user_rows = max(0, sum(1 for _ in uf.open(encoding="utf-8")) - 1) if uf.exists() else 0
        rows.append({
            "Πρωτάθλημα": LEAGUE_LABELS.get(lg, lg),
            "Αγώνες (παιγμένοι)": int(df["is_played"].sum()),
            "Από": dates.min().date().isoformat(),
            "Έως": dates.max().date().isoformat(),
            "Γραμμές χρήστη": user_rows,
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


def run_prediction(league: str, home: str, away: str, oh: float, od: float, oa: float):
    state = load_state(league)
    (_params, meta_model, meta_cfg, mlp_model, mlp_meta,
     logreg_model, logreg_meta, blend_cfg) = load_artifacts()
    return predict_custom_match(
        home, away, oh, od, oa, state,
        meta_model, meta_cfg, mlp_model, mlp_meta, blend_cfg,
        logreg_model=logreg_model, logreg_meta=logreg_meta,
    )


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def confidence_band(probs: np.ndarray) -> tuple[str, float]:
    p = np.asarray(probs, dtype=float)
    top = float(p.max())
    if top >= 0.55:
        return "Υψηλή", top
    if top >= 0.45:
        return "Μέτρια", top
    return "Χαμηλή", top


def probs_table(res: dict) -> pd.DataFrame:
    rows = []
    for key, label in MODEL_LABELS.items():
        if key not in res:
            continue
        p = np.asarray(res[key], dtype=float)
        rows.append({
            "Μοντέλο": label,
            "1 (%)": round(100 * p[0], 1),
            "X (%)": round(100 * p[1], 1),
            "2 (%)": round(100 * p[2], 1),
            "Πρόβλεψη": OUTCOME_SHORT[int(p.argmax())],
            "Confidence (%)": round(100 * float(p.max()), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# pages                                                                       #
# --------------------------------------------------------------------------- #
def page_predict():
    st.header("Πρόβλεψη Αγώνα")
    st.caption(
        "Επίλεξε πρωτάθλημα, ομάδες και (προαιρετικά) αποδόσεις. "
        "Διάλεξε ποιο μοντέλο εμπιστεύεσαι — το εργαλείο δίνει πιθανότητες 1/X/2 "
        "και επίπεδο εμπιστοσύνης."
    )

    col_l, col_m = st.columns([1, 1])
    with col_l:
        league = st.selectbox(
            "Πρωτάθλημα", LEAGUES,
            format_func=lambda x: LEAGUE_LABELS.get(x, x),
        )
    teams = league_teams(league, _dataset_token())
    st.caption(f"Ομάδες τρέχουσας σεζόν ({len(teams)})")

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("Γηπεδούχος", teams, key="home")
    with c2:
        default_away = 1 if len(teams) > 1 else 0
        away = st.selectbox("Φιλοξενούμενη", teams, index=default_away, key="away")

    st.markdown("**Αποδόσεις αγοράς** (προαιρετικό — χρειάζονται για τα μοντέλα *Αγορά / Διορθωμένη Αγορά / Ensemble*)")
    o1, o2, o3 = st.columns(3)
    with o1:
        oh = st.number_input("Απόδοση 1 (Γηπεδούχος)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    with o2:
        od = st.number_input("Απόδοση X (Ισοπαλία)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    with o3:
        oa = st.number_input("Απόδοση 2 (Φιλοξενούμενη)", min_value=0.0, value=0.0, step=0.05, format="%.2f")

    model_key = st.selectbox(
        "Μοντέλο πρόβλεψης",
        list(MODEL_LABELS.keys()),
        index=list(MODEL_LABELS.keys()).index("ensemble"),
        format_func=lambda k: MODEL_LABELS[k],
        help="Το backend υπολογίζει όλα τα μοντέλα· εδώ επιλέγεις ποιο θα προβληθεί ως κύρια πρόβλεψη.",
    )

    if home == away:
        st.warning("Ο γηπεδούχος και η φιλοξενούμενη δεν μπορεί να είναι η ίδια ομάδα.")
        return

    if not st.button("Πρόβλεψη", type="primary"):
        return

    try:
        res = run_prediction(league, home, away, oh, od, oa)
    except Exception as exc:  # noqa: BLE001 - surface any backend error to the UI
        st.error(f"Σφάλμα πρόβλεψης: {exc}")
        return

    used_market = oh > 1.0 and od > 1.0 and oa > 1.0
    if model_key in {"market", "market_corr", "ensemble"} and not used_market:
        st.info(
            "Δεν δόθηκαν έγκυρες αποδόσεις, οπότε τα μοντέλα που βασίζονται στην αγορά "
            "πέφτουν πίσω στις πιθανότητες του στατιστικού μοντέλου."
        )

    probs = np.asarray(res[model_key], dtype=float)
    pick_idx = int(probs.argmax())
    band, top = confidence_band(probs)

    st.subheader(f"{home} vs {away}")
    st.markdown(f"**Μοντέλο:** {MODEL_LABELS[model_key]}")

    m1, m2, m3 = st.columns(3)
    m1.metric(OUTCOME_LABELS[0], f"{100 * probs[0]:.1f}%")
    m2.metric(OUTCOME_LABELS[1], f"{100 * probs[1]:.1f}%")
    m3.metric(OUTCOME_LABELS[2], f"{100 * probs[2]:.1f}%")

    st.success(f"Πρόβλεψη: **{OUTCOME_LABELS[pick_idx]}** — Εμπιστοσύνη: **{band}** ({100 * top:.1f}%)")

    e1, e2 = st.columns(2)
    e1.metric("Elo Γηπεδούχου", f"{res['elo'][0]:.0f}")
    e2.metric("Elo Φιλοξενούμενης", f"{res['elo'][1]:.0f}")
    e1.metric("Αναμενόμενα γκολ (Γηπ.)", f"{res['xg'][0]:.2f}")
    e2.metric("Αναμενόμενα γκολ (Φιλ.)", f"{res['xg'][1]:.2f}")

    st.markdown("**Πιθανότερα σκορ**")
    st.table(pd.DataFrame(
        [{"Σκορ": f"{hg}-{ag}", "Πιθανότητα (%)": round(100 * psc, 1)} for (hg, ag), psc in res["scores"]]
    ))

    with st.expander("Σύγκριση όλων των μοντέλων"):
        st.dataframe(probs_table(res), hide_index=True, width="stretch")


def page_evaluation():
    st.header("Αξιολόγηση & Επιλογή Μοντέλου")
    st.caption(
        "Αποθηκευμένα αποτελέσματα αξιολόγησης από το backtest. "
        "Χαμηλότερο log loss / Brier / ECE = καλύτερο μοντέλο. "
        "Χρησιμοποίησέ τα για να επιλέξεις μοντέλο στην καρτέλα Πρόβλεψη."
    )

    experiments = list_eval_experiments()
    c1, c2 = st.columns([2, 1])
    with c1:
        exp = st.selectbox(
            "Πείραμα", experiments,
            format_func=lambda e: f"{e} (canonical)" if e == CANONICAL_EXP else e,
            help="Το canonical είναι το πείραμα που αναφέρει η εργασία· το user_retrain παράγεται από την καρτέλα Εκπαίδευσης.",
        )
    with c2:
        split = st.radio(
            "Σύνολο", ["test", "validation"], horizontal=True,
            format_func=lambda s: "Test" if s == "test" else "Validation",
        )

    try:
        df = load_latest_eval(exp, split)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Δεν βρέθηκαν αποτελέσματα αξιολόγησης: {exc}")
        return

    # Build the table from whatever columns this experiment's report actually has,
    # so older reports with a leaner schema still render instead of crashing.
    col_labels = {
        "model": "Μοντέλο", "matches": "Αγώνες", "logloss": "Log loss",
        "brier": "Brier", "ece": "ECE", "accuracy": "Accuracy",
        "macro_f1": "Macro F1", "draw_recall": "Draw recall",
        "draw_pick_rate": "Draw pick rate",
    }
    present = [c for c in col_labels if c in df.columns]
    if "model" not in present or "logloss" not in present:
        st.error("Το αρχείο αξιολόγησης δεν έχει τις βασικές στήλες (model/logloss).")
        return

    show = df[present].copy()
    show["model"] = show["model"].map(lambda m: MODEL_LABELS.get(m, m))
    show = show.rename(columns=col_labels).sort_values("Log loss").reset_index(drop=True)

    best_model = show.iloc[0]["Μοντέλο"]
    best_ll = show.iloc[0]["Log loss"]
    st.success(f"Χαμηλότερο log loss: **{best_model}** ({best_ll:.4f})")

    fmt = {label: "{:.4f}" if label in ("Log loss", "Brier", "ECE") else "{:.3f}"
           for label in ("Log loss", "Brier", "ECE", "Accuracy", "Macro F1", "Draw recall", "Draw pick rate")
           if label in show.columns}
    highlight = [c for c in ("Log loss", "Brier", "ECE") if c in show.columns]
    styler = show.style.format(fmt)
    if highlight:
        styler = styler.highlight_min(subset=highlight, color="#1b5e20")
    st.dataframe(styler, hide_index=True, width="stretch")

    st.caption(
        "Σημείωση: Log loss/Brier/ECE μετρούν την ποιότητα των *πιθανοτήτων* "
        "(βαθμονόμηση), όχι μόνο το ποσοστό επιτυχίας. Το draw pick rate δείχνει "
        "πόσο συχνά ένα μοντέλο επιλέγει «ισοπαλία» ως πιο πιθανή έκβαση."
    )


def page_train():
    st.header("Εκπαίδευση / Επέκταση Δεδομένων")

    if st.session_state.get("last_add"):
        st.success(st.session_state.pop("last_add"))

    # --- 1) dataset (FR11) ------------------------------------------------- #
    st.subheader("1) Σύνολο δεδομένων")
    st.dataframe(dataset_summary(_dataset_token()), hide_index=True, width="stretch")

    st.markdown("**Προσθήκη αγώνα στο dataset** (γράφεται σε ξεχωριστό αρχείο `zz_user_added.csv` — τα αρχικά δεδομένα δεν αλλάζουν)")
    league = st.selectbox(
        "Πρωτάθλημα", LEAGUES,
        format_func=lambda x: LEAGUE_LABELS.get(x, x), key="train_league",
    )
    teams = league_teams(league, _dataset_token())
    with st.form("add_match", clear_on_submit=True):
        a, b = st.columns(2)
        home = a.selectbox("Γηπεδούχος", teams, key="tr_home")
        away = b.selectbox("Φιλοξενούμενη", teams, index=min(1, len(teams) - 1), key="tr_away")
        c, d_, e = st.columns(3)
        match_date = c.date_input("Ημερομηνία", value=date.today())
        fthg = d_.number_input("Γκολ γηπεδούχου", min_value=0, max_value=30, value=0, step=1)
        ftag = e.number_input("Γκολ φιλοξενούμενης", min_value=0, max_value=30, value=0, step=1)
        submitted = st.form_submit_button("Προσθήκη γραμμής")
    if submitted:
        if home == away:
            st.warning("Ο γηπεδούχος και η φιλοξενούμενη δεν μπορεί να είναι η ίδια ομάδα.")
        else:
            path = append_user_match(league, match_date, home, away, int(fthg), int(ftag))
            st.session_state["last_add"] = (
                f"Προστέθηκε: {home} {int(fthg)}–{int(ftag)} {away} "
                f"({match_date.isoformat()}) → {path.name}"
            )
            st.rerun()

    st.divider()

    # --- 2) retrain (FR10) ------------------------------------------------- #
    st.subheader("2) (Επαν)εκπαίδευση μοντέλων")
    st.caption(
        "Τρέχει ολόκληρο το pipeline σε **ξεχωριστό** πείραμα «user_retrain», ώστε τα "
        "canonical αποτελέσματα που αναφέρει η εργασία να μην αλλάζουν. Επαναχρησιμοποιεί "
        "τις βελτιστοποιημένες υπερπαραμέτρους και ξανα-εκπαιδεύει τα μοντέλα στα τρέχοντα "
        "(+ προστιθέμενα) δεδομένα. Αναμένεται να διαρκέσει μερικά λεπτά."
    )
    if st.button("Εκπαίδευση (retrain)", type="primary"):
        log_box = st.empty()
        lines: list[str] = []
        cmd = [sys.executable, str(ROOT / "retrain_runner.py"),
               "--experiment", "user_retrain", "--reset"]
        with st.spinner("Εκπαίδευση σε εξέλιξη... (μην κλείσεις τη σελίδα)"):
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
            st.success(
                "Η εκπαίδευση ολοκληρώθηκε. Δες τα αποτελέσματα στην καρτέλα "
                "«Αξιολόγηση Μοντέλων» → πείραμα **user_retrain**."
            )
        else:
            st.error("Η εκπαίδευση απέτυχε — δες το log παραπάνω.")


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Πρόβλεψη Ποδοσφαιρικών Αγώνων", page_icon="⚽", layout="wide")
    st.title("⚽ Εργαλείο Πρόβλεψης Αποτελεσμάτων Ποδοσφαιρικών Αγώνων")

    with st.sidebar:
        st.markdown("### Πλοήγηση")
        page = st.radio(
            "Σελίδα",
            ["Πρόβλεψη Αγώνα", "Αξιολόγηση Μοντέλων", "Εκπαίδευση / Δεδομένα"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            f"Πείραμα: `{FINAL_CONFIG.experiment_name}`\n\n"
            "Opening odds, leakage-safe χρονικά splits. "
            "5 πρωταθλήματα (Αγγλία, Ισπανία, Ιταλία, Γερμανία, Γαλλία)."
        )

    if page == "Πρόβλεψη Αγώνα":
        page_predict()
    elif page == "Αξιολόγηση Μοντέλων":
        page_evaluation()
    else:
        page_train()


if __name__ == "__main__":
    main()
