from pathlib import Path
import numpy as np
import pandas as pd

from src.team_names import normalize_team_name
from src.understat_data import UNDERSTAT_VALUE_COLUMNS, add_understat_xg
from src.external_context import EXTERNAL_CONTEXT_VALUE_COLUMNS, add_external_match_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLOSING_COLS = ("PSCH", "PSCD", "PSCA")
OPENING_AVG_COLS = ("AvgH", "AvgD", "AvgA")
CLOSING_AVG_COLS = ("AvgCH", "AvgCD", "AvgCA")
OVER_UNDER_COLS = (
    ("AvgC>2.5", "AvgC<2.5"),
    ("Avg>2.5", "Avg<2.5"),
    ("B365C>2.5", "B365C<2.5"),
    ("B365>2.5", "B365<2.5"),
)
AH_LINE_COLS = ("AHCh", "AHh")
MATCH_STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR", "HF", "AF"]

BOOK_TRIPLES = [
    ("B365H", "B365D", "B365A"),
    ("BWH", "BWD", "BWA"),
    ("GBH", "GBD", "GBA"),
    ("IWH", "IWD", "IWA"),
    ("LBH", "LBD", "LBA"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("SJH", "SJD", "SJA"),
    ("VCH", "VCD", "VCA"),
    ("BSH", "BSD", "BSA"),
]

REQUIRED_BASE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def _valid_odds_triplet(oh, od, oa):
    if not (np.isfinite(oh) and np.isfinite(od) and np.isfinite(oa)):
        return False
    if oh <= 1.0001 or od <= 1.0001 or oa <= 1.0001:
        return False
    return True


def _odds_to_fair_probs(oh, od, oa):
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa], dtype=float)
    s = inv.sum()
    if s <= 0 or not np.isfinite(s):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return inv / s


def _fair_probs_to_odds(p):
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        return (np.nan, np.nan, np.nan)

    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()

    return (float(1.0 / p[0]), float(1.0 / p[1]), float(1.0 / p[2]))


def _pick_best_or_avg_odds_row(row, available_cols):
    # 1) Pinnacle closing
    if all(c in available_cols for c in CLOSING_COLS):
        oh = pd.to_numeric(row.get(CLOSING_COLS[0]), errors="coerce")
        od = pd.to_numeric(row.get(CLOSING_COLS[1]), errors="coerce")
        oa = pd.to_numeric(row.get(CLOSING_COLS[2]), errors="coerce")

        if _valid_odds_triplet(oh, od, oa):
            return float(oh), float(od), float(oa)

    # 2) Average across books in fair-probability space
    probs_list = []
    for h, d, a in BOOK_TRIPLES:
        if h in available_cols and d in available_cols and a in available_cols:
            oh = pd.to_numeric(row.get(h), errors="coerce")
            od = pd.to_numeric(row.get(d), errors="coerce")
            oa = pd.to_numeric(row.get(a), errors="coerce")

            if _valid_odds_triplet(oh, od, oa):
                probs_list.append(_odds_to_fair_probs(float(oh), float(od), float(oa)))

    if len(probs_list) == 0:
        return (np.nan, np.nan, np.nan)

    p_avg = np.nanmean(np.vstack(probs_list), axis=0)
    return _fair_probs_to_odds(p_avg)


def _pick_triplet(row, available_cols, preferred_cols, fallback_book_triples=()):
    if all(c in available_cols for c in preferred_cols):
        oh = pd.to_numeric(row.get(preferred_cols[0]), errors="coerce")
        od = pd.to_numeric(row.get(preferred_cols[1]), errors="coerce")
        oa = pd.to_numeric(row.get(preferred_cols[2]), errors="coerce")
        if _valid_odds_triplet(oh, od, oa):
            return float(oh), float(od), float(oa)

    probs_list = []
    for h, d, a in fallback_book_triples:
        if h in available_cols and d in available_cols and a in available_cols:
            oh = pd.to_numeric(row.get(h), errors="coerce")
            od = pd.to_numeric(row.get(d), errors="coerce")
            oa = pd.to_numeric(row.get(a), errors="coerce")
            if _valid_odds_triplet(oh, od, oa):
                probs_list.append(_odds_to_fair_probs(float(oh), float(od), float(oa)))

    if not probs_list:
        return (np.nan, np.nan, np.nan)
    return _fair_probs_to_odds(np.nanmean(np.vstack(probs_list), axis=0))


def _pick_over_under_25(row, available_cols):
    for over_col, under_col in OVER_UNDER_COLS:
        if over_col in available_cols and under_col in available_cols:
            over = pd.to_numeric(row.get(over_col), errors="coerce")
            under = pd.to_numeric(row.get(under_col), errors="coerce")
            if np.isfinite(over) and np.isfinite(under) and over > 1.0001 and under > 1.0001:
                inv = np.array([1.0 / over, 1.0 / under], dtype=float)
                inv_sum = inv.sum()
                if inv_sum > 0:
                    return float(inv[0] / inv_sum)
    return np.nan


def _pick_over_under_25_odds(row, available_cols):
    for over_col, under_col in OVER_UNDER_COLS:
        if over_col in available_cols and under_col in available_cols:
            over = pd.to_numeric(row.get(over_col), errors="coerce")
            under = pd.to_numeric(row.get(under_col), errors="coerce")
            if np.isfinite(over) and np.isfinite(under) and over > 1.0001 and under > 1.0001:
                return float(over), float(under)
    return (np.nan, np.nan)


def _pick_ah_line(row, available_cols):
    for col in AH_LINE_COLS:
        if col in available_cols:
            value = pd.to_numeric(row.get(col), errors="coerce")
            if np.isfinite(value):
                return float(value)
    return np.nan


def load_league_data(league_name):
    data_path = PROJECT_ROOT / "data" / "raw" / league_name
    files = list(data_path.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_path}\n"
            f"Make sure you have downloaded the data for {league_name}."
        )

    dfs = []

    for file in sorted(files):
        df = pd.read_csv(file)

        missing = [c for c in REQUIRED_BASE if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{file.name}] Missing required columns {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )

        want_cols = set(REQUIRED_BASE)

        optional_cols = set(MATCH_STAT_COLS) | set(OPENING_AVG_COLS) | set(CLOSING_AVG_COLS) | set(CLOSING_COLS) | set(AH_LINE_COLS)
        for over_col, under_col in OVER_UNDER_COLS:
            optional_cols.update([over_col, under_col])

        for c in optional_cols:
            if c in df.columns:
                want_cols.add(c)

        for h, d, a in BOOK_TRIPLES:
            if h in df.columns and d in df.columns and a in df.columns:
                want_cols.update([h, d, a])

        df = df[list(want_cols)].copy()

        df = df.rename(columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        })
        df["home_team"] = df["home_team"].map(lambda x: normalize_team_name(x, league_name))
        df["away_team"] = df["away_team"].map(lambda x: normalize_team_name(x, league_name))

        # parse dates
        dt1 = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        dt2 = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        df["date"] = dt1.fillna(dt2)

        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, "date"], dayfirst=True, errors="coerce"
            )

        df = df.dropna(subset=["date"])
        if df.empty:
            # An empty CSV (e.g. a fixtures file once the season is over and there
            # are no upcoming matches) would otherwise break the result_type="expand"
            # odds apply below, which cannot infer three columns from zero rows.
            continue

        # keep played + future fixtures
        df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
        df["is_played"] = df["home_goals"].notna() & df["away_goals"].notna()

        available_cols = set(df.columns)

        odds = df.apply(
            lambda r: _pick_best_or_avg_odds_row(r, available_cols),
            axis=1,
            result_type="expand"
        )
        odds.columns = ["odds_home", "odds_draw", "odds_away"]

        opening_odds = df.apply(
            lambda r: _pick_triplet(r, available_cols, OPENING_AVG_COLS, BOOK_TRIPLES),
            axis=1,
            result_type="expand",
        )
        opening_odds.columns = ["open_odds_home", "open_odds_draw", "open_odds_away"]

        closing_odds = df.apply(
            lambda r: _pick_triplet(r, available_cols, CLOSING_AVG_COLS, [CLOSING_COLS]),
            axis=1,
            result_type="expand",
        )
        closing_odds.columns = ["close_odds_home", "close_odds_draw", "close_odds_away"]

        df = pd.concat([df, odds, opening_odds, closing_odds], axis=1)

        for col in [
            "odds_home",
            "odds_draw",
            "odds_away",
            "open_odds_home",
            "open_odds_draw",
            "open_odds_away",
            "close_odds_home",
            "close_odds_draw",
            "close_odds_away",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        stat_renames = {
            "HS": "home_shots",
            "AS": "away_shots",
            "HST": "home_shots_target",
            "AST": "away_shots_target",
            "HC": "home_corners",
            "AC": "away_corners",
            "HY": "home_yellows",
            "AY": "away_yellows",
            "HR": "home_reds",
            "AR": "away_reds",
            "HF": "home_fouls",
            "AF": "away_fouls",
        }
        df = df.rename(columns=stat_renames)
        stat_output_cols = list(stat_renames.values())
        for col in stat_output_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["ou25_over_prob"] = df.apply(lambda r: _pick_over_under_25(r, available_cols), axis=1)
        ou25_odds = df.apply(
            lambda r: _pick_over_under_25_odds(r, available_cols),
            axis=1,
            result_type="expand",
        )
        ou25_odds.columns = ["ou25_over_odds", "ou25_under_odds"]
        df = pd.concat([df, ou25_odds], axis=1)
        df["ah_line"] = df.apply(lambda r: _pick_ah_line(r, available_cols), axis=1)
        df["ou25_over_prob"] = pd.to_numeric(df["ou25_over_prob"], errors="coerce")
        df["ou25_over_odds"] = pd.to_numeric(df["ou25_over_odds"], errors="coerce")
        df["ou25_under_odds"] = pd.to_numeric(df["ou25_under_odds"], errors="coerce")
        df["ah_line"] = pd.to_numeric(df["ah_line"], errors="coerce")

        dfs.append(
            df[[
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
                *stat_output_cols,
                "is_played",
            ]]
        )

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)

    # remove exact duplicates if fixture files overlap with historical rows
    out = out.drop_duplicates(
        subset=["date", "home_team", "away_team"],
        keep="first"
    ).reset_index(drop=True)

    out = add_understat_xg(out, league_name)
    for col in UNDERSTAT_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = add_external_match_context(out, league_name)
    for col in EXTERNAL_CONTEXT_VALUE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col.endswith("_available") else np.nan
    return out
