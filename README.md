# Football Match Prediction and Value Betting Meta-Model

Machine learning system for predicting football match outcomes across the top 5 European leagues: England, Spain, Italy, Germany, and France. The pipeline combines Elo ratings, a Poisson goal model with Dixon-Coles adjustment, market odds, and stacked meta-models to estimate match probabilities and evaluate value betting strategies.

Developed as part of a diploma thesis in Computer and Informatics Engineering.

## Features

- Historical and fixture data sync from external CSV sources
- Optional Understat team-match xG sync with cross-source team-name normalization
- Optional pre-match context from lineups, injuries, suspensions, manager changes, and weather
- Dynamic Elo ratings with home advantage, goal-margin updates, and promoted-team initialization
- Recency-weighted Poisson team strengths with Dixon-Coles low-score correction
- Meta-features built from model probabilities, market probabilities, Elo differences, expected goals, rolling attacking/defensive match stats, and optional context
- XGBoost meta-model plus MLP baseline
- Probability calibration with temperature scaling
- Ensemble blending across base model, market, XGBoost, and MLP outputs
- Fixed temporal train / validation / test split with per-league chronological processing
- Explicit opening/closing odds snapshots for market features and simulated bet prices
- Leakage-safe single-season backtests with season-specific cache manifests
- CLI diagnostics for tuned league parameters, blend weights, and tuned-vs-baseline parameter impact
- Evaluation with log loss, Brier score, ECE, accuracy, macro F1, per-class precision/recall, betting simulation, and audit reports
- Validation-locked bet selection with ROI by edge, confidence, odds bucket, league, and pick type
- Alternative-market audits for double chance, draw-no-bet, over/under 2.5, and Asian handicap coverage
- Upcoming matchday predictions from dedicated fixture files

## Repository Structure

- `app.py`: Streamlit interactive tool (match prediction, model evaluation, dataset extension, and retraining)
- `requirements.txt`: pinned runtime dependencies for the interactive tool
- `main.py`: training and evaluation pipeline entry point
- `backtest_season.py`: leakage-safe single-season betting backtest entry point
- `predict_match.py`: interactive custom match predictor entry point
- `retrain_runner.py`: retrain into a separate experiment, leaving canonical artifacts untouched
- `analyze_ci.py`: bootstrap confidence intervals (log loss vs market, ROI, CLV) for season backtests
- `update_team_news.py`: optional API-Football team-news updater
- `src/update_understat.py`: optional Understat xG refresh command
- `src/update_weather_context.py`: optional Open-Meteo weather context updater
- `src/trainer.py`: end-to-end pipeline orchestration
- `src/bet_selection.py`: validation-locked bet filters and alternative-market audits
- `src/final_report.py`: console summary for the saved final reports
- `src/thesis_report.py`: thesis-ready markdown/CSV results summary from saved artifacts
- `src/literature_audit.py`: fast literature-grounded benchmark of public-feature models vs the market
- `src/predictor.py`: runtime predictor utilities
- `src/services/upcoming.py`: upcoming matchday predictions

## Installation

Requires Python 3.10+.

### Windows

1. Clone the repository.
2. Run:

```powershell
.\setup.ps1
```

3. Activate the environment:

```powershell
.\venv\Scripts\activate
```

### Linux / macOS

1. Clone the repository.
2. Make the setup script executable.
3. Run:

```bash
chmod +x setup.sh
./setup.sh
```

4. Activate the environment:

```bash
source venv/bin/activate
```

## Usage

### Interactive Prediction Tool (Streamlit)

The main deliverable is an interactive tool. Install the runtime dependencies and launch it:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The tool has three pages (the UI is in Greek):

- **Πρόβλεψη Αγώνα** (match prediction): pick a league, the two teams, optional 1X2 odds, and which model to trust; it returns 1/X/2 probabilities, a confidence level, Elo, expected goals, and the most likely scorelines.
- **Αξιολόγηση Μοντέλων** (model evaluation): stored probability-quality metrics (log loss, Brier, ECE, accuracy, macro F1) per model, to help decide which model to use.
- **Εκπαίδευση / Δεδομένα** (training & data): append your own matches to a local user CSV and retrain into a separate `user_retrain` experiment, leaving the canonical thesis numbers untouched.

The tool loads the cached artifacts of the `final_opening_market_pre_match` experiment (opening odds, leakage-safe temporal splits).

### 1. Update Data

Download the latest historical results and future fixtures:

```bash
python src/update_data.py
```

Refresh optional Understat expected-goals data when that external dataset is needed:

```bash
python src/update_understat.py
```

Optional enrichment files live in `data/external/` and stay local by default. See [`data/external/README.md`](data/external/README.md) for the expected schemas and the weather/API-Football update commands.

### 2. Run the Main Pipeline

Runs the final configured pipeline:

```bash
python main.py
```

`main.py` uses `FINAL_CONFIG` from `src/config.py`. The final config is an opening-odds pre-match setup: market probabilities and simulated bet prices come from opening odds, and closing-movement features are disabled. It loads compatible cached tuning artifacts when possible, but the current final config still refits the final XGBoost/MLP models and retunes the blend unless you change those config flags.

During each league pass the CLI prints the active data split and the league-level parameters in this format:

```text
----------------------------------------------------------------
LEAGUE  ENGLAND
----------------------------------------------------------------
Rows    train_fit=4,560 | validation=380 | test=349
Params  cached
        Elo:   K=70  | home_adv=110
        Goals: beta=0.1300 | decay=0.0005 | rho=+0.1200
        Calib: T=2.0699
```

### 3. Predict a Custom Match

Run the root entry point from the project root:

```bash
python predict_match.py
```

### 4. Print Saved Final Reports

Print the latest saved summary tables without retraining:

```bash
python -m src.final_report
```

### 5. Run a Leakage-Safe Season Backtest

Evaluate one completed season with training and validation kept strictly earlier in time:

```bash
python backtest_season.py --season 2024
```

That command evaluates only the `2024-2025` season. It uses:

- fit history before `2023-07-01`
- validation/meta selection from `2023-07-01` through `2024-06-30`
- test-only betting from `2024-07-01` through `2025-06-30`

The first run for a season creates season-specific cached artifacts. Later reruns can reuse those caches safely because they belong to that exact pre-test window.

Useful flags:

- `--force-refit`: reuse cached season-specific tuning, but refit final models and retune the blend.
- `--force-retune`: retune league parameters, XGBoost, MLP, and blend settings from scratch.
- `--full-report`: print the long diagnostic tables in the terminal. By default they are written to CSV files.
- `--market-odds opening|closing`: choose which odds snapshot is converted to market-implied probabilities.
- `--betting-odds opening|closing`: choose which odds snapshot is used as the simulated bet price.
- `--no-market-movement`: disable closing-minus-opening movement features.

For a clean pre-match betting backtest, use:

```bash
python backtest_season.py --season 2024 --market-odds opening --betting-odds opening --no-market-movement
```

For a closing-market benchmark, use the default command:

```bash
python backtest_season.py --season 2024
```

Season backtests also print two compact diagnostics:

```text
Effective blend weights: {'base': 0.0, 'market': 1.0, 'xgb': 0.0, 'mlp': 0.0}

League parameter impact on base model:
league      matches  tuned_ll baseline_ll     delta avg_prob_diff
england        349    1.0382      1.1303   -0.0922        0.0949
ALL           1573    0.9953      1.0728   -0.0776        0.1089
```

This separates two questions that are easy to confuse: whether tuned league parameters improve the base Poisson/Elo model, and whether the final blend decides to use that base model or fall back to market probabilities.

## Methodology Overview

```text
Raw match data
  -> normalized team names and optional Understat / pre-match context merge
  -> Elo ratings
  -> Recency-weighted Poisson team strengths
  -> Dixon-Coles outcome probabilities
  -> Market implied probabilities
  -> Auxiliary features: Elo diff, expected goals, rest, form, rolling stats, context
  -> Meta-features
  -> XGBoost / MLP / blended ensemble
  -> Final probabilities
  -> Betting simulation and upcoming picks
```

## Evaluation

The main reported metrics are:

- Log loss
- Multiclass Brier score
- Expected calibration error
- Accuracy
- Macro F1
- Home/draw/away precision and recall

The training pipeline uses:

- a fixed date-based split defined in `src/config.py`
- league-level parameter tuning on the validation period
- early / late validation sub-splits for meta-model and blend tuning

Betting evaluation is a simulation, not a claim of deployable profitability. The implementation now separates:

- market odds snapshot: the odds used to build market-implied probabilities/features
- betting odds snapshot: the odds used as the simulated price taken
- market movement features: closing probabilities minus opening probabilities, only appropriate for closing-time experiments

The current implementation uses:

- edge threshold `0.05`
- fractional Kelly staking with `kelly_fraction=0.25`
- max stake cap `0.05`
- filters for extreme odds and extreme theoretical EV

Reported betting outputs include bet count, hit rate, profit, ROI, average odds, opening-to-closing value where prices are available, and additional audit tables. The current final artifacts include probability-quality, validation-locked bet-selector, bet-bucket, alternative-market, data-enrichment, robustness, and league-strategy CSV reports.

## Artifacts and Git Hygiene

Training and backtest runs write JSON, pickle, and CSV outputs under `artifacts/`. These are generated files and are ignored for new outputs by default. Existing tracked artifacts remain tracked until explicitly removed from Git history or the index.

Important artifact groups:

- `best_params_*`: tuned league-level Elo/Poisson/Dixon-Coles/calibration parameters
- `best_meta_*`, `meta_model_*`: XGBoost tuning and trained model
- `best_mlp_*`, `mlp_model_*`: MLP tuning and trained model
- `best_blend_*`: validation-selected blend weights
- `manifest_*`: cache compatibility records with config, feature schema, pipeline version, and data fingerprint
- `final_*`: report CSVs for model quality, probability quality, bet selection, robustness, and league strategy

Use `git add -f artifacts/<file>` only when you intentionally want to version a generated artifact.

## Data Sources

- Historical match results and bookmaker odds: `football-data.co.uk`
- Future fixtures: `fixturedownload.com`
- Optional xG, non-penalty xG, and expected-points match rows: `understat.com`
- Optional weather context: Open-Meteo historical weather
- Optional team news context: API-Football

## Notes

- Upcoming fixtures come from dedicated fixture CSV files.
- Upcoming picks are limited to the current or next available matchday window.
- Hyperparameter tuning uses Optuna when available; otherwise the active code falls back to a smaller manual search.
- Betting ROI is a backtest diagnostic, not a claim of future profitability.
- The README describes the active refactored path, not older duplicate modules that may have existed during development.

## Testing

Run the core behavior tests with:

```bash
python -m unittest tests.test_core_behaviors
```

## Thesis Context

This repository evolved from an earlier narrower prototype into a modular top-5-leagues project. The current active code path is the refactored one described above.
