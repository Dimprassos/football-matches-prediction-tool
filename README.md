# Football Match Result Prediction Tool

A configurable tool that predicts the result of a football match (home win / draw /
away win) for the top-5 European leagues: England, Spain, Italy, Germany, and France.

Developed as a diploma thesis in Computer and Informatics Engineering.

## What this is

The user picks a league and the two teams (optionally also the bookmaker odds and
which model to trust), and the tool returns the 1/X/2 probabilities together with a
confidence level. Behind the interface it combines several machine-learning
approaches rather than a single algorithm:

- a **base statistical model** — Elo ratings + a Poisson goal model with the
  Dixon-Coles correction;
- the **bookmaker market** — opening odds converted to probabilities (used both as a
  feature and as an external benchmark);
- **learned models** trained on historical matches — XGBoost, a neural network (MLP),
  and Logistic Regression;
- an **ensemble** that blends the classical models above;
- a **deep-learning model, "FootyNet"** — a recurrent late-fusion network (two
  shared-weight LSTM encoders over each team's last matches + a static feature
  branch), built with PyTorch;
- a **stacking blend** (`FootyNet + market`) whose weights are learned on the
  validation split — the project's exploration of *ensemble of a deep model with the
  market*.

The user can compare the models, see which one is preferred, and **retrain the models
on their own data** (by adding matches through the interface). The interface is
bilingual — Greek or English, selectable from the sidebar (defaults to Greek).

> **Scope.** This is a *prediction tool*: it produces calibrated probabilities and
> lets the user compare and configure models. It is **not** a betting-profit study.
> A model that matches but does not beat the bookmaker market is reported as an honest
> evaluation result, not a failure.

## Setup

Requires Python 3.10 or newer.

### Windows

```bat
git clone <repository-url>
cd football-matches-prediction-tool
setup.bat
```

`setup.bat` creates the `.venv` and installs the dependencies. Use it rather than
`.\setup.ps1` directly: by default Windows blocks running `.ps1` scripts
("running scripts is disabled on this system"), and the `.bat` wrapper sidesteps that.
After it finishes, activate the environment in your shell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation itself is blocked, allow it once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (or just call the venv's
interpreter directly, e.g. `.\.venv\Scripts\python.exe scripts\main.py`).

### Linux / macOS

```bash
git clone <repository-url>
cd football-matches-prediction-tool
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

### Manual install

If you prefer not to use the setup scripts:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\Activate.ps1
# Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

> The deep-learning model (FootyNet) needs PyTorch, which is listed in
> `requirements.txt`. If the default install is slow, the CPU-only wheel is enough:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Using the tool

The trained models are not stored in the repository, so generate them once first
(this runs the training/evaluation pipeline and takes a few minutes):

```bash
python scripts/main.py                     # canonical model (Elo+Poisson, XGBoost, MLP, LogReg, ensemble, market)
python scripts/train_footynet.py           # optional: deep-learning model + stacking blend
python scripts/train_context_variant.py    # optional: feature-rich (understat/form) variant
```

Then launch the interactive application:

```bash
streamlit run app.py
```

Pick the interface language (Greek / English) in the sidebar. The application has
four pages:

- **Match prediction** — choose an *experiment* (which family of models to use), a
  league, the home and away team, optionally the 1/X/2 odds, and the specific model.
  The tool shows the outcome probabilities, a confidence level, the value relative to
  the market, and — for the classical models — the teams' Elo, expected goals and the
  most likely scorelines.
- **Model evaluation** — shows the stored evaluation metrics for each model on the
  held-out test set, so the user can decide which one to trust.
- **Training & data** — add your own matches to the dataset and retrain the models on
  the updated data.
- **About / Methodology** — explains the pipeline, the data, how leakage is avoided,
  the metrics, and the role of the market benchmark.

### Command-line scripts (optional)

The interactive tool is the main deliverable, but the same pipeline can be run from
the command line. Run these from the project root:

```bash
python scripts/main.py                       # train and evaluate the models
python scripts/predict_match.py              # predict a single match in the terminal
python scripts/backtest_season.py --season 2024   # evaluate one past season
```

Refreshing the input data is optional (the repository already ships with data):

```bash
python src/update_data.py                    # download newer results and fixtures
python src/update_understat.py               # refresh optional expected-goals data
```

## How it is evaluated

Models are compared on a **fixed, time-based split**: older matches are used for
training, a later period for tuning and model selection, and the most recent matches
are held out for testing. Splitting by time (rather than randomly) avoids using
future information to predict the past. The bookmaker odds used are the **opening
odds**, which are known before kick-off, and the final score never enters the
features — so the evaluation is leakage-safe.

The quality of the predicted probabilities is measured with several metrics, so a
model is judged on how well-calibrated its probabilities are, not only on how often
its top pick is correct:

- **Log loss** and **Brier score** — quality of the probabilities;
- **Expected Calibration Error (ECE)** — whether stated confidence matches reality;
- **Accuracy** and **Macro F1** — classification quality across the three outcomes;
- **per-class precision and recall** (home / draw / away).

The tool also includes a betting simulation that compares the models against the
bookmaker market. This is used purely as an evaluation diagnostic, not as a claim of
profitability.

## Project structure

```
app.py                  Streamlit interface (the main deliverable) — 4 pages, bilingual
requirements.txt        Pinned runtime dependencies
setup.ps1 / setup.sh    One-shot environment setup (Windows / Linux-macOS)
data/
  raw/<league>/         Historical results + odds (football-data.co.uk); user-added rows
src/
  config.py             Experiment definitions (canonical, feature-rich, lineup, FootyNet)
  data_processing.py    Load + clean + merge the league datasets
  elo.py, poisson_model.py   Base statistical model (Elo + Poisson/Dixon-Coles)
  feature_builder.py    Engineered feature columns + market probabilities from odds
  state_builder.py      Leakage-safe, pre-match feature reconstruction
  models/
    meta.py             XGBoost / MLP / Logistic Regression / ensemble blend
    footynet.py         FootyNet — the recurrent late-fusion deep-learning model
  footynet_data.py      Aligns static features + match sequences + labels for FootyNet
  footynet_stack.py     Learns the FootyNet + market stacking weights (validation-only)
  footynet_serve.py     Serves FootyNet for one interactive fixture
  trainer.py            Training / tuning / evaluation pipeline + artifact caching
  predictor.py          Runtime artifact loading + interactive prediction
  metrics.py, evaluation.py   Evaluation metrics and betting diagnostics
scripts/
  main.py               Run the full training/evaluation pipeline (canonical)
  train_footynet.py     Train FootyNet + the stacking blend
  train_context_variant.py   Train the feature-rich variant
  predict_match.py, backtest_season.py   Command-line entry points
tests/
  test_core_behaviors.py     Unit tests for the core behaviours
```

## Testing

```bash
python -m unittest tests.test_core_behaviors
```

## Data sources

- Historical results and bookmaker odds: `football-data.co.uk`
- Future fixtures: `fixturedownload.com`
- Optional expected-goals (xG) data: `understat.com`
- Optional weather context: Open-Meteo
- Optional team-news context: API-Football
