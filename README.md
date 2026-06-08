# Football Match Result Prediction Tool

A configurable tool that predicts the result of a football match (home win / draw /
away win) for the top-5 European leagues: England, Spain, Italy, Germany, and France.

Developed as a diploma thesis in Computer and Informatics Engineering.

## What this is

The user picks a league and the two teams (optionally also the bookmaker odds and
which model to trust), and the tool returns the 1/X/2 probabilities together with a
confidence level. Behind the interface it combines several machine-learning
approaches rather than a single algorithm:

- a **base statistical model**: Elo ratings + a Poisson goal model with the
  Dixon-Coles correction;
- the **bookmaker market** (odds converted to probabilities);
- **learned models** trained on historical matches: XGBoost, a neural network (MLP),
  and Logistic Regression;
- an **ensemble** that blends the above.

The user can compare the models, see which one is preferred, and **retrain the models
on their own data** (by adding matches through the interface). The interface is
bilingual — Greek or English, selectable from the sidebar (defaults to Greek).

## Setup

Requires Python 3.10 or newer.

### Windows

```powershell
git clone <repository-url>
cd Football-predictive-model-for-betting-purposes
.\setup.ps1
.\venv\Scripts\activate
```

### Linux / macOS

```bash
git clone <repository-url>
cd Football-predictive-model-for-betting-purposes
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Manual install

If you prefer not to use the setup scripts:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## Using the tool

The trained models are not stored in the repository, so generate them once first
(this runs the training/evaluation pipeline and takes a few minutes):

```bash
python scripts/main.py                 # canonical model
python scripts/train_context_variant.py  # optional: feature-rich variant
```

Then launch the interactive application:

```bash
streamlit run app.py
```

Pick the interface language (Greek / English) in the sidebar. The application has
three pages:

- **Match prediction** — choose a league, the home and away team, optionally the
  1/X/2 odds, and the model you want to use. The tool shows the outcome
  probabilities, a confidence level, the teams' Elo and expected goals, and the
  most likely scorelines.
- **Model evaluation** — shows the evaluation metrics for each model so the user can
  decide which one to trust.
- **Training & data** — add your own matches to the dataset and retrain the models
  on the updated data.

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
future information to predict the past.

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
