import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifact_store import append_rows_to_csv
import src.update_api_football_context as api_football_context
import src.data_processing as data_processing
from update_team_news import _backtest_window
from src.cli.backtest_season_cli import build_backtest_config, season_window
from src.calibration import temperature_scale_probs
from src.config import ExperimentConfig
from src.evaluation import betting_records, simulate_value_betting
from src.external_context import add_external_match_context
from src.feature_builder import ensure_market_probs, feature_indices, market_probs_from_odds_row
from src.metrics import class_metric_summary
from src.models.base import _validation_rows_with_elo
from src.models.meta import blend_probabilities, market_logit_correction_probs, tune_market_logit_correction
from src.state_builder import EXTRA_AUX_LEN, compute_pre_match_extra_features, odds_triplet_from_row
from src.team_names import normalize_team_name
from src.trainer import (
    _best_betting_roi_row,
    _effective_blend_weights,
    _parameter_impact_row,
    _select_recommended_betting_model,
    _split_played_periods,
)
from src.understat_data import add_understat_xg
from src.update_api_football_context import (
    injury_fields,
    lineup_fields,
    match_api_fixtures,
    merge_match_context as merge_api_match_context,
    skip_existing_context_rows,
    team_match_key,
)
from src.update_weather_context import (
    _parse_kickoff_hour,
    load_team_locations,
    matches_with_locations,
    merge_match_context,
    write_team_location_template,
)


class FeatureBuilderTests(unittest.TestCase):
    def test_market_probs_normalize_valid_odds(self):
        probs = market_probs_from_odds_row(2.0, 3.5, 4.0)

        self.assertTrue(np.isfinite(probs).all())
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_market_probs_returns_nan_for_invalid_odds(self):
        probs = market_probs_from_odds_row(2.0, np.nan, 4.0)

        self.assertTrue(np.isnan(probs).all())

    def test_missing_market_probs_fall_back_to_model_probs(self):
        model = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
        market = np.array([[np.nan, np.nan, np.nan], [0.3, 0.3, 0.4]])

        fixed = ensure_market_probs(model, market)

        np.testing.assert_allclose(fixed[0], model[0])
        np.testing.assert_allclose(fixed[1], market[1])


class MetricTests(unittest.TestCase):
    def test_class_metric_summary_reports_draw_recall(self):
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.4, 0.3, 0.3],
        ])
        y_true = np.array([0, 1, 1])

        metrics = class_metric_summary(probs, y_true)

        self.assertAlmostEqual(metrics["draw_recall"], 0.5)
        self.assertGreater(metrics["macro_f1"], 0.0)


class TeamNameTests(unittest.TestCase):
    def test_fixture_team_aliases_match_historical_names(self):
        self.assertEqual(normalize_team_name("FC Barcelona", "spain"), "Barcelona")
        self.assertEqual(normalize_team_name("Rayo Vallecano", "spain"), "Vallecano")
        self.assertEqual(normalize_team_name("Borussia Dortmund", "germany"), "Dortmund")
        self.assertEqual(normalize_team_name("1. FC Heidenheim 1846", "germany"), "Heidenheim")

    def test_loader_normalizes_downloaded_fixture_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw" / "spain"
            raw_dir.mkdir(parents=True)
            pd.DataFrame([
                {
                    "Date": "01/01/2026",
                    "HomeTeam": "Barcelona",
                    "AwayTeam": "Alaves",
                    "FTHG": 2,
                    "FTAG": 0,
                },
                {
                    "Date": "08/01/2026",
                    "HomeTeam": "Alaves",
                    "AwayTeam": "FC Barcelona",
                    "FTHG": np.nan,
                    "FTAG": np.nan,
                },
            ]).to_csv(raw_dir / "SP1_2025_fixtures.csv", index=False)

            original_root = data_processing.PROJECT_ROOT
            original_add_understat_xg = data_processing.add_understat_xg
            try:
                data_processing.PROJECT_ROOT = root
                data_processing.add_understat_xg = lambda df, league_name: df
                loaded = data_processing.load_league_data("spain")
            finally:
                data_processing.PROJECT_ROOT = original_root
                data_processing.add_understat_xg = original_add_understat_xg

        self.assertIn("Barcelona", set(loaded["away_team"]))
        self.assertNotIn("FC Barcelona", set(loaded["away_team"]))


class SeasonBacktestTests(unittest.TestCase):
    def test_season_window_keeps_validation_before_target_season(self):
        self.assertEqual(
            season_window(2024),
            ("2023-07-01", "2024-07-01", "2025-07-01"),
        )

        config = build_backtest_config(2024)
        self.assertEqual(config.experiment_name, "season_backtest_2024_2025")
        self.assertEqual(config.train_cut, "2023-07-01")
        self.assertEqual(config.test_cut, "2024-07-01")
        self.assertEqual(config.test_end, "2025-07-01")
        self.assertTrue(config.allow_partial_param_cache)

    def test_opening_backtest_config_disables_market_movement_by_default(self):
        config = build_backtest_config(
            2024,
            market_odds_source="opening",
            betting_odds_source="opening",
        )

        self.assertEqual(
            config.experiment_name,
            "season_backtest_2024_2025_opening_market_opening_price_no_move",
        )
        self.assertEqual(config.market_odds_source, "opening")
        self.assertEqual(config.betting_odds_source, "opening")
        self.assertFalse(config.include_market_movement_features)
        self.assertTrue(config.print_parameter_impact)
        self.assertFalse(config.generate_upcoming_picks)

    def test_test_end_caps_backtest_to_exact_target_season(self):
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2023-06-30",
                "2023-07-01",
                "2024-06-30",
                "2024-07-01",
                "2025-06-30",
                "2025-07-01",
            ])
        })
        config = ExperimentConfig(
            train_cut="2023-07-01",
            test_cut="2024-07-01",
            test_end="2025-07-01",
        )

        train_fit, val, test = _split_played_periods(df, config)

        self.assertEqual(train_fit["date"].dt.strftime("%Y-%m-%d").tolist(), ["2023-06-30"])
        self.assertEqual(val["date"].dt.strftime("%Y-%m-%d").tolist(), ["2023-07-01", "2024-06-30"])
        self.assertEqual(test["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-07-01", "2025-06-30"])

    def test_team_news_backtest_window_defaults_to_test_period(self):
        self.assertEqual(_backtest_window(2023, "test"), (pd.Timestamp("2023-07-01").date(), pd.Timestamp("2024-06-30").date()))
        self.assertEqual(_backtest_window(2023, "validation"), (pd.Timestamp("2022-07-01").date(), pd.Timestamp("2023-06-30").date()))
        self.assertEqual(_backtest_window(2023, "both"), (pd.Timestamp("2022-07-01").date(), pd.Timestamp("2024-06-30").date()))


class ArtifactStoreTests(unittest.TestCase):
    def test_append_rows_to_csv_migrates_existing_header_for_new_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment_results.csv"
            path.write_text(
                "run_ts_utc,experiment_name,train_cut,test_cut,model,logloss\n"
                "2026-01-01T00:00:00+00:00,old,2023-07-01,2024-07-01,base,0.9\n",
                encoding="utf-8",
            )

            append_rows_to_csv(path, [{
                "run_ts_utc": "2026-01-02T00:00:00+00:00",
                "experiment_name": "new",
                "train_cut": "2023-07-01",
                "test_cut": "2024-07-01",
                "test_end": "2025-07-01",
                "model": "meta",
                "logloss": 0.8,
            }])

            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(
            list(rows[0].keys()),
            ["run_ts_utc", "experiment_name", "train_cut", "test_cut", "test_end", "model", "logloss"],
        )
        self.assertEqual(rows[0]["test_end"], "")
        self.assertEqual(rows[0]["model"], "base")
        self.assertEqual(rows[1]["test_end"], "2025-07-01")
        self.assertEqual(rows[1]["model"], "meta")


class ModelSelectionTests(unittest.TestCase):
    def test_temperature_scaling_identity_at_one(self):
        probs = np.array([
            [0.50, 0.30, 0.20],
            [0.20, 0.35, 0.45],
        ])

        scaled = temperature_scale_probs(probs, T=1.0)

        np.testing.assert_allclose(scaled, probs, atol=1e-12)

    def test_market_logit_correction_alpha_zero_returns_market(self):
        market = np.array([
            [0.50, 0.30, 0.20],
            [0.25, 0.35, 0.40],
        ])
        correction = np.array([
            [0.80, 0.10, 0.10],
            [0.10, 0.20, 0.70],
        ])

        out = market_logit_correction_probs(market, correction, alpha=0.0)

        np.testing.assert_allclose(out, market)

    def test_market_logit_correction_tuning_selects_helpful_source(self):
        y = np.array([0, 1, 2, 0])
        market = np.array([
            [0.45, 0.30, 0.25],
            [0.30, 0.40, 0.30],
            [0.30, 0.30, 0.40],
            [0.42, 0.31, 0.27],
        ])
        xgb = np.array([
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.75, 0.15, 0.10],
        ])

        cfg = tune_market_logit_correction(
            y,
            market,
            {"xgb": xgb},
            alpha_grid=[0.0, 0.5, 1.0],
            min_improvement=0.0,
        )

        self.assertEqual(cfg["source_model"], "xgb")
        self.assertGreater(cfg["alpha"], 0.0)
        self.assertTrue(cfg["accepted"])

    def test_effective_blend_weights_normalize_active_models(self):
        effective = _effective_blend_weights({
            "weights": {"base": 0.0, "market": 0.05, "xgb": 0.0, "mlp": 0.5},
            "mlp_allowed": False,
        })

        self.assertEqual(effective["market"], 1.0)
        self.assertEqual(effective["mlp"], 0.0)

    def test_parameter_impact_row_quantifies_probability_change(self):
        y = np.array([0, 1, 2])
        tuned = np.array([
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ])
        baseline = np.array([
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
        ])

        row = _parameter_impact_row("test", y, tuned, baseline, None)

        self.assertLess(row["tuned_logloss"], row["baseline_logloss"])
        self.assertLess(row["delta_logloss"], 0.0)
        self.assertGreater(row["avg_abs_prob_diff"], 0.0)

    def test_no_bet_market_does_not_win_betting_roi_or_recommendation(self):
        rows = [
            {
                "name": "market",
                "logloss": 0.95,
                "accuracy": 0.56,
                "bets": 0,
                "hit_rate": 0.0,
                "roi": 0.0,
                "profit": 0.0,
                "avg_odds": 0.0,
            },
            {
                "name": "meta",
                "logloss": 0.96,
                "accuracy": 0.55,
                "bets": 100,
                "hit_rate": 40.0,
                "roi": -7.0,
                "profit": -1.0,
                "avg_odds": 3.0,
            },
            {
                "name": "ensemble",
                "logloss": 0.953,
                "accuracy": 0.56,
                "bets": 13,
                "hit_rate": 15.0,
                "roi": -2.5,
                "profit": -0.1,
                "avg_odds": 8.0,
            },
        ]

        self.assertEqual(_best_betting_roi_row(rows)["name"], "ensemble")
        recommended, reason = _select_recommended_betting_model(rows, rows)
        self.assertEqual(recommended["name"], "no_bet")
        self.assertEqual(reason, "no_positive_validation_roi")

    def test_recommendation_locks_on_validation_but_reports_test_row(self):
        # logreg looks best on validation; ensemble only looks good on test.
        # The recommendation must follow validation, and report logreg's test row.
        validation_rows = [
            {"name": "market", "logloss": 0.95, "accuracy": 0.55, "bets": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0.0, "avg_odds": 0.0},
            {"name": "logreg", "logloss": 0.951, "accuracy": 0.55, "bets": 120, "hit_rate": 35.0, "roi": 4.0, "profit": 1.0, "avg_odds": 3.0},
            {"name": "ensemble", "logloss": 0.96, "accuracy": 0.55, "bets": 80, "hit_rate": 30.0, "roi": 1.0, "profit": 0.2, "avg_odds": 3.5},
        ]
        test_rows = [
            {"name": "market", "logloss": 0.95, "accuracy": 0.55, "bets": 0, "hit_rate": 0.0, "roi": 0.0, "profit": 0.0, "avg_odds": 0.0},
            {"name": "logreg", "logloss": 0.952, "accuracy": 0.55, "bets": 110, "hit_rate": 33.0, "roi": -1.5, "profit": -0.3, "avg_odds": 3.0},
            {"name": "ensemble", "logloss": 0.961, "accuracy": 0.55, "bets": 70, "hit_rate": 31.0, "roi": 9.9, "profit": 0.8, "avg_odds": 3.5},
        ]

        recommended, reason = _select_recommended_betting_model(validation_rows, test_rows)

        self.assertEqual(recommended["name"], "logreg")
        self.assertEqual(reason, "validation_locked_positive_roi")
        # Reported ROI is the honest out-of-sample test number, not the validation one.
        self.assertEqual(recommended["roi"], -1.5)


class WeatherContextTests(unittest.TestCase):
    def test_weather_location_merge_normalizes_home_team_names(self):
        matches = pd.DataFrame([{
            "date": "2024-08-16",
            "kickoff_hour": 20,
            "league": "england",
            "home_team": "Man United",
            "away_team": "Fulham",
        }])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "team_locations.csv"
            pd.DataFrame([{
                "league": "england",
                "team": "Man Utd",
                "latitude": 53.4631,
                "longitude": -2.2913,
            }]).to_csv(path, index=False)

            locations = load_team_locations(path)

        located, missing = matches_with_locations(matches, locations)

        self.assertTrue(missing.empty)
        self.assertAlmostEqual(float(located.loc[0, "latitude"]), 53.4631)
        self.assertAlmostEqual(float(located.loc[0, "longitude"]), -2.2913)

    def test_weather_context_merge_preserves_existing_team_news(self):
        existing = pd.DataFrame([{
            "date": "2024-08-16",
            "league": "england",
            "home_team": "Man United",
            "away_team": "Fulham",
            "lineup_available": 1,
            "home_lineup_strength": 0.94,
        }])
        weather = pd.DataFrame([{
            "date": "2024-08-16",
            "league": "england",
            "home_team": "Man United",
            "away_team": "Fulham",
            "weather_available": 1,
            "temperature_c": 18.5,
            "wind_kph": 12.0,
            "precipitation_mm": 0.4,
        }])

        merged = merge_match_context(existing, weather)

        self.assertEqual(float(merged.loc[0, "lineup_available"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "home_lineup_strength"]), 0.94)
        self.assertEqual(float(merged.loc[0, "weather_available"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "temperature_c"]), 18.5)

    def test_kickoff_hour_defaults_safely(self):
        self.assertEqual(_parse_kickoff_hour("20:45"), 20)
        self.assertEqual(_parse_kickoff_hour("bad"), 15)
        self.assertEqual(_parse_kickoff_hour(np.nan), 15)

    def test_write_team_location_template_uses_raw_match_teams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "team_locations.csv"
            template = write_team_location_template(
                output_file=path,
                start_season=2023,
                end_season=2023,
            )

        self.assertTrue({"league", "team", "latitude", "longitude"}.issubset(template.columns))
        self.assertIn("Arsenal", set(template[template["league"] == "england"]["team"]))


class ApiFootballContextTests(unittest.TestCase):
    def test_api_football_team_match_key_handles_common_aliases(self):
        self.assertEqual(team_match_key("Man United", "england"), team_match_key("Manchester United", "england"))
        self.assertEqual(team_match_key("Tottenham", "england"), team_match_key("Tottenham Hotspur", "england"))
        self.assertEqual(team_match_key("Paris Saint-Germain", "france"), team_match_key("Paris SG", "france"))

    def test_api_fixture_matching_accepts_provider_team_names(self):
        local = pd.DataFrame([{
            "date": "2026-05-17",
            "league": "england",
            "api_season": 2025,
            "home_team": "Man United",
            "away_team": "Tottenham",
            "is_played": False,
        }])
        api = pd.DataFrame([{
            "date": "2026-05-17",
            "league": "england",
            "api_fixture_id": 123,
            "api_home_team": "Manchester United",
            "api_away_team": "Tottenham Hotspur",
            "api_home_id": 33,
            "api_away_id": 47,
        }])

        matched = match_api_fixtures(local, api)

        self.assertEqual(matched.loc[0, "api_match_status"], "matched")
        self.assertEqual(int(matched.loc[0, "api_fixture_id"]), 123)

    def test_lineup_and_injury_payloads_become_context_fields(self):
        lineups = [
            {"team": {"id": 10}, "startXI": [{"player": {"id": i}} for i in range(11)]},
            {"team": {"id": 20}, "startXI": [{"player": {"id": i}} for i in range(10)]},
        ]
        injuries = [
            {"team": {"id": 10}, "player": {"id": 1, "reason": "Hamstring Injury"}},
            {"team": {"id": 20}, "player": {"id": 2, "reason": "Suspended"}},
            {"team": {"id": 20}, "player": {"id": 2, "reason": "Suspended"}},
        ]

        lineup = lineup_fields(lineups, home_id=10, away_id=20)
        injury = injury_fields(injuries, home_id=10, away_id=20)

        self.assertEqual(lineup["lineup_available"], 1.0)
        self.assertAlmostEqual(lineup["home_lineup_strength"], 1.0)
        self.assertAlmostEqual(lineup["away_lineup_strength"], 10 / 11)
        self.assertEqual(injury["home_injury_count"], 1.0)
        self.assertEqual(injury["away_suspension_count"], 1.0)
        self.assertEqual(injury["away_absence_count"], 1.0)
        self.assertEqual(injury["team_news_available"], 1.0)

    def test_empty_injury_payload_marks_known_zero_team_news(self):
        injury = injury_fields([], home_id=10, away_id=20)

        self.assertEqual(injury["team_news_available"], 1.0)
        self.assertEqual(injury["home_absence_count"], 0.0)
        self.assertEqual(injury["away_absence_count"], 0.0)

    def test_api_context_merge_preserves_weather_columns(self):
        existing = pd.DataFrame([{
            "date": "2026-05-17",
            "league": "england",
            "home_team": "Man United",
            "away_team": "Tottenham",
            "weather_available": 1,
            "temperature_c": 14.5,
        }])
        incoming = pd.DataFrame([{
            "date": "2026-05-17",
            "league": "england",
            "home_team": "Man United",
            "away_team": "Tottenham",
            "team_news_available": 1,
            "home_injury_count": 2,
            "away_suspension_count": 1,
            "api_football_fixture_id": 123,
        }])

        merged = merge_api_match_context(existing, incoming)

        self.assertEqual(float(merged.loc[0, "weather_available"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "temperature_c"]), 14.5)
        self.assertEqual(float(merged.loc[0, "team_news_available"]), 1.0)
        self.assertEqual(float(merged.loc[0, "home_injury_count"]), 2.0)
        self.assertEqual(int(merged.loc[0, "api_football_fixture_id"]), 123)

    def test_api_context_backfill_skips_existing_rows(self):
        matched = pd.DataFrame([
            {
                "date": "2024-08-16",
                "league": "england",
                "home_team": "Man United",
                "away_team": "Fulham",
                "api_match_status": "matched",
            },
            {
                "date": "2024-08-17",
                "league": "england",
                "home_team": "Arsenal",
                "away_team": "Wolves",
                "api_match_status": "matched",
            },
        ])
        existing = pd.DataFrame([{
            "date": "2024-08-16",
            "league": "england",
            "home_team": "Man United",
            "away_team": "Fulham",
            "api_football_fixture_id": 123,
        }])

        remaining, skipped = skip_existing_context_rows(matched, existing)

        self.assertEqual(skipped, 1)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining.loc[0, "home_team"], "Arsenal")

    def test_api_client_turns_429_into_rate_limit_state(self):
        class FakeResponse:
            status_code = 429
            text = "Too Many Requests"
            headers = {"Retry-After": "60"}

            def json(self):
                return {"errors": {"requests": "rate limit"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            original_get = api_football_context.requests.get
            try:
                api_football_context.requests.get = lambda *args, **kwargs: FakeResponse()
                client = api_football_context.ApiFootballClient(api_key="fake", cache_dir=Path(tmpdir))
                rows = client.response("/fixtures", {"league": 39, "season": 2024})
            finally:
                api_football_context.requests.get = original_get

        self.assertEqual(rows, [])
        self.assertEqual(len(client.rate_limit_errors), 1)
        self.assertEqual(client.rate_limit_errors[0]["retry_after"], "60")


class BaseTuningTests(unittest.TestCase):
    def test_validation_elo_rows_do_not_expand_on_duplicate_dates(self):
        train_fit = pd.DataFrame([
            {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
            {"date": pd.Timestamp("2020-01-08"), "home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 0},
        ])
        val = pd.DataFrame([
            {"date": pd.Timestamp("2020-01-08"), "home_team": "A", "away_team": "C", "home_goals": 2, "away_goals": 1},
            {"date": pd.Timestamp("2020-01-15"), "home_team": "B", "away_team": "D", "home_goals": 1, "away_goals": 3},
        ])

        val_part = _validation_rows_with_elo(train_fit, val, K=40, home_adv=60)

        self.assertEqual(len(val_part), len(val))
        self.assertEqual(val_part["home_team"].tolist(), ["A", "B"])
        self.assertTrue(np.isfinite(val_part[["elo_home", "elo_away"]].to_numpy()).all())


class BettingSimulationTests(unittest.TestCase):
    def test_betting_simulation_can_run_silently(self):
        probs = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[2.1, 3.2, 4.5]])
        y_true = np.array([0])

        buf = io.StringIO()
        with redirect_stdout(buf):
            result = simulate_value_betting(probs, odds, y_true, verbose=False)

        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 1)

    def test_betting_records_include_open_to_close_value_when_available(self):
        probs = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[2.1, 3.2, 4.5]])
        y_true = np.array([0])
        match_info = [{
            "date": pd.Timestamp("2025-01-01"),
            "league": "england",
            "home_team": "A",
            "away_team": "B",
            "open_odds_home": 2.1,
            "close_odds_home": 2.0,
        }]

        records = betting_records(probs, odds, y_true, match_info=match_info)

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(float(records.loc[0, "clv_decimal"]), 0.05)

    def test_betting_records_leave_clv_blank_without_opening_price(self):
        probs = np.array([[0.55, 0.25, 0.20]])
        odds = np.array([[2.1, 3.2, 4.5]])
        y_true = np.array([0])
        match_info = [{
            "date": pd.Timestamp("2025-01-01"),
            "league": "england",
            "home_team": "A",
            "away_team": "B",
            "close_odds_home": 2.0,
        }]

        records = betting_records(probs, odds, y_true, match_info=match_info)

        self.assertEqual(len(records), 1)
        self.assertTrue(np.isnan(float(records.loc[0, "clv_decimal"])))


class BlendTests(unittest.TestCase):
    def test_blend_probabilities_renormalizes_output(self):
        base = np.array([[0.6, 0.2, 0.2]])
        xgb = np.array([[0.3, 0.4, 0.3]])

        blended = blend_probabilities(
            {"base": 2.0, "xgb": 1.0},
            {"base": base, "xgb": xgb},
        )

        self.assertAlmostEqual(float(blended.sum()), 1.0)
        self.assertEqual(blended.shape, (1, 3))


class RollingFeatureTests(unittest.TestCase):
    def _extra_feature_map(self, features: np.ndarray, names: list[str]) -> dict[str, float]:
        offset = 6 + 12
        return {
            name: float(features[feature_indices([name])[0] - offset])
            for name in names
        }

    def test_odds_triplet_respects_requested_snapshot(self):
        row = pd.Series({
            "odds_home": 1.7,
            "odds_draw": 3.6,
            "odds_away": 5.2,
            "open_odds_home": 2.0,
            "open_odds_draw": 3.4,
            "open_odds_away": 4.1,
            "close_odds_home": 1.8,
            "close_odds_draw": 3.2,
            "close_odds_away": 4.5,
        })

        self.assertEqual(odds_triplet_from_row(row, "opening"), (2.0, 3.4, 4.1))
        self.assertEqual(odds_triplet_from_row(row, "closing"), (1.8, 3.2, 4.5))
        self.assertEqual(odds_triplet_from_row(row, "legacy"), (1.7, 3.6, 5.2))

    def test_market_movement_features_can_be_disabled(self):
        row = pd.Series({
            "home_team": "A",
            "away_team": "B",
            "open_odds_home": 2.0,
            "open_odds_draw": 3.0,
            "open_odds_away": 4.0,
            "close_odds_home": 1.8,
            "close_odds_draw": 3.2,
            "close_odds_away": 4.5,
        })

        features = compute_pre_match_extra_features(
            row,
            pd.DataFrame(),
            include_market_movement_features=False,
        )
        feature_map = self._extra_feature_map(features, [
            "market_move_home",
            "market_move_draw",
            "market_move_away",
        ])

        np.testing.assert_allclose(list(feature_map.values()), np.zeros(3))

    def test_pre_match_features_use_only_past_matches(self):
        past = pd.DataFrame([
            {
                "date": pd.Timestamp("2024-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
                "home_shots": 10,
                "away_shots": 5,
                "home_shots_target": 4,
                "away_shots_target": 2,
                "home_corners": 6,
                "away_corners": 3,
                "home_yellows": 1,
                "away_yellows": 2,
                "home_reds": 0,
                "away_reds": 1,
            },
        ])
        row = pd.Series({
            "home_team": "A",
            "away_team": "B",
            "open_odds_home": 2.0,
            "open_odds_draw": 3.0,
            "open_odds_away": 4.0,
            "close_odds_home": 1.8,
            "close_odds_draw": 3.2,
            "close_odds_away": 4.5,
            "ou25_over_prob": 0.57,
            "ah_line": -0.5,
        })

        features = compute_pre_match_extra_features(row, past)

        self.assertEqual(len(features), EXTRA_AUX_LEN)
        feature_map = self._extra_feature_map(features, [
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
            "cards_home_5",
            "cards_away_5",
            "cards_diff_5",
            "ou25_over_prob",
            "ah_line",
        ])
        self.assertEqual(feature_map["goals_for_home_5"], 2.0)
        self.assertEqual(feature_map["goals_for_away_5"], 1.0)
        self.assertEqual(feature_map["goals_for_diff_5"], 1.0)
        self.assertEqual(feature_map["goals_against_home_5"], 1.0)
        self.assertEqual(feature_map["goals_against_away_5"], 2.0)
        self.assertEqual(feature_map["goals_against_diff_5"], -1.0)
        self.assertEqual(feature_map["shots_for_home_5"], 10.0)
        self.assertEqual(feature_map["shots_for_away_5"], 5.0)
        self.assertEqual(feature_map["shots_for_diff_5"], 5.0)
        self.assertEqual(feature_map["shots_against_home_5"], 5.0)
        self.assertEqual(feature_map["shots_against_away_5"], 10.0)
        self.assertEqual(feature_map["shots_against_diff_5"], -5.0)
        self.assertEqual(feature_map["cards_home_5"], 1.0)
        self.assertEqual(feature_map["cards_away_5"], 4.0)
        self.assertEqual(feature_map["cards_diff_5"], -3.0)
        self.assertAlmostEqual(feature_map["ou25_over_prob"], 0.57)
        self.assertAlmostEqual(feature_map["ah_line"], -0.5)

    def test_pre_match_features_include_external_context(self):
        row = pd.Series({
            "home_team": "A",
            "away_team": "B",
            "lineup_available": 1,
            "home_lineup_strength": 0.92,
            "away_lineup_strength": 0.84,
            "team_news_available": 1,
            "home_absence_count": 2,
            "away_absence_count": 4,
            "home_injury_count": 1,
            "away_injury_count": 3,
            "home_suspension_count": 1,
            "away_suspension_count": 0,
            "home_key_absence_count": 1,
            "away_key_absence_count": 2,
            "home_manager_change_recent": 0,
            "away_manager_change_recent": 1,
            "weather_available": 1,
            "temperature_c": 8.0,
            "wind_kph": 30.0,
            "precipitation_mm": 4.0,
        })

        features = compute_pre_match_extra_features(row, pd.DataFrame())
        offset = 6 + 12
        feature_map = {name: features[idx - offset] for idx, name in zip(feature_indices([
            "lineup_available",
            "lineup_strength_diff",
            "absence_count_diff",
            "injury_count_diff",
            "suspension_count_diff",
            "key_absence_count_diff",
            "manager_change_recent_diff",
            "weather_available",
            "temperature_c",
            "wind_kph",
            "precipitation_mm",
        ]), [
            "lineup_available",
            "lineup_strength_diff",
            "absence_count_diff",
            "injury_count_diff",
            "suspension_count_diff",
            "key_absence_count_diff",
            "manager_change_recent_diff",
            "weather_available",
            "temperature_c",
            "wind_kph",
            "precipitation_mm",
        ])}

        self.assertEqual(len(features), EXTRA_AUX_LEN)
        self.assertEqual(feature_map["lineup_available"], 1.0)
        self.assertAlmostEqual(feature_map["lineup_strength_diff"], 0.08)
        self.assertEqual(feature_map["absence_count_diff"], -2.0)
        self.assertEqual(feature_map["injury_count_diff"], -2.0)
        self.assertEqual(feature_map["suspension_count_diff"], 1.0)
        self.assertEqual(feature_map["key_absence_count_diff"], -1.0)
        self.assertEqual(feature_map["manager_change_recent_diff"], -1.0)
        self.assertEqual(feature_map["weather_available"], 1.0)
        self.assertEqual(feature_map["temperature_c"], 8.0)
        self.assertEqual(feature_map["wind_kph"], 30.0)
        self.assertEqual(feature_map["precipitation_mm"], 4.0)

    def test_loader_extracts_over_under_25_odds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw" / "england"
            raw_dir.mkdir(parents=True)
            pd.DataFrame([{
                "Date": "01/01/2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 2,
                "FTAG": 1,
                "B365H": 2.0,
                "B365D": 3.5,
                "B365A": 4.0,
                "AvgC>2.5": 1.8,
                "AvgC<2.5": 2.1,
            }]).to_csv(raw_dir / "E0_2024.csv", index=False)

            original_root = data_processing.PROJECT_ROOT
            original_add_understat_xg = data_processing.add_understat_xg
            try:
                data_processing.PROJECT_ROOT = root
                data_processing.add_understat_xg = lambda df, league_name: df
                loaded = data_processing.load_league_data("england")
            finally:
                data_processing.PROJECT_ROOT = original_root
                data_processing.add_understat_xg = original_add_understat_xg

        self.assertAlmostEqual(float(loaded.loc[0, "ou25_over_odds"]), 1.8)
        self.assertAlmostEqual(float(loaded.loc[0, "ou25_under_odds"]), 2.1)
        self.assertTrue(np.isfinite(float(loaded.loc[0, "ou25_over_prob"])))

    def test_understat_match_rows_join_to_loaded_matches(self):
        base = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"),
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        }])
        understat = pd.DataFrame([{
            "date": "2024-01-01",
            "league": "EPL",
            "team_h": "Arsenal",
            "team_a": "Chelsea",
            "h_xg": 1.7,
            "a_xg": 0.8,
            "h_npxg": 1.4,
            "a_npxg": 0.7,
            "h_xpts": 2.1,
            "a_xpts": 0.6,
        }])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "understat_matches.csv"
            understat.to_csv(path, index=False)

            merged = add_understat_xg(base, "england", path=path)

        self.assertAlmostEqual(float(merged.loc[0, "home_understat_xg"]), 1.7)
        self.assertAlmostEqual(float(merged.loc[0, "away_understat_npxg"]), 0.7)

    def test_external_match_context_join_adds_lineup_weather_team_news(self):
        base = pd.DataFrame([{
            "date": pd.Timestamp("2024-08-16"),
            "home_team": "Man United",
            "away_team": "Fulham",
        }])
        context = pd.DataFrame([{
            "date": "2024-08-16",
            "league": "england",
            "home_team": "Man Utd",
            "away_team": "Fulham",
            "lineup_available": 1,
            "home_lineup_strength": 0.94,
            "away_lineup_strength": 0.88,
            "home_injury_count": 2,
            "away_suspension_count": 1,
            "home_manager_change_days": 10,
            "weather_available": 1,
            "temperature_c": 18.5,
            "wind_kph": 12.0,
            "precipitation_mm": 0.4,
        }])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "match_context.csv"
            context.to_csv(path, index=False)

            merged = add_external_match_context(base, "england", path=path)

        self.assertEqual(float(merged.loc[0, "lineup_available"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "home_lineup_strength"]), 0.94)
        self.assertAlmostEqual(float(merged.loc[0, "away_lineup_strength"]), 0.88)
        self.assertEqual(float(merged.loc[0, "home_absence_count"]), 2.0)
        self.assertEqual(float(merged.loc[0, "away_suspension_count"]), 1.0)
        self.assertEqual(float(merged.loc[0, "home_manager_change_recent"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "temperature_c"]), 18.5)


if __name__ == "__main__":
    unittest.main()
