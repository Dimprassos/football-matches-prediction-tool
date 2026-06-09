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
import src.update_understat as understat_scraper
import src.data_processing as data_processing
from scripts.update_team_news import _backtest_window
from src.cli.backtest_season_cli import build_backtest_config, season_window
from src.calibration import temperature_scale_probs
from src.config import ExperimentConfig
from src.evaluation import betting_records, simulate_value_betting
from src.external_context import add_external_match_context
from src.feature_builder import FEATURE_COLUMNS, ensure_market_probs, feature_indices, market_probs_from_odds_row
from src.metrics import class_metric_summary
from src.models.base import _validation_rows_with_elo
from src.models.meta import blend_probabilities, market_logit_correction_probs, tune_market_logit_correction
from src.player_context import (
    MATCH_ABSENCES_COLUMNS,
    MATCH_LINEUPS_COLUMNS,
    PLAYER_MATCH_STATS_COLUMNS,
    PLAYER_REGISTRY_COLUMNS,
    build_runtime_player_context,
    build_player_match_context,
    build_player_strength_index,
    compute_player_strength,
    merge_player_match_context,
    load_match_absences,
    load_match_lineups,
    load_player_match_stats,
    load_player_context_tables,
    load_player_registry,
)
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
            "home_absence_strength_loss": 0.85,
            "away_absence_strength_loss": 0.25,
            "home_player_context_available": 1,
            "away_player_context_available": 1,
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
            "home_absence_strength_loss": 0.85,
            "away_absence_strength_loss": 0.25,
            "home_player_context_available": 1,
            "away_player_context_available": 1,
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
        self.assertAlmostEqual(float(merged.loc[0, "home_absence_strength_loss"]), 0.85)
        self.assertAlmostEqual(float(merged.loc[0, "away_absence_strength_loss"]), 0.25)
        self.assertAlmostEqual(float(merged.loc[0, "absence_strength_loss_diff"]), 0.60)
        self.assertEqual(float(merged.loc[0, "home_player_context_available"]), 1.0)


class PlayerContextSchemaTests(unittest.TestCase):
    def test_missing_player_context_files_return_empty_canonical_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = load_player_context_tables(Path(tmpdir))

        self.assertEqual(tables.registry.columns.tolist(), PLAYER_REGISTRY_COLUMNS)
        self.assertEqual(tables.match_stats.columns.tolist(), PLAYER_MATCH_STATS_COLUMNS)
        self.assertEqual(tables.lineups.columns.tolist(), MATCH_LINEUPS_COLUMNS)
        self.assertEqual(tables.absences.columns.tolist(), MATCH_ABSENCES_COLUMNS)
        self.assertTrue(tables.registry.empty)
        self.assertTrue(tables.match_stats.empty)
        self.assertTrue(tables.lineups.empty)
        self.assertTrue(tables.absences.empty)

    def test_player_context_loaders_normalize_valid_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pd.DataFrame([{
                "player_id": "barcelona_lamine_yamal_2025",
                "player_name": "Lamine Yamal",
                "team": "FC Barcelona",
                "league": "SP1",
                "season": 2025,
                "position": "RW",
                "importance_tier": "key",
            }]).to_csv(root / "player_registry.csv", index=False)
            pd.DataFrame([{
                "date": "2025-09-14",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Valencia CF",
                "player_id": "barcelona_lamine_yamal_2025",
                "started": 1,
                "minutes": 87,
                "goals": 1,
            }]).to_csv(root / "player_match_stats.csv", index=False)
            pd.DataFrame([{
                "date": "2026-01-18",
                "kickoff_at": "2026-01-18T20:00:00+02:00",
                "league": "La Liga",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "team": "FC Barcelona",
                "player_id": "barcelona_lamine_yamal_2025",
                "is_starter": 1,
                "is_sub": 0,
                "source": "manual",
                "available_at": "2026-01-18T18:45:00+02:00",
            }]).to_csv(root / "match_lineups.csv", index=False)
            pd.DataFrame([{
                "date": "2026-01-18",
                "kickoff_at": "2026-01-18T20:00:00+02:00",
                "league": "spain",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "team": "FC Barcelona",
                "player_id": "barcelona_lamine_yamal_2025",
                "absence_type": "injury",
                "status": "doubtful",
                "source": "manual",
                "available_at": "2026-01-17T12:00:00+02:00",
            }]).to_csv(root / "match_absences.csv", index=False)

            tables = load_player_context_tables(root)

        self.assertEqual(tables.registry.loc[0, "league"], "spain")
        self.assertEqual(tables.registry.loc[0, "team"], "Barcelona")
        self.assertEqual(tables.registry.loc[0, "normalized_player_name"], "lamine yamal")
        self.assertEqual(tables.registry.loc[0, "importance_tier"], "key")
        self.assertEqual(tables.match_stats.loc[0, "team"], "Barcelona")
        self.assertEqual(tables.match_stats.loc[0, "opponent"], "Valencia")
        self.assertEqual(tables.lineups.loc[0, "home_team"], "Barcelona")
        self.assertEqual(tables.lineups.loc[0, "league"], "spain")
        self.assertEqual(tables.absences.loc[0, "status"], "doubtful")

    def test_lineup_available_after_kickoff_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "match_lineups.csv"
            pd.DataFrame([{
                "date": "2026-01-18",
                "kickoff_at": "2026-01-18T20:00:00+02:00",
                "league": "spain",
                "home_team": "Barcelona",
                "away_team": "Real Madrid",
                "team": "Barcelona",
                "player_id": "p1",
                "is_starter": 1,
                "is_sub": 0,
                "available_at": "2026-01-18T20:05:00+02:00",
            }]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "available_at is after kickoff_at"):
                load_match_lineups(path)

    def test_invalid_importance_tier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "player_registry.csv"
            pd.DataFrame([{
                "player_id": "p1",
                "player_name": "Player One",
                "team": "Barcelona",
                "league": "spain",
                "season": 2025,
                "importance_tier": "superstar",
            }]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "importance_tier contains invalid values"):
                load_player_registry(path)


class PlayerContextStrengthTests(unittest.TestCase):
    def test_rolling_player_strength_ignores_future_rows(self):
        rows = [
            {
                "date": "2025-01-01",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Valencia CF",
                "player_id": "p1",
                "started": 1,
                "minutes": 90,
                "goals": 0,
                "assists": 0,
                "team_goal_diff": 1,
            },
            {
                "date": "2025-01-08",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Sevilla FC",
                "player_id": "teammate",
                "started": 1,
                "minutes": 90,
                "team_goal_diff": -1,
            },
            {
                "date": "2025-01-15",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Getafe CF",
                "player_id": "p1",
                "started": 1,
                "minutes": 90,
                "goals": 0,
                "assists": 0,
                "team_goal_diff": 2,
            },
        ]
        future_row = {
            "date": "2025-02-01",
            "league": "spain",
            "team": "FC Barcelona",
            "opponent": "Real Betis",
            "player_id": "p1",
            "started": 1,
            "minutes": 90,
            "goals": 8,
            "assists": 4,
            "team_goal_diff": 9,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            before_path = Path(tmpdir) / "before.csv"
            after_path = Path(tmpdir) / "after.csv"
            pd.DataFrame(rows).to_csv(before_path, index=False)
            pd.DataFrame([*rows, future_row]).to_csv(after_path, index=False)
            before_stats = load_player_match_stats(before_path)
            after_stats = load_player_match_stats(after_path)

        before = compute_player_strength("p1", "Barcelona", "spain", "2025-01-20", before_stats)
        after = compute_player_strength("p1", "Barcelona", "spain", "2025-01-20", after_stats)
        future_visible = compute_player_strength("p1", "Barcelona", "spain", "2025-02-10", after_stats)

        self.assertEqual(before.source, "rolling_stats")
        self.assertAlmostEqual(before.player_strength, after.player_strength)
        self.assertAlmostEqual(before.minutes_share_last_10, 2 / 3)
        self.assertAlmostEqual(before.starter_rate_last_10, 2 / 3)
        self.assertGreater(before.on_off_component, 0.5)
        self.assertGreater(future_visible.player_strength, after.player_strength)

    def test_player_strength_uses_importance_tier_when_stats_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "player_registry.csv"
            pd.DataFrame([{
                "player_id": "p1",
                "player_name": "Player One",
                "team": "FC Barcelona",
                "league": "SP1",
                "season": 2025,
                "importance_tier": "key",
            }]).to_csv(path, index=False)
            registry = load_player_registry(path)

        strength = compute_player_strength(
            "p1",
            "Barcelona",
            "spain",
            "2025-01-20",
            pd.DataFrame(columns=PLAYER_MATCH_STATS_COLUMNS),
            registry,
        )

        self.assertEqual(strength.source, "importance_tier")
        self.assertTrue(strength.context_available)
        self.assertAlmostEqual(strength.player_strength, 0.85)

    def test_player_strength_returns_neutral_without_stats_or_registry(self):
        strength = compute_player_strength(
            "missing",
            "Barcelona",
            "spain",
            "2025-01-20",
            pd.DataFrame(columns=PLAYER_MATCH_STATS_COLUMNS),
            registry=None,
        )

        self.assertEqual(strength.source, "neutral")
        self.assertFalse(strength.context_available)
        self.assertEqual(strength.player_strength, 0.0)


class PlayerMatchContextGenerationTests(unittest.TestCase):
    def _write_player_context_inputs(self, root: Path):
        pd.DataFrame([
            {
                "player_id": "p1",
                "player_name": "Starter One",
                "team": "FC Barcelona",
                "league": "SP1",
                "season": 2025,
                "importance_tier": "key",
            },
            {
                "player_id": "p2",
                "player_name": "Starter Two",
                "team": "FC Barcelona",
                "league": "SP1",
                "season": 2025,
                "importance_tier": "rotation",
            },
            {
                "player_id": "p_abs_home",
                "player_name": "Absent Key",
                "team": "FC Barcelona",
                "league": "SP1",
                "season": 2025,
                "importance_tier": "key",
            },
            {
                "player_id": "p_abs_away",
                "player_name": "Away Rotation",
                "team": "Real Madrid",
                "league": "spain",
                "season": 2025,
                "importance_tier": "rotation",
            },
        ]).to_csv(root / "player_registry.csv", index=False)
        pd.DataFrame([
            {
                "date": "2025-01-01",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Valencia CF",
                "player_id": "p1",
                "started": 1,
                "minutes": 90,
                "team_goal_diff": 2,
                "goals": 1,
                "assists": 0,
            },
            {
                "date": "2025-01-08",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Sevilla FC",
                "player_id": "p2",
                "started": 1,
                "minutes": 70,
                "team_goal_diff": 1,
                "goals": 0,
                "assists": 1,
            },
            {
                "date": "2025-01-15",
                "league": "spain",
                "team": "FC Barcelona",
                "opponent": "Getafe CF",
                "player_id": "other",
                "started": 1,
                "minutes": 90,
                "team_goal_diff": -1,
            },
        ]).to_csv(root / "player_match_stats.csv", index=False)
        pd.DataFrame([
            {
                "date": "2025-02-01",
                "kickoff_at": "2025-02-01T20:00:00+02:00",
                "league": "spain",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "team": "FC Barcelona",
                "player_id": "p1",
                "is_starter": 1,
                "is_sub": 0,
                "available_at": "2025-02-01T18:45:00+02:00",
            },
            {
                "date": "2025-02-01",
                "kickoff_at": "2025-02-01T20:00:00+02:00",
                "league": "spain",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "team": "FC Barcelona",
                "player_id": "p2",
                "is_starter": 1,
                "is_sub": 0,
                "available_at": "2025-02-01T18:45:00+02:00",
            },
        ]).to_csv(root / "match_lineups.csv", index=False)
        pd.DataFrame([
            {
                "date": "2025-02-01",
                "kickoff_at": "2025-02-01T20:00:00+02:00",
                "league": "spain",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "team": "FC Barcelona",
                "player_id": "p_abs_home",
                "absence_type": "injury",
                "status": "out",
                "available_at": "2025-01-31T12:00:00+02:00",
            },
            {
                "date": "2025-02-01",
                "kickoff_at": "2025-02-01T20:00:00+02:00",
                "league": "spain",
                "home_team": "FC Barcelona",
                "away_team": "Real Madrid",
                "team": "Real Madrid",
                "player_id": "p_abs_away",
                "absence_type": "suspension",
                "status": "doubtful",
                "available_at": "2025-01-31T12:00:00+02:00",
            },
        ]).to_csv(root / "match_absences.csv", index=False)

    def test_build_player_match_context_weights_lineups_and_absences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_player_context_inputs(root)
            tables = load_player_context_tables(root)

            context = build_player_match_context(
                tables.registry,
                tables.match_stats,
                tables.lineups,
                tables.absences,
            )

        self.assertEqual(len(context), 1)
        row = context.iloc[0]
        self.assertEqual(row["home_team"], "Barcelona")
        self.assertEqual(float(row["lineup_available"]), 1.0)
        self.assertGreater(float(row["home_lineup_strength"]), 0.0)
        self.assertEqual(float(row["team_news_available"]), 1.0)
        self.assertEqual(float(row["home_injury_count"]), 1.0)
        self.assertEqual(float(row["away_suspension_count"]), 0.5)
        self.assertEqual(float(row["home_key_absence_count"]), 1.0)
        self.assertAlmostEqual(float(row["home_absence_strength_loss"]), 0.85)
        self.assertAlmostEqual(float(row["away_absence_strength_loss"]), 0.275)
        self.assertAlmostEqual(float(row["absence_strength_loss_diff"]), 0.575)
        self.assertEqual(float(row["home_player_context_available"]), 1.0)
        self.assertEqual(float(row["away_player_context_available"]), 1.0)

    def test_merge_player_match_context_preserves_existing_weather_columns(self):
        existing = pd.DataFrame([{
            "date": "2025-02-01",
            "league": "SP1",
            "home_team": "FC Barcelona",
            "away_team": "Real Madrid",
            "weather_available": 1,
            "temperature_c": 12.5,
            "home_absence_count": 99,
        }])
        player_context = pd.DataFrame([{
            "date": pd.Timestamp("2025-02-01"),
            "league": "spain",
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "home_absence_count": 1.0,
            "home_absence_strength_loss": 0.85,
            "home_player_context_available": 1.0,
        }])

        merged = merge_player_match_context(existing, player_context)

        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.loc[0, "weather_available"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "temperature_c"]), 12.5)
        self.assertEqual(float(merged.loc[0, "home_absence_count"]), 1.0)
        self.assertAlmostEqual(float(merged.loc[0, "home_absence_strength_loss"]), 0.85)

    def test_runtime_player_context_builds_manual_context_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_player_context_inputs(root)
            tables = load_player_context_tables(root)

            context, diagnostics = build_runtime_player_context(
                league="spain",
                home_team="FC Barcelona",
                away_team="Real Madrid",
                match_date="2025-02-01",
                registry=tables.registry,
                match_stats=tables.match_stats,
                home_starters=["p1", "p2"],
                away_absences=[{
                    "player_id": "p_abs_away",
                    "absence_type": "suspension",
                    "status": "doubtful",
                }],
            )

        self.assertEqual(context["lineup_available"], 1.0)
        self.assertGreater(context["home_lineup_strength"], 0.0)
        self.assertEqual(context["team_news_available"], 1.0)
        self.assertEqual(context["away_suspension_count"], 0.5)
        self.assertAlmostEqual(context["away_absence_strength_loss"], 0.275)
        self.assertEqual(context["home_player_context_available"], 1.0)
        self.assertEqual(context["away_player_context_available"], 1.0)
        self.assertEqual(len(diagnostics), 3)
        self.assertIn("rolling_stats", set(diagnostics["source"]))
        self.assertIn("importance_tier", set(diagnostics["source"]))

    def test_runtime_player_context_keeps_neutral_fallback_without_player_data(self):
        context, diagnostics = build_runtime_player_context(
            league="spain",
            home_team="Barcelona",
            away_team="Real Madrid",
            match_date="2025-02-01",
            home_starters=["unknown_player"],
        )

        self.assertEqual(context["lineup_available"], 1.0)
        self.assertEqual(context["home_player_context_available"], 0.0)
        self.assertEqual(context["away_player_context_available"], 0.0)
        self.assertNotIn("home_lineup_strength", context)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics.loc[0, "source"], "neutral")

    def test_runtime_player_context_rejects_invalid_absence_status(self):
        with self.assertRaisesRegex(ValueError, "Invalid status"):
            build_runtime_player_context(
                league="spain",
                home_team="Barcelona",
                away_team="Real Madrid",
                match_date="2025-02-01",
                home_absences=[{"player_id": "p1", "status": "maybe"}],
            )


class PlayerContextFeatureAlignmentTests(unittest.TestCase):
    def test_player_context_features_are_appended_and_read_by_state_builder(self):
        self.assertEqual(FEATURE_COLUMNS[-5:], [
            "home_absence_strength_loss",
            "away_absence_strength_loss",
            "absence_strength_loss_diff",
            "home_player_context_available",
            "away_player_context_available",
        ])
        row = pd.Series({
            "home_team": "A",
            "away_team": "B",
            "home_absence_strength_loss": 0.85,
            "away_absence_strength_loss": 0.25,
            "home_player_context_available": 1,
            "away_player_context_available": 1,
        })

        features = compute_pre_match_extra_features(row, pd.DataFrame())
        offset = 6 + 12
        home_loss = features[feature_indices(["home_absence_strength_loss"])[0] - offset]
        diff = features[feature_indices(["absence_strength_loss_diff"])[0] - offset]
        home_available = features[feature_indices(["home_player_context_available"])[0] - offset]

        self.assertEqual(len(features), EXTRA_AUX_LEN)
        self.assertAlmostEqual(float(home_loss), 0.85)
        self.assertAlmostEqual(float(diff), 0.60)
        self.assertEqual(float(home_available), 1.0)


class RuntimePredictorFeatureTests(unittest.TestCase):
    def test_runtime_extra_features_reconstruct_understat_from_history(self):
        from types import SimpleNamespace

        from src.predictor import build_runtime_extra_features

        past = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"),
            "home_team": "A",
            "away_team": "B",
            "home_goals": 2,
            "away_goals": 1,
            "home_understat_xg": 1.7,
            "away_understat_xg": 0.8,
            "home_understat_npxg": 1.4,
            "away_understat_npxg": 0.7,
            "home_understat_xpts": 2.1,
            "away_understat_xpts": 0.6,
        }])
        state = SimpleNamespace(played_df=past)

        extra = build_runtime_extra_features("A", "B", state)

        self.assertEqual(len(extra), EXTRA_AUX_LEN)
        offset = 6 + 12
        xg_home = extra[feature_indices(["understat_xg_for_home_5"])[0] - offset]
        xg_away = extra[feature_indices(["understat_xg_for_away_5"])[0] - offset]
        self.assertAlmostEqual(float(xg_home), 1.7)
        self.assertAlmostEqual(float(xg_away), 0.8)

    def test_runtime_extra_features_fall_back_to_neutral_without_history(self):
        from types import SimpleNamespace

        from src.predictor import build_runtime_extra_features
        from src.state_builder import neutral_extra_features

        for played in (None, pd.DataFrame()):
            extra = build_runtime_extra_features("A", "B", SimpleNamespace(played_df=played))
            np.testing.assert_allclose(extra, neutral_extra_features())

    def test_runtime_extra_features_apply_manual_player_context(self):
        from types import SimpleNamespace

        from src.predictor import build_runtime_extra_features

        past = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01"),
            "home_team": "A",
            "away_team": "B",
            "home_goals": 1,
            "away_goals": 0,
        }])
        context = {
            "home_absence_strength_loss": 0.85,
            "away_absence_strength_loss": 0.25,
            "home_player_context_available": 1.0,
            "away_player_context_available": 1.0,
        }

        extra = build_runtime_extra_features("A", "B", SimpleNamespace(played_df=past), context=context)
        offset = 6 + 12
        home_loss = extra[feature_indices(["home_absence_strength_loss"])[0] - offset]
        diff = extra[feature_indices(["absence_strength_loss_diff"])[0] - offset]
        home_available = extra[feature_indices(["home_player_context_available"])[0] - offset]

        self.assertAlmostEqual(float(home_loss), 0.85)
        self.assertAlmostEqual(float(diff), 0.60)
        self.assertEqual(float(home_available), 1.0)


class UnderstatPlayerScrapeTests(unittest.TestCase):
    """The Understat per-player scraper must emit files the loaders accept."""

    def _sample_match(self):
        meta = {
            "id": "555",
            "date": "2023-08-12",
            "home_title": "Arsenal",
            "away_title": "Chelsea",
            "home_goals": 2,
            "away_goals": 1,
        }
        rosters = {
            "h": {
                "r1": {"player_id": "100", "player": "Alice Keeper", "position": "GK",
                       "time": "90", "goals": "0", "assists": "0", "shots": "0",
                       "key_passes": "1", "xG": "0", "xA": "0.10", "h_a": "h"},
                "r2": {"player_id": "101", "player": "Bob Striker", "position": "FW",
                       "time": "70", "goals": "1", "assists": "0", "shots": "3",
                       "key_passes": "0", "xG": "0.40", "xA": "0", "h_a": "h"},
                "r3": {"player_id": "102", "player": "Carl Sub", "position": "Sub",
                       "time": "20", "goals": "0", "assists": "1", "shots": "1",
                       "key_passes": "2", "xG": "0.05", "xA": "0.30", "h_a": "h"},
            },
            "a": {
                "r4": {"player_id": "200", "player": "Dan Defender", "position": "DC",
                       "time": "90", "goals": "0", "assists": "0", "shots": "0",
                       "key_passes": "0", "xG": "0", "xA": "0", "h_a": "a"},
            },
        }
        return meta, rosters

    def test_observations_map_understat_roster_to_schema(self):
        meta, rosters = self._sample_match()
        obs = understat_scraper._match_player_observations("EPL", 2023, meta, rosters)
        self.assertEqual(len(obs), 4)
        by_id = {row["player_id"]: row for row in obs}
        # started is derived from position: "Sub" means an off-the-bench appearance
        self.assertEqual(by_id["100"]["started"], 1)
        self.assertEqual(by_id["102"]["started"], 0)
        self.assertEqual(by_id["102"]["minutes"], 20.0)
        # home/away orientation, opponent, and scoreline-derived goal difference
        self.assertEqual(by_id["101"]["home_away"], "home")
        self.assertEqual(by_id["101"]["team"], "Arsenal")
        self.assertEqual(by_id["101"]["opponent"], "Chelsea")
        self.assertEqual(by_id["101"]["team_goals"], 2)
        self.assertEqual(by_id["101"]["opponent_goals"], 1)
        self.assertEqual(by_id["101"]["team_goal_diff"], 1)
        self.assertEqual(by_id["200"]["home_away"], "away")
        self.assertEqual(by_id["200"]["team_goal_diff"], -1)

    def test_observations_translate_titles_through_team_name_bridge(self):
        from src.understat_data import normalize_team_name as understat_join_key
        meta = {
            "id": "1", "date": "2023-08-12",
            "home_title": "Manchester City", "away_title": "Wolverhampton Wanderers",
            "home_goals": 3, "away_goals": 0,
        }
        rosters = {
            "h": {"a": {"player_id": "1", "player": "P", "position": "FW", "time": "90", "h_a": "h"}},
            "a": {"b": {"player_id": "2", "player": "Q", "position": "DC", "time": "90", "h_a": "a"}},
        }
        bridge = {
            understat_join_key("Manchester City"): "Man City",
            understat_join_key("Wolverhampton Wanderers"): "Wolves",
        }
        obs = understat_scraper._match_player_observations(
            "EPL", 2023, meta, rosters, team_name_bridge=bridge
        )
        teams = {row["player_id"]: row["team"] for row in obs}
        # Understat long names are translated to the main dataset's short names
        self.assertEqual(teams["1"], "Man City")
        self.assertEqual(teams["2"], "Wolves")

    def test_generated_player_csvs_pass_loaders_and_feed_strength(self):
        meta, rosters = self._sample_match()
        observations = pd.DataFrame(
            understat_scraper._match_player_observations("EPL", 2023, meta, rosters)
        )
        stats = understat_scraper.player_match_stats_from_observations(observations)
        registry = understat_scraper.player_registry_from_observations(observations)
        # producer output uses the canonical schemas exactly (no drift)
        self.assertEqual(list(stats.columns), PLAYER_MATCH_STATS_COLUMNS)
        self.assertEqual(list(registry.columns), PLAYER_REGISTRY_COLUMNS)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "player_match_stats.csv"
            registry_path = Path(tmpdir) / "player_registry.csv"
            understat_scraper.write_player_match_stats(stats, stats_path)
            understat_scraper.write_player_registry(registry, registry_path)
            # the strict loaders accept the generated files without raising
            loaded_stats = load_player_match_stats(stats_path)
            loaded_registry = load_player_registry(registry_path)

        self.assertEqual(len(loaded_stats), 4)
        # the loader normalized the Understat league code to the canonical name
        self.assertEqual(set(loaded_stats["league"]), {"england"})
        # the generated stats are real history the rolling-strength path can use
        strength = compute_player_strength(
            "101", "Arsenal", "england", "2023-08-20", loaded_stats
        )
        self.assertEqual(strength.source, "rolling_stats")
        self.assertGreater(strength.player_strength, 0.0)
        self.assertEqual(strength.starter_rate_last_10, 1.0)

        self.assertEqual(len(loaded_registry), 4)
        # importance_tier is left as a manual field, not invented from participation
        self.assertEqual(set(loaded_registry["importance_tier"]), {"unknown"})


class PlayerStrengthIndexTests(unittest.TestCase):
    """The fast index must return identical results to compute_player_strength."""

    def _stats(self):
        rows = []
        # two teams, several matches each, one tracked player + teammates
        for i, opp in enumerate(["Chelsea", "Arsenal", "Everton", "Spurs", "Leeds"]):
            rows.append({
                "date": f"2023-09-0{i+1}", "league": "EPL", "team": "Man City",
                "opponent": opp, "player_id": "star", "started": 1, "minutes": 90,
                "goals": 1, "assists": 0, "shots": 3, "key_passes": 1, "xg": 0.5, "xa": 0.1,
                "team_goal_diff": 2,
            })
            rows.append({
                "date": f"2023-09-0{i+1}", "league": "EPL", "team": "Man City",
                "opponent": opp, "player_id": "sub", "started": 0, "minutes": 10,
                "goals": 0, "assists": 0, "shots": 0, "key_passes": 0, "xg": 0.0, "xa": 0.0,
                "team_goal_diff": 2,
            })
            rows.append({
                "date": f"2023-09-0{i+1}", "league": "EPL", "team": "Liverpool",
                "opponent": opp, "player_id": "lfc1", "started": 1, "minutes": 90,
                "goals": 0, "assists": 1, "shots": 1, "key_passes": 2, "xg": 0.1, "xa": 0.3,
                "team_goal_diff": 0,
            })
        return pd.DataFrame(rows)

    def test_index_matches_compute_player_strength(self):
        stats = self._stats()
        index = build_player_strength_index(stats)
        cases = [
            ("star", "Man City", "england"),
            ("sub", "Man City", "england"),
            ("lfc1", "Liverpool", "england"),
            ("ghost", "Man City", "england"),       # unknown player -> fallback
            ("star", "Real Madrid", "spain"),        # unknown team -> fallback
        ]
        for pid, team, league in cases:
            direct = compute_player_strength(pid, team, league, "2023-10-01", stats)
            via_index = index.strength(pid, team, league, "2023-10-01")
            self.assertEqual(direct.source, via_index.source, pid)
            self.assertAlmostEqual(direct.player_strength, via_index.player_strength, places=9, msg=pid)
            self.assertAlmostEqual(direct.starter_rate_last_10, via_index.starter_rate_last_10, places=9, msg=pid)
            self.assertAlmostEqual(direct.minutes_share_last_10, via_index.minutes_share_last_10, places=9, msg=pid)
            self.assertEqual(direct.history_matches, via_index.history_matches, pid)

    def test_fixture_strengths_matches_per_player(self):
        stats = self._stats()
        index = build_player_strength_index(stats)
        ids = ["star", "sub", "ghost"]
        batched = index.fixture_strengths("Man City", "england", "2023-10-01", ids)
        self.assertEqual(len(batched), len(ids))
        for pid, b in zip(ids, batched):
            single = index.strength(pid, "Man City", "england", "2023-10-01")
            self.assertEqual(single.source, b.source, pid)
            self.assertAlmostEqual(single.player_strength, b.player_strength, places=9, msg=pid)
            self.assertAlmostEqual(single.minutes_share_last_10, b.minutes_share_last_10, places=9, msg=pid)

    def test_likely_xi_ranks_recent_minutes(self):
        index = build_player_strength_index(self._stats())
        xi = index.likely_xi("Man City", "england", "2023-10-01", n=2)
        self.assertEqual(xi[0], "star")        # 90' every match -> top
        self.assertIn("sub", xi)
        self.assertEqual(index.likely_xi("Real Madrid", "spain", "2023-10-01"), [])  # unknown team

    def test_build_lineup_strength_context_is_leakage_safe(self):
        from src.player_context import build_lineup_strength_context
        rows = []
        for d, opp in [("2023-09-01", "Chelsea"), ("2023-09-08", "Arsenal")]:
            for pid in ["a", "b", "c"]:
                rows.append({"date": d, "league": "EPL", "team": "Man City", "opponent": opp,
                             "home_away": "home", "player_id": pid, "started": 1, "minutes": 90,
                             "goals": 0, "assists": 0, "shots": 0, "key_passes": 0, "xg": 0.0,
                             "xa": 0.0, "team_goal_diff": 1})
            for pid in ["x", "y"]:
                rows.append({"date": d, "league": "EPL", "team": opp, "opponent": "Man City",
                             "home_away": "away", "player_id": pid, "started": 1, "minutes": 90,
                             "goals": 0, "assists": 0, "shots": 0, "key_passes": 0, "xg": 0.0,
                             "xa": 0.0, "team_goal_diff": -1})
        ctx = build_lineup_strength_context(pd.DataFrame(rows))
        ctx["date"] = pd.to_datetime(ctx["date"])
        self.assertEqual(len(ctx), 2)
        first = ctx[ctx["date"] == pd.Timestamp("2023-09-01")].iloc[0]
        second = ctx[ctx["date"] == pd.Timestamp("2023-09-08")].iloc[0]
        # fixture 1: no prior history -> no usable lineup strength
        self.assertTrue(pd.isna(first["home_lineup_strength"]))
        # fixture 2: Man City has fixture-1 history -> finite; Arsenal (first appearance) -> NaN
        self.assertTrue(np.isfinite(second["home_lineup_strength"]))
        self.assertTrue(pd.isna(second["away_lineup_strength"]))
        self.assertEqual(second["home_team"], "Man City")

    def test_index_respects_as_of_date_cutoff(self):
        stats = self._stats()
        index = build_player_strength_index(stats)
        # before any match -> no history -> neutral/fallback
        early = index.strength("star", "Man City", "england", "2023-08-01")
        self.assertEqual(early.source, "neutral")
        # after matches -> rolling stats
        late = index.strength("star", "Man City", "england", "2023-10-01")
        self.assertEqual(late.source, "rolling_stats")
        self.assertGreater(late.player_strength, 0.0)


if __name__ == "__main__":
    unittest.main()
