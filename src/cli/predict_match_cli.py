"""Interactive command-line predictor for a single custom match.

Prompts for a league, the two teams, and odds, then prints every model's 1/X/2
probabilities. The terminal counterpart of the Streamlit predict page; both call
:func:`src.predictor.predict_custom_match`.
"""
from __future__ import annotations

import warnings

import numpy as np

from src.predictor import get_league_runtime_state, load_runtime_artifacts, predict_custom_match


def pick_team(teams, prompt):
    """Prompt for a team name and resolve it via case-insensitive substring match."""
    raw = input(prompt).strip()
    if not raw:
        return None
    matches = [t for t in teams if raw.lower() in t.lower()]
    if not matches:
        print("Team not found.")
        return None
    if len(matches) == 1:
        print(f"Selected: {matches[0]}")
        return matches[0]
    print("Possible matches:")
    for i, t in enumerate(matches[:10], 1):
        print(f"{i}. {t}")
    try:
        idx = int(input("Choose number: ")) - 1
        return matches[idx]
    except Exception:
        return matches[0]


def main():
    print("=== Interactive Match Predictor ===")
    params, meta_model, meta_cfg, mlp_model, mlp_meta, logreg_model, logreg_meta, blend_cfg = load_runtime_artifacts()
    leagues = ["england", "spain", "italy", "germany", "france"]

    while True:
        print("\nAvailable Leagues:")
        for i, l in enumerate(leagues, 1):
            print(f"{i}. {l}")
        try:
            choice = int(input("\nSelect League (number) or 0 to exit: "))
        except ValueError:
            continue
        if choice == 0:
            break
        if choice < 1 or choice > len(leagues):
            continue
        league = leagues[choice - 1]
        print(f"Loading data for {league}...")
        state = get_league_runtime_state(league, params)
        teams = sorted(state.ratings.keys())

        while True:
            print(f"\n--- {league.upper()} Prediction ---")
            home_real = pick_team(teams, "Home Team Name (part or full, enter to go back): ")
            if home_real is None:
                break
            away_real = pick_team(teams, "Away Team Name (part or full): ")
            if away_real is None:
                continue
            if home_real == away_real:
                print("Home and Away teams cannot be the same.")
                continue
            try:
                odds_str = input("Enter Odds (Home Draw Away), e.g. '1.90 3.50 4.00' (enter to skip): ").strip()
                if odds_str:
                    oh, od, oa = map(float, odds_str.split())
                else:
                    oh, od, oa = 0.0, 0.0, 0.0
            except Exception:
                print("Invalid odds format.")
                continue

            res = predict_custom_match(
                home_real,
                away_real,
                oh,
                od,
                oa,
                state,
                meta_model,
                meta_cfg,
                mlp_model,
                mlp_meta,
                blend_cfg,
                logreg_model=logreg_model,
                logreg_meta=logreg_meta,
            )
            print("\n--- Prediction Summary ---")
            print(f"Match: {home_real} vs {away_real}")
            print(f"Elo: {res['elo'][0]:.1f} vs {res['elo'][1]:.1f}")
            print(f"Expected Goals: {res['xg'][0]:.3f} - {res['xg'][1]:.3f}")
            print(f"Base probs     : H={res['base'][0]:.3f} D={res['base'][1]:.3f} A={res['base'][2]:.3f}")
            print(f"Market probs   : H={res['market'][0]:.3f} D={res['market'][1]:.3f} A={res['market'][2]:.3f}")
            print(f"MktCorr probs  : H={res['market_corr'][0]:.3f} D={res['market_corr'][1]:.3f} A={res['market_corr'][2]:.3f}")
            print(f"XGBoost probs  : H={res['meta'][0]:.3f} D={res['meta'][1]:.3f} A={res['meta'][2]:.3f}")
            print(f"LogReg probs   : H={res['logreg'][0]:.3f} D={res['logreg'][1]:.3f} A={res['logreg'][2]:.3f}")
            print(f"MLP probs      : H={res['mlp'][0]:.3f} D={res['mlp'][1]:.3f} A={res['mlp'][2]:.3f}")
            print(f"Ensemble probs : H={res['ensemble'][0]:.3f} D={res['ensemble'][1]:.3f} A={res['ensemble'][2]:.3f}")
            pick = ["H", "D", "A"][int(np.argmax(res['ensemble']))]
            print(f"Final Pick     : {pick}")
            print("Top scorelines:")
            for (hg, ag), psc in res["scores"]:
                print(f"  {hg}-{ag}  ({psc:.3f})")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
