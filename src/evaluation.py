"""Betting simulation and diagnostic audits used to evaluate models economically.

A model is only useful here if it can beat the market, so this module turns
probabilities + odds into realistic betting outcomes: :func:`simulate_value_betting`
runs a fractional-Kelly value-betting simulation, :func:`betting_records` returns the
per-bet ledger (reused for bootstrap CIs and ROI), and the ``print_*`` functions are
console audits (market dependency, strategy comparison, profit profile, alignment).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def simulate_value_betting(
    probs,
    raw_odds,
    y_true,
    edge_threshold=0.05,
    kelly_fraction=0.25,
    max_stake=0.05,
    match_info=None,
    max_odds=10.0,
    max_ev=1.0,
    verbose=True,
):
    """
    Simulates betting using a Fractional Kelly Criterion.
    - edge_threshold: Minimum EV to place a bet.
    - kelly_fraction: Multiplier for the Kelly fraction (e.g., 0.25 for Quarter Kelly) to reduce variance.
    - max_stake: Maximum allowed percentage of bankroll to risk on a single bet (e.g., 0.05 = 5%).
    - max_odds: Ignore bets with odds higher than this value (prevents extreme longshot bias).
    - max_ev: Ignore bets with theoretical EV higher than this value (prevents model overconfidence traps).
    """
    stats = {
        "Home (1)": {"count": 0, "wins": 0, "invested": 0.0, "return": 0.0, "odds_sum": 0.0},
        "Draw (X)": {"count": 0, "wins": 0, "invested": 0.0, "return": 0.0, "odds_sum": 0.0},
        "Away (2)": {"count": 0, "wins": 0, "invested": 0.0, "return": 0.0, "odds_sum": 0.0},
    }
    
    if verbose and match_info is not None:
        print("\n--- Detailed Betting Log (Test Set) ---")

    for i in range(len(probs)):
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]

        if not (np.isfinite(o_h) and np.isfinite(o_d) and np.isfinite(o_a)):
            continue

        # Add probabilities to the tuple so we can calculate Kelly later
        evs = [
            (p_h * o_h - 1, 0, o_h, "Home (1)", p_h),
            (p_d * o_d - 1, 1, o_d, "Draw (X)", p_d),
            (p_a * o_a - 1, 2, o_a, "Away (2)", p_a),
        ]

        # Find the best option based on EV
        best_ev, choice, odds_taken, label, prob_taken = max(evs, key=lambda x: x[0])

        # Ignore extreme odds and implausibly high EV estimates.
        if odds_taken > max_odds or best_ev > max_ev:
            continue

        if best_ev > edge_threshold:
            # 1. Calculate Kelly Stake: f* = EV / (odds - 1)
            b = odds_taken - 1.0
            f_star = best_ev / b if b > 0 else 0.0
            
            # 2. Apply Fractional Kelly and Cap it
            stake = f_star * kelly_fraction
            stake = min(stake, max_stake)
            
            # Minimum stake safety (e.g., don't place micro-bets less than 0.1%)
            if stake < 0.001:
                continue

            # 3. Record the bet
            stats[label]["count"] += 1
            stats[label]["invested"] += stake
            stats[label]["odds_sum"] += odds_taken

            if choice == y_true[i]:
                stats[label]["wins"] += 1
                stats[label]["return"] += stake * odds_taken
                
            if verbose and match_info is not None:
                m = match_info[i]
                date_str = m['date'].strftime('%Y-%m-%d') if hasattr(m['date'], 'strftime') else m['date']
                res_str = "WIN" if choice == y_true[i] else "LOSS"
                print(f"[{date_str}] {m['home_team']} vs {m['away_team']} | Bet: {label} @ {odds_taken:.2f} | EV: {best_ev*100:.1f}% | Stake: {stake*100:.1f}% | {res_str}")

    if verbose:
        print(f"\n{'Market Segment':<15} | {'Bets':<5} | {'Win%':<7} | {'ROI%':<8}")
        print("-" * 45)

    total_bets = 0
    total_wins = 0
    total_inv = 0.0
    total_ret = 0.0
    total_odds_sum = 0.0

    for label, s in stats.items():
        if s["count"] > 0:
            win_pc = (s["wins"] / s["count"]) * 100
            roi = ((s["return"] - s["invested"]) / s["invested"]) * 100
            if verbose:
                print(f"{label:<15} | {s['count']:<5} | {win_pc:>6.1f}% | {roi:>7.2f}%")

            total_bets += s["count"]
            total_wins += s["wins"]
            total_inv += s["invested"]
            total_ret += s["return"]
            total_odds_sum += s["odds_sum"]

    final_profit = total_ret - total_inv
    final_roi = (final_profit / total_inv * 100) if total_inv > 0 else 0
    avg_odds = (total_odds_sum / total_bets) if total_bets > 0 else 0

    if verbose:
        print("-" * 45)
        print(f"{'TOTAL':<15} | {total_bets:<5} | {'-':>7} | {final_roi:>7.2f}%")

    return total_bets, total_wins, final_profit, final_roi, avg_odds


CLASS_NAMES = ["H", "D", "A"]
ODDS_SUFFIXES = ["home", "draw", "away"]


def betting_records(
    probs,
    raw_odds,
    y_true,
    edge_threshold=0.05,
    kelly_fraction=0.25,
    max_stake=0.05,
    match_info=None,
    max_odds=10.0,
    max_ev=1.0,
):
    """Return the per-bet ledger (one row per placed value bet) under the same rules.

    Same value-betting/fractional-Kelly logic as :func:`simulate_value_betting`, but
    instead of printing a summary it returns the individual bet records (stake, odds,
    outcome, profit, CLV...). This is the reusable source for ROI and bootstrap CIs.
    """
    rows = []
    for i in range(len(probs)):
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]

        if not (np.isfinite(o_h) and np.isfinite(o_d) and np.isfinite(o_a)):
            continue

        evs = [
            (p_h * o_h - 1, 0, o_h, "Home (1)", p_h),
            (p_d * o_d - 1, 1, o_d, "Draw (X)", p_d),
            (p_a * o_a - 1, 2, o_a, "Away (2)", p_a),
        ]
        best_ev, choice, odds_taken, label, prob_taken = max(evs, key=lambda x: x[0])

        if odds_taken > max_odds or best_ev > max_ev:
            continue
        if best_ev <= edge_threshold:
            continue

        b = odds_taken - 1.0
        f_star = best_ev / b if b > 0 else 0.0
        stake = min(f_star * kelly_fraction, max_stake)

        if stake < 0.001:
            continue

        won = int(choice == y_true[i])
        ret = stake * odds_taken if won else 0.0
        profit = ret - stake

        info = match_info[i] if match_info is not None else {}
        suffix = ODDS_SUFFIXES[int(choice)]
        open_odds_taken = info.get(f"open_odds_{suffix}", np.nan)
        close_odds_taken = info.get(f"close_odds_{suffix}", np.nan)
        open_odds_value = pd.to_numeric(open_odds_taken, errors="coerce")
        close_odds_value = pd.to_numeric(close_odds_taken, errors="coerce")
        clv_decimal = np.nan
        if np.isfinite(open_odds_value) and np.isfinite(close_odds_value) and float(close_odds_value) > 1.0001:
            clv_decimal = (float(open_odds_value) / float(close_odds_value)) - 1.0
        rows.append({
            "idx": i,
            "date": info.get("date", None),
            "league": info.get("league", None),
            "home_team": info.get("home_team", None),
            "away_team": info.get("away_team", None),
            "y_true": int(y_true[i]),
            "y_true_label": CLASS_NAMES[int(y_true[i])],
            "pred_choice": int(choice),
            "pred_label": CLASS_NAMES[int(choice)],
            "prob_taken": float(prob_taken),
            "odds_taken": float(odds_taken),
            "open_odds_taken": float(open_odds_value) if np.isfinite(open_odds_value) else np.nan,
            "close_odds_taken": float(close_odds_value) if np.isfinite(close_odds_value) else np.nan,
            "clv_decimal": float(clv_decimal) if np.isfinite(clv_decimal) else np.nan,
            "best_ev": float(best_ev),
            "stake": float(stake),
            "return": float(ret),
            "profit": float(profit),
            "won": int(won),
            "p_h": float(p_h),
            "p_d": float(p_d),
            "p_a": float(p_a),
            "o_h": float(o_h),
            "o_d": float(o_d),
            "o_a": float(o_a),
        })

    return pd.DataFrame(rows)


def print_alignment_audit(probs, raw_odds, y_true, match_info, title="ALIGNMENT AUDIT", n=20):
    """Print a sample of matches showing model probs vs odds vs outcome (sanity check)."""
    print("\n" + "=" * 70)
    print(f"=== {title} ===")
    print("=" * 70)

    n = min(n, len(probs))
    rows = []
    for i in range(n):
        info = match_info[i]
        p_h, p_d, p_a = probs[i]
        o_h, o_d, o_a = raw_odds[i]
        pred = int(np.argmax(probs[i]))
        evs = np.array([p_h * o_h - 1, p_d * o_d - 1, p_a * o_a - 1], dtype=float)
        best_ev_idx = int(np.argmax(evs))

        rows.append({
            "idx": i,
            "date": info["date"],
            "home": info["home_team"],
            "away": info["away_team"],
            "true": CLASS_NAMES[int(y_true[i])],
            "pred": CLASS_NAMES[pred],
            "best_ev_pick": CLASS_NAMES[best_ev_idx],
            "p_h": round(float(p_h), 3),
            "p_d": round(float(p_d), 3),
            "p_a": round(float(p_a), 3),
            "o_h": round(float(o_h), 2),
            "o_d": round(float(o_d), 2),
            "o_a": round(float(o_a), 2),
            "ev_h": round(float(evs[0]), 3),
            "ev_d": round(float(evs[1]), 3),
            "ev_a": round(float(evs[2]), 3),
        })

    print(pd.DataFrame(rows).to_string(index=False))


def print_strategy_comparison(strategy_probs, raw_odds, y_true, edge_threshold=0.05):
    """Compare betting ROI across models/strategies side by side."""
    print("\n" + "=" * 70)
    print("=== BETTING STRATEGY COMPARISON ===")
    print("=" * 70)

    rows = []
    for name, probs in strategy_probs.items():
        bets, wins, profit, roi, avg_odds = simulate_value_betting(
            probs,
            raw_odds,
            y_true,
            edge_threshold=edge_threshold,
            match_info=None,
            verbose=False,
        )
        hit_rate = (wins / bets * 100.0) if bets > 0 else 0.0
        rows.append({
            "strategy": name,
            "bets": bets,
            "wins": wins,
            "hit_rate_%": round(hit_rate, 2),
            "avg_odds": round(avg_odds, 2),
            "profit_units": round(profit, 3),
            "roi_%": round(roi, 2),
        })

    print(pd.DataFrame(rows).sort_values("roi_%", ascending=False).to_string(index=False))


def print_market_dependency_audit(y_true, p_base, p_market, p_meta, p_mlp, p_ens):
    """Audit how far each model's probabilities deviate from the market (and whether it helps)."""
    print("\n" + "=" * 70)
    print("=== MARKET DEPENDENCY AUDIT ===")
    print("=" * 70)

    rows = []
    for name, probs in [
        ("base", p_base),
        ("market", p_market),
        ("meta", p_meta),
        ("mlp", p_mlp),
        ("ensemble", p_ens),
    ]:
        rows.append({
            "model": name,
            "logloss": round(float(log_loss(y_true, probs)), 4),
            "argmax_acc_%": round(float((np.argmax(probs, axis=1) == y_true).mean() * 100), 2),
            "avg_conf": round(float(np.max(probs, axis=1).mean()), 4),
            "draw_rate_%": round(float((np.argmax(probs, axis=1) == 1).mean() * 100), 2),
        })

    print(pd.DataFrame(rows).to_string(index=False))

    agreement_rows = []
    for name, probs in [
        ("base", p_base),
        ("meta", p_meta),
        ("mlp", p_mlp),
        ("ensemble", p_ens),
    ]:
        agreement_rows.append({
            "vs_market": name,
            "argmax_agreement_%": round(float((np.argmax(probs, axis=1) == np.argmax(p_market, axis=1)).mean() * 100), 2),
            "mean_abs_diff": round(float(np.mean(np.abs(probs - p_market))), 5),
        })

    print("\nAgreement against MARKET:")
    print(pd.DataFrame(agreement_rows).to_string(index=False))


def print_profit_profile_audit(probs, raw_odds, y_true, match_info, edge_threshold=0.05):
    """Break down betting profit by odds band / edge to expose where returns come from."""
    print("\n" + "=" * 70)
    print("=== PROFIT PROFILE AUDIT ===")
    print("=" * 70)

    bets_df = betting_records(
        probs,
        raw_odds,
        y_true,
        edge_threshold=edge_threshold,
        match_info=match_info,
    )

    if bets_df.empty:
        print("No bets placed.")
        return

    invested = bets_df["stake"].sum()
    returned = bets_df["return"].sum()
    profit = bets_df["profit"].sum()
    roi = (profit / invested * 100.0) if invested > 0 else 0.0

    print(f"bets={len(bets_df)} | invested={invested:.4f} | returned={returned:.4f} | profit={profit:.4f} | roi={roi:.2f}%")
    print(f"mean_stake={bets_df['stake'].mean():.4f} | median_stake={bets_df['stake'].median():.4f} | max_stake={bets_df['stake'].max():.4f}")

    print("\nTop 10 winning bets by profit:")
    print(
        bets_df.sort_values("profit", ascending=False)[
            ["date", "home_team", "away_team", "pred_label", "y_true_label", "odds_taken", "best_ev", "stake", "profit"]
        ].head(10).to_string(index=False)
    )

    print("\nTop 10 losing bets by loss:")
    print(
        bets_df.sort_values("profit", ascending=True)[
            ["date", "home_team", "away_team", "pred_label", "y_true_label", "odds_taken", "best_ev", "stake", "profit"]
        ].head(10).to_string(index=False)
    )
