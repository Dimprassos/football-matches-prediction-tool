"""Console report of per-league test log loss for each model."""
from sklearn.metrics import log_loss


def print_per_league_test_metrics(leagues, per_league_test, probs_meta, probs_mlp, probs_ens):
    """Print a per-league table of base/market/meta/MLP/ensemble test log loss.

    The pooled meta/MLP/ensemble probability arrays are sliced back per league in the
    same order they were stacked, so each league's metrics align with its rows.
    """
    print("\n=== PER-LEAGUE TEST METRICS ===")
    print("=" * 80)

    league_slice_start = 0
    col_w = 10
    header = (
        f"{'League':<10} | {'N':>4} | "
        f"{'Base LL':>{col_w}} | {'Mkt LL':>{col_w}} | {'Meta LL':>{col_w}} | {'MLP LL':>{col_w}} | {'Ens LL':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for league in leagues:
        if league not in per_league_test:
            continue

        ld = per_league_test[league]
        y_l = ld["y"]
        pm_l = ld["p_model"]
        pmkt_l = ld["p_mkt"]
        n = len(y_l)

        meta_l = probs_meta[league_slice_start: league_slice_start + n]
        mlp_l = probs_mlp[league_slice_start: league_slice_start + n]
        ens_l = probs_ens[league_slice_start: league_slice_start + n]
        league_slice_start += n

        base_ll = round(log_loss(y_l, pm_l), 4)
        mkt_ll = round(log_loss(y_l, pmkt_l), 4)
        meta_ll = round(log_loss(y_l, meta_l), 4)
        mlp_ll = round(log_loss(y_l, mlp_l), 4)
        ens_ll = round(log_loss(y_l, ens_l), 4)

        print(
            f"{league.upper():<10} | {n:>4} | "
            f"{base_ll:>{col_w}} | {mkt_ll:>{col_w}} | {meta_ll:>{col_w}} | {mlp_ll:>{col_w}} | {ens_ll:>{col_w}}"
        )
