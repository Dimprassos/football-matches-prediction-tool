"""Generate historical ``lineup_strength`` context from the Understat player stats.

Reconstructs each match's starting XI from ``player_match_stats.csv`` and scores it with
the leakage-safe rolling player strength (only prior matches feed each player), producing
per-fixture ``home_lineup_strength``/``away_lineup_strength``. The result is merged into
``data/external/match_context.csv`` (existing weather / API-Football / team-news columns
are preserved), so the pipeline's lineup-strength features get populated for training.

Prerequisite: build the player datasets first with
``python src/update_understat.py --players-only``.

Usage:
    python scripts/build_player_lineup_context.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root on path

import numpy as np
import pandas as pd

from src.external_context import MATCH_CONTEXT_FILE
from src.player_context import (
    EXTERNAL_DATA_DIR,
    build_lineup_strength_context,
    load_player_context_tables,
    merge_player_match_context,
)


def main() -> None:
    start = time.time()
    tables = load_player_context_tables(EXTERNAL_DATA_DIR)
    if tables.match_stats.empty:
        raise SystemExit(
            "No player_match_stats.csv found. Run "
            "`python src/update_understat.py --players-only` first."
        )
    print(f"Loaded {len(tables.match_stats):,} player-match rows.", flush=True)

    generated = build_lineup_strength_context(tables.match_stats)
    home = generated["home_lineup_strength"]
    away = generated["away_lineup_strength"]
    print(
        f"Generated {len(generated):,} fixtures | "
        f"home_lineup_strength finite {home.notna().sum():,} ({100 * home.notna().mean():.1f}%), "
        f"away {away.notna().sum():,} ({100 * away.notna().mean():.1f}%)",
        flush=True,
    )

    existing = pd.read_csv(MATCH_CONTEXT_FILE) if MATCH_CONTEXT_FILE.exists() else None
    merged = merge_player_match_context(existing, generated)
    MATCH_CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MATCH_CONTEXT_FILE, index=False)
    print(
        f"Wrote {len(merged):,} rows -> {MATCH_CONTEXT_FILE} in {time.time() - start:.0f}s",
        flush=True,
    )
    print("PLAYER_LINEUP_CONTEXT_DONE", flush=True)


if __name__ == "__main__":
    main()
