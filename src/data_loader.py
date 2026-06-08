"""Stable import surface for league data loading.

Re-exports :func:`src.data_processing.load_league_data` so callers can depend on
``src.data_loader`` without reaching into the larger data-processing module.
"""
from __future__ import annotations

from src.data_processing import load_league_data

__all__ = ["load_league_data"]
