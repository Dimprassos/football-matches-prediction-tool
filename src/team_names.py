"""Canonical team-name mapping across data sources.

Different feeds (football-data CSVs, Understat, API-Football) spell the same club
differently. :func:`normalize_team_name` maps each source's spelling to the single
canonical name used throughout the project, so merges line up correctly.
"""
from __future__ import annotations

import pandas as pd


# Per-league {source spelling -> canonical name}. Multiple encodings of accented
# names are listed so mojibake from different feeds still maps correctly.
TEAM_NAME_MAP = {
    "england": {
        "Spurs": "Tottenham",
        "Man Utd": "Man United",
        "Nottm Forest": "Nott'm Forest",
    },
    "spain": {
        "Athletic Club": "Ath Bilbao",
        "Atletico Madrid": "Ath Madrid",
        "Atl\u00e9tico Madrid": "Ath Madrid",
        "Atl\u00c3\u00a9tico Madrid": "Ath Madrid",
        "Atl\u00e9tico de Madrid": "Ath Madrid",
        "Atl\u00c3\u00a9tico de Madrid": "Ath Madrid",
        "CA Osasuna": "Osasuna",
        "Deportivo Alaves": "Alaves",
        "Deportivo Alav\u00e9s": "Alaves",
        "Deportivo Alav\u00c3\u00a9s": "Alaves",
        "Elche CF": "Elche",
        "FC Barcelona": "Barcelona",
        "Getafe CF": "Getafe",
        "Girona FC": "Girona",
        "Levante UD": "Levante",
        "Rayo Vallecano": "Vallecano",
        "RCD Espanyol de Barcelona": "Espanol",
        "RCD Mallorca": "Mallorca",
        "Real Betis": "Betis",
        "Real Oviedo": "Oviedo",
        "Real Sociedad": "Sociedad",
        "RC Celta": "Celta",
        "Sevilla FC": "Sevilla",
        "Valencia CF": "Valencia",
        "Villarreal CF": "Villarreal",
    },
    "italy": {
        "Inter Milan": "Inter",
        "AC Milan": "Milan",
        "Hellas Verona": "Verona",
    },
    "germany": {
        "1. FC Heidenheim 1846": "Heidenheim",
        "1. FC K\u00f6ln": "FC Koln",
        "1. FC K\u00c3\u00b6ln": "FC Koln",
        "1. FC Union Berlin": "Union Berlin",
        "1. FSV Mainz 05": "Mainz",
        "Bayer 04 Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund",
        "Borussia M\u00f6nchengladbach": "M'gladbach",
        "Borussia M\u00c3\u00b6nchengladbach": "M'gladbach",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Augsburg": "Augsburg",
        "FC Bayern M\u00fcnchen": "Bayern Munich",
        "FC Bayern M\u00c3\u00bcnchen": "Bayern Munich",
        "FC St. Pauli": "St Pauli",
        "Hamburger SV": "Hamburg",
        "Sport-Club Freiburg": "Freiburg",
        "SV Werder Bremen": "Werder Bremen",
        "TSG Hoffenheim": "Hoffenheim",
        "VfB Stuttgart": "Stuttgart",
        "VfL Wolfsburg": "Wolfsburg",
    },
    "france": {
        "Paris Saint-Germain": "Paris SG",
        "Olympique de Marseille": "Marseille",
        "Olympique Lyonnais": "Lyon",
        "RC Strasbourg Alsace": "Strasbourg",
        "Havre Athletic Club": "Le Havre",
        "Stade Rennais FC": "Rennes",
        "Stade Brestois 29": "Brest",
        "AS Monaco": "Monaco",
        "OGC Nice": "Nice",
        "Angers SCO": "Angers",
        "Toulouse FC": "Toulouse",
        "FC Nantes": "Nantes",
        "Paris FC": "Paris FC",
        "AJ Auxerre": "Auxerre",
        "FC Lorient": "Lorient",
        "FC Metz": "Metz",
        "LOSC Lille": "Lille",
        "RC Lens": "Lens",
    },
}


def normalize_team_name(name, league_folder):
    """Map a raw team name to its canonical form for ``league_folder`` (whitespace-normalised)."""
    if pd.isna(name):
        return name
    name = " ".join(str(name).strip().split())
    return TEAM_NAME_MAP.get(league_folder, {}).get(name, name)
