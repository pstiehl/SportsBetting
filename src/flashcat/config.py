"""Runtime configuration and path constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
EVENT_PAGES_DIR = DOCS_DIR / "event"

DB_PATH = DATA_DIR / "flashcat.db"
SOURCE_WEIGHTS_PATH = DATA_DIR / "source_weights.json"
SOURCE_SCOREBOARD_PATH = DATA_DIR / "source_scoreboard.json"

# Flat bet size — $100 on every event.
FLAT_STAKE: float = 100.0

# Tie-breaker band: when blended prob is within [0.48, 0.52],
# bet the underdog by moneyline (Phil's "favorite is a sucker's bet" hunch).
TIE_BREAK_BAND = (0.48, 0.52)

# Signal thresholds
CHALK_OVERPRICED_DELTA = 0.05  # implied_fav_prob > blended_fav_prob + 0.05
BOOK_DISPERSION_THRESHOLD = 0.04  # max-min implied prob across books on dog side


def the_odds_api_key() -> str | None:
    return os.getenv("THE_ODDS_API_KEY")


def ensure_dirs() -> None:
    for d in (DATA_DIR, SAMPLES_DIR, CACHE_DIR, DOCS_DIR, ASSETS_DIR, EVENT_PAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
