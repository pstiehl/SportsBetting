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


def use_samples_fallback() -> bool:
    """Opt-in flag to allow stale-sample fallback for local dev.

    CI never sets this. Production builds fail loud when no live data arrives.
    Set FLASHCAT_USE_SAMPLES=1 (or true/yes) locally if you want offline mode.
    """
    val = os.getenv("FLASHCAT_USE_SAMPLES", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def stake_mode() -> str:
    """Stake sizing: flat, kelly_quarter, kelly_half, kelly_full. Default kelly_quarter."""
    return os.getenv("FLASHCAT_STAKE_MODE", "kelly_quarter").strip().lower()


def edge_threshold() -> float:
    """Minimum edge (blended_prob - devigged_market) to take a bet. Default 0.03."""
    try:
        return float(os.getenv("FLASHCAT_EDGE_THRESHOLD", "0.03"))
    except Exception:
        return 0.03


def bankroll() -> float:
    """Notional bankroll for Kelly sizing on the rendered site. Default $10,000."""
    try:
        return float(os.getenv("FLASHCAT_BANKROLL", "10000"))
    except Exception:
        return 10_000.0


class NoLiveDataError(RuntimeError):
    """Raised when no live source returned events for any in-season sport.

    The build pipeline must fail loud (CI red) rather than silently shipping
    stale samples to the rendered site. Opt out with FLASHCAT_USE_SAMPLES=1.
    """


def ensure_dirs() -> None:
    for d in (DATA_DIR, SAMPLES_DIR, CACHE_DIR, DOCS_DIR, ASSETS_DIR, EVENT_PAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
