"""Regression tests: CFB appears in the per-sport mode resolver.

Phil's per-sport gate: a sport stays in RESEARCH until blended backtest
ROI ≥ +1% AND n_bets ≥ 200. PR #14 adds CFB; it should start in
RESEARCH regardless of how good the backtest looks, until both gates
clear.
"""

from __future__ import annotations

from flashcat.build_site import resolve_sport_modes
from flashcat.types import SPORTS


def test_cfb_in_sports_tuple():
    assert "cfb" in SPORTS


def test_cfb_research_mode_when_insufficient_backtest():
    """CFB with 50 bets and +5% ROI is RESEARCH because n_bets < 200."""
    sb = {
        "per_sport": {
            "cfb": {
                "blended": {
                    "roi": 0.05,
                    "wins": 28,
                    "losses": 22,
                    "n_events": 50,
                    "brier": 0.22,
                }
            }
        }
    }
    modes = resolve_sport_modes(sb)
    assert "cfb" in modes
    assert modes["cfb"]["mode"] == "research"
    assert "200" in modes["cfb"]["reason"] or "scored" in modes["cfb"]["reason"]


def test_cfb_research_mode_when_roi_below_floor():
    """CFB with 500 bets but -2% ROI stays RESEARCH (below +1% floor)."""
    sb = {
        "per_sport": {
            "cfb": {
                "blended": {
                    "roi": -0.02,
                    "wins": 240,
                    "losses": 260,
                    "n_events": 500,
                    "brier": 0.25,
                }
            }
        }
    }
    modes = resolve_sport_modes(sb)
    assert modes["cfb"]["mode"] == "research"


def test_cfb_can_go_live_when_both_gates_clear():
    """CFB with +2% ROI and 300 bets graduates to LIVE."""
    sb = {
        "per_sport": {
            "cfb": {
                "blended": {
                    "roi": 0.02,
                    "wins": 152,
                    "losses": 148,
                    "n_events": 300,
                    "brier": 0.23,
                }
            }
        }
    }
    modes = resolve_sport_modes(sb)
    assert modes["cfb"]["mode"] == "live"


def test_cfb_listed_in_multi_sport_backtest_default():
    """run_multi_sport_backtest's default sports list includes cfb."""
    from flashcat.backtest.runner import run_multi_sport_backtest
    import inspect
    src = inspect.getsource(run_multi_sport_backtest)
    # CFB is in the default sports list (signature reads it via sports or fallback)
    assert '"cfb"' in src


def test_cfb_in_sport_loaders_registry():
    """SPORT_LOADERS has a CFB entry with at least one connector."""
    from flashcat.backtest.runner import SPORT_LOADERS
    assert "cfb" in SPORT_LOADERS
    assert len(SPORT_LOADERS["cfb"]) >= 1
