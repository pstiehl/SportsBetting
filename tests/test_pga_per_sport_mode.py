"""Integration test: PGA appears in the per-sport mode table when scored.

Pins the wiring from ``flashcat.backtest.runner.SPORT_LOADERS`` ↔
``flashcat.build_site.resolve_sport_modes`` for the new ``pga`` sport.
"""

from __future__ import annotations

from flashcat.build_site import resolve_sport_modes


def test_pga_classified_research_with_low_sample():
    """A pga sport block with low n_bets is RESEARCH (not LIVE) by gate rule."""
    sb = {
        "per_sport": {
            "pga": {
                "n_events": 12,
                "sources": {},
                "blended": {
                    "n_events": 12,
                    "roi": 0.04,  # +4% looks "live" on ROI alone…
                    "brier": 0.23,
                    "wins": 6,
                    "losses": 6,
                },
            }
        }
    }
    modes = resolve_sport_modes(sb)
    assert "pga" in modes
    pga = modes["pga"]
    # …but sample size (12) is well below the 200-bet floor → RESEARCH.
    assert pga["mode"] == "research"
    assert pga["roi_str"] == "+4.0%"


def test_pga_can_go_live_with_clean_backtest():
    sb = {
        "per_sport": {
            "pga": {
                "n_events": 800,
                "sources": {},
                "blended": {
                    "n_events": 800,
                    "roi": 0.06,
                    "brier": 0.21,
                    "wins": 420,
                    "losses": 380,
                },
            }
        }
    }
    modes = resolve_sport_modes(sb)
    assert modes["pga"]["mode"] == "live"
    # ROI sits above the marginal ceiling so the marginal flag is False.
    assert modes["pga"]["marginal"] is False


def test_pga_negative_roi_stays_research():
    sb = {
        "per_sport": {
            "pga": {
                "n_events": 500,
                "sources": {},
                "blended": {
                    "n_events": 500,
                    "roi": -0.02,
                    "brier": 0.25,
                    "wins": 230,
                    "losses": 270,
                },
            }
        }
    }
    modes = resolve_sport_modes(sb)
    assert modes["pga"]["mode"] == "research"


def test_pga_listed_alongside_other_sports():
    """When the per-sport scoreboard carries multiple sports including pga,
    resolve_sport_modes returns a row for every one of them. Pins that
    adding pga didn't break the multi-sport dict shape."""
    sb = {
        "per_sport": {
            "nfl": {"n_events": 600, "sources": {},
                     "blended": {"n_events": 600, "roi": 0.12,
                                 "wins": 320, "losses": 280}},
            "pga": {"n_events": 300, "sources": {},
                     "blended": {"n_events": 300, "roi": 0.03,
                                 "wins": 150, "losses": 150}},
            "atp": {"n_events": 500, "sources": {},
                     "blended": {"n_events": 500, "roi": -0.06,
                                 "wins": 235, "losses": 265}},
        }
    }
    modes = resolve_sport_modes(sb)
    assert set(modes) == {"nfl", "pga", "atp"}
    # PGA at +3.0% is in the marginal band [+2.0%, +4.0%) under the
    # blender-de-dilution PR.
    assert modes["pga"]["mode"] == "live"
    assert modes["pga"]["marginal"] is True
