"""Unit tests for the ESPN PGA scoreboard connector.

We never hit the real ESPN endpoint in tests — ``_fetch_scoreboard`` is
monkey-patched to return a captured-shape payload.
"""

from __future__ import annotations

from datetime import date

import pytest

from flashcat.sources.pga_espn_bpi import PGAESPNScoreboard, _logistic


SCOREBOARD_FIXTURE = {
    "events": [
        {
            "id": "401811949",
            "name": "Charles Schwab Challenge",
            "date": "2026-05-28T04:00Z",
            "competitions": [
                {
                    "competitors": [
                        {
                            "athlete": {"displayName": "Scottie Scheffler"},
                            "statistics": [
                                {"name": "scoreToPar", "value": -8.0,
                                 "displayValue": "-8"},
                            ],
                        },
                        {
                            "athlete": {"displayName": "Rory McIlroy"},
                            "statistics": [
                                {"name": "scoreToPar", "value": -5.0,
                                 "displayValue": "-5"},
                            ],
                        },
                        {
                            "athlete": {"displayName": "Xander Schauffele"},
                            "statistics": [
                                {"name": "scoreToPar", "value": -3.0,
                                 "displayValue": "-3"},
                            ],
                        },
                        {
                            "athlete": {"displayName": "Collin Morikawa"},
                            "statistics": [
                                {"name": "scoreToPar", "value": 0.0,
                                 "displayValue": "E"},
                            ],
                        },
                        # Mid-pack with no scoreToPar — should be skipped.
                        {
                            "athlete": {"displayName": "Jordan Smith"},
                            "statistics": [],
                        },
                    ]
                }
            ],
        }
    ]
}


def _stub_fetch(monkeypatch, payload):
    monkeypatch.setattr(
        PGAESPNScoreboard, "_fetch_scoreboard", lambda self: payload
    )


def test_logistic_zero_gap_is_half():
    assert _logistic(0.0) == pytest.approx(0.5, abs=1e-6)


def test_logistic_monotonic():
    assert _logistic(1.0) > _logistic(0.0)
    assert _logistic(-1.0) < _logistic(0.0)


def test_no_events_no_output(monkeypatch):
    _stub_fetch(monkeypatch, {"events": []})
    out = PGAESPNScoreboard().fetch_events(date(2026, 5, 26), date(2026, 6, 5))
    assert out == []


def test_wrong_sport_returns_empty():
    out = PGAESPNScoreboard().fetch_events(
        date(2026, 5, 26), date(2026, 6, 5), sport="atp"
    )
    assert out == []


def test_emits_adjacent_pairs(monkeypatch):
    _stub_fetch(monkeypatch, SCOREBOARD_FIXTURE)
    # 4 players have scoreToPar (Smith dropped) → 2 adjacent pairs.
    out = PGAESPNScoreboard().fetch_events(date(2026, 5, 26), date(2026, 6, 5))
    assert len(out) == 2
    for e in out:
        assert e.sport == "pga"
        assert e.league.startswith("PGA:")
        assert len(e.source_probs) == 1
        sp = e.source_probs[0]
        assert sp.source == "pga-espn-scoreboard"
        assert 0.001 <= sp.home_win_prob <= 0.999
        assert "leaderboard proxy" in sp.notes


def test_leader_is_favorite(monkeypatch):
    _stub_fetch(monkeypatch, SCOREBOARD_FIXTURE)
    out = PGAESPNScoreboard().fetch_events(date(2026, 5, 26), date(2026, 6, 5))
    top = out[0]
    # Scheffler (-8) is paired against McIlroy (-5): 3-stroke gap → ~0.77.
    names = {top.home, top.away}
    assert "Scottie Scheffler" in names
    assert "Rory McIlroy" in names
    scheff_prob = (
        top.source_probs[0].home_win_prob
        if top.home == "Scottie Scheffler"
        else 1 - top.source_probs[0].home_win_prob
    )
    assert 0.7 < scheff_prob < 0.85


def test_event_after_window_skipped(monkeypatch):
    _stub_fetch(monkeypatch, SCOREBOARD_FIXTURE)
    # Event is on 2026-05-28; ask for a window ending 2026-05-20.
    out = PGAESPNScoreboard().fetch_events(date(2026, 5, 14), date(2026, 5, 20))
    assert out == []


def test_top_level_score_fallback(monkeypatch):
    """Between rounds ESPN clears `statistics` and surfaces `score` instead."""
    payload = {
        "events": [
            {
                "id": "x",
                "name": "Round-Break Event",
                "date": "2026-05-28T04:00Z",
                "competitions": [
                    {
                        "competitors": [
                            {"athlete": {"displayName": "Player A"},
                             "score": "-10", "statistics": []},
                            {"athlete": {"displayName": "Player B"},
                             "score": "-7", "statistics": []},
                            {"athlete": {"displayName": "Player C"},
                             "score": "E", "statistics": []},
                            {"athlete": {"displayName": "Player D"},
                             "score": "+3", "statistics": []},
                            # WD player should be excluded.
                            {"athlete": {"displayName": "Player WD"},
                             "score": "WD", "statistics": []},
                        ]
                    }
                ],
            }
        ]
    }
    _stub_fetch(monkeypatch, payload)
    out = PGAESPNScoreboard().fetch_events(date(2026, 5, 26), date(2026, 6, 5))
    # 4 active players → 2 pairs (WD dropped).
    assert len(out) == 2


def test_only_one_player_skipped(monkeypatch):
    payload = {
        "events": [
            {
                "id": "x",
                "name": "Lonely Event",
                "date": "2026-05-28T04:00Z",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "athlete": {"displayName": "Solo Player"},
                                "statistics": [
                                    {"name": "scoreToPar", "value": -1.0}
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }
    _stub_fetch(monkeypatch, payload)
    out = PGAESPNScoreboard().fetch_events(date(2026, 5, 26), date(2026, 6, 5))
    assert out == []
