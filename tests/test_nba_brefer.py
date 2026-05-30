"""Unit tests for the NBA Basketball-Reference SRS connector."""

from __future__ import annotations

import math
from datetime import date

import pytest

from flashcat.sources import nba_brefer as nbb
from flashcat.sources.nba_brefer import (
    NBA_MARGIN_SIGMA,
    NBA_HFA_POINTS,
    NBABasketballReferenceSRS,
    diff_to_home_prob,
)


def test_diff_to_home_prob_calibrated_to_sigma_11():
    # +11 point diff should yield ~84% (1 sigma).
    p = diff_to_home_prob(11.0)
    assert 0.80 < p < 0.86


def test_diff_to_home_prob_zero_neutral():
    p = diff_to_home_prob(0.0)
    assert abs(p - 0.5) < 1e-6


def test_diff_to_home_prob_bounded():
    assert 0.05 <= diff_to_home_prob(100.0) <= 0.95
    assert 0.05 <= diff_to_home_prob(-100.0) <= 0.95


def test_other_sports_return_nothing(monkeypatch):
    src = NBABasketballReferenceSRS()
    assert src.fetch_events(date(2024, 12, 1), date(2024, 12, 5), sport="mlb") == []


def test_connector_returns_empty_when_offline(monkeypatch):
    """Network failure → empty list, not raise.

    Both the bref path (``_cached_get``) and the stats.nba.com fallback
    (``_fetch_team_ratings_nba_stats`` / ``_fetch_schedule_nba_stats``)
    are stubbed to None so the connector has nowhere to source data from.
    """
    monkeypatch.setattr(nbb, "_cached_get", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nbb, "_fetch_team_ratings_nba_stats",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nbb, "_fetch_schedule_nba_stats",
                        lambda *_args, **_kwargs: [])
    src = NBABasketballReferenceSRS()
    out = src.fetch_events(date(2024, 12, 1), date(2024, 12, 5))
    assert out == []


def test_hfa_constant_matches_spec():
    assert NBA_HFA_POINTS == 2.5
    assert NBA_MARGIN_SIGMA == 11.0


# --- PR #12: polite UA headers fix --------------------------------------------


def test_bref_headers_are_polite_and_identify_project():
    """basketball-reference.com 403s on generic UAs. We send a realistic,
    project-identifying UA with browser-like Accept headers and a Referer."""
    from flashcat.sources.nba_brefer import BREF_HEADERS, BREF_UA

    assert "flashcat-research" in BREF_UA
    assert "github.com/pstiehl/SportsBetting" in BREF_UA
    assert BREF_HEADERS["User-Agent"] == BREF_UA
    assert "text/html" in BREF_HEADERS["Accept"]
    assert BREF_HEADERS["Accept-Language"].startswith("en-US")
    assert BREF_HEADERS["Referer"] == "https://www.basketball-reference.com/"


def test_fetch_team_ratings_passes_polite_headers(monkeypatch):
    """``fetch_team_ratings`` forwards ``BREF_HEADERS`` into ``_cached_get``."""
    captured = {}

    def fake_get(url, cache_file, *, ttl_seconds=3600, headers=None, timeout=10.0):
        captured["url"] = url
        captured["headers"] = headers
        return None

    monkeypatch.setattr(nbb, "_cached_get", fake_get)
    # Stub the nba-stats fallback so this test is hermetic.
    monkeypatch.setattr(nbb, "_fetch_team_ratings_nba_stats",
                        lambda *_a, **_kw: None)
    nbb.fetch_team_ratings(2024)
    assert captured["headers"] is not None
    assert "flashcat-research" in captured["headers"]["User-Agent"]


# --- PR #13: stats.nba.com fallback for bref 403s -----------------------------


def test_fetch_team_ratings_falls_back_to_nba_stats_on_bref_404(monkeypatch):
    """When bref returns None (network error / 403), ratings should come
    from the ``_fetch_team_ratings_nba_stats`` fallback. The fallback is
    documented in PR #13 as the 'data.nba.com' fallback path required by
    the original PR #12 ticket.
    """
    monkeypatch.setattr(nbb, "_cached_get", lambda *_a, **_kw: None)
    fake_ratings = {
        "Boston Celtics": {"srs": 11.3, "pace": 0.0, "ortg": 120.6, "drtg": 109.2},
        "Denver Nuggets": {"srs": 5.3, "pace": 0.0, "ortg": 114.9, "drtg": 109.6},
    }
    monkeypatch.setattr(nbb, "_fetch_team_ratings_nba_stats",
                        lambda *_a, **_kw: fake_ratings)
    got = nbb.fetch_team_ratings(2024)
    assert got == fake_ratings


def test_fetch_team_ratings_falls_back_when_bref_html_is_malformed(monkeypatch):
    """bref sometimes returns 200 with a Cloudflare interstitial instead of
    the ratings table. The connector must still fall through to nba-stats.
    """
    interstitial = b"<html><body><h1>Just a moment...</h1></body></html>"
    monkeypatch.setattr(nbb, "_cached_get", lambda *_a, **_kw: interstitial)
    fake_ratings = {"Boston Celtics": {"srs": 11.3, "pace": 0.0,
                                       "ortg": 120.6, "drtg": 109.2}}
    monkeypatch.setattr(nbb, "_fetch_team_ratings_nba_stats",
                        lambda *_a, **_kw: fake_ratings)
    got = nbb.fetch_team_ratings(2024)
    assert got == fake_ratings


def test_connector_uses_nba_stats_schedule_when_bref_schedule_empty(monkeypatch):
    """End-to-end: when bref schedule comes back empty but ratings + nba-stats
    schedule work, the connector should still emit events."""
    fake_ratings = {
        "Boston Celtics": {"srs": 8.0, "pace": 0.0, "ortg": 0.0, "drtg": 0.0},
        "Brooklyn Nets":  {"srs": -3.0, "pace": 0.0, "ortg": 0.0, "drtg": 0.0},
    }
    fake_schedule = [
        {"date": date(2024, 12, 3), "home": "Boston Celtics", "away": "Brooklyn Nets"},
    ]
    monkeypatch.setattr(nbb, "_cached_get", lambda *_a, **_kw: None)
    monkeypatch.setattr(nbb, "_fetch_team_ratings_nba_stats",
                        lambda *_a, **_kw: fake_ratings)
    monkeypatch.setattr(nbb, "_fetch_schedule_nba_stats",
                        lambda *_a, **_kw: fake_schedule)
    src = NBABasketballReferenceSRS()
    out = src.fetch_events(date(2024, 12, 1), date(2024, 12, 5))
    assert len(out) == 1
    ev = out[0]
    assert ev.home == "Boston Celtics"
    assert ev.away == "Brooklyn Nets"
    sp = ev.source_probs[0]
    assert sp.source == "nba-bref-srs-pace"
    # 8 - (-3) + 2.5 = 13.5 → well above 0.5
    assert sp.home_win_prob > 0.85
