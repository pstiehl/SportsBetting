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
    """Network failure → empty list, not raise."""
    monkeypatch.setattr(nbb, "_cached_get", lambda *_args, **_kwargs: None)
    src = NBABasketballReferenceSRS()
    out = src.fetch_events(date(2024, 12, 1), date(2024, 12, 5))
    assert out == []


def test_hfa_constant_matches_spec():
    assert NBA_HFA_POINTS == 2.5
    assert NBA_MARGIN_SIGMA == 11.0
