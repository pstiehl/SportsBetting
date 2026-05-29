"""Tests for American odds math."""

import math

from flashcat.types import (
    american_to_decimal,
    american_to_prob,
    american_to_profit,
    devig_two_way,
)


def test_american_to_prob_negative():
    # -200 → 200/(200+100) = 0.6667
    assert math.isclose(american_to_prob(-200), 2 / 3, rel_tol=1e-6)


def test_american_to_prob_positive():
    # +150 → 100/(150+100) = 0.40
    assert math.isclose(american_to_prob(150), 0.40, rel_tol=1e-6)


def test_american_to_decimal():
    assert math.isclose(american_to_decimal(-200), 1.5, rel_tol=1e-6)
    assert math.isclose(american_to_decimal(150), 2.5, rel_tol=1e-6)


def test_profit_on_winning_underdog():
    # $100 at +150 wins $150
    assert math.isclose(american_to_profit(150, 100.0), 150.0, rel_tol=1e-6)


def test_profit_on_winning_favorite():
    # $100 at -200 wins $50
    assert math.isclose(american_to_profit(-200, 100.0), 50.0, rel_tol=1e-6)


def test_devig_two_way():
    # 0.55 + 0.50 = 1.05 (vig). Devigged: 0.5238 / 0.4762
    h, a = devig_two_way(0.55, 0.50)
    assert math.isclose(h + a, 1.0, abs_tol=1e-9)
    assert math.isclose(h, 0.55 / 1.05, rel_tol=1e-6)
