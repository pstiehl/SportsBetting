"""Tests for the probability blender + pick rule."""

import math

from flashcat.model.blend import _resolve_weights, blend_event


def test_resolve_weights_uniform_default():
    w = _resolve_weights(["a", "b"], {})
    assert math.isclose(w["a"], 0.5)
    assert math.isclose(w["b"], 0.5)


def test_resolve_weights_normalize():
    w = _resolve_weights(["a", "b"], {"a": 3.0, "b": 1.0})
    assert math.isclose(w["a"], 0.75)
    assert math.isclose(w["b"], 0.25)


def test_resolve_weights_missing_source_gets_default():
    w = _resolve_weights(["a", "b"], {"a": 1.0})
    # b defaults to 1.0 (equal-weight); both normalize to 0.5
    assert math.isclose(w["a"], 0.5)
    assert math.isclose(w["b"], 0.5)


def test_blend_equal_weights(basic_event):
    e = blend_event(basic_event, weights={})
    # Two sources at 0.60 and 0.70, equal weight → 0.65
    assert math.isclose(e.blended_home_prob, 0.65, rel_tol=1e-6)
    assert e.pick == "home"
    assert math.isclose(e.pick_prob, 0.65, rel_tol=1e-6)


def test_blend_unequal_weights(basic_event):
    # src-a weight 3, src-b weight 1 → 0.75*0.60 + 0.25*0.70 = 0.625
    e = blend_event(basic_event, weights={"src-a": 3.0, "src-b": 1.0})
    assert math.isclose(e.blended_home_prob, 0.625, rel_tol=1e-6)
    # Market devigged: home -195 avg ≈ 0.639. Blended home 0.625 is BELOW
    # the devigged market on home, meaning the model thinks home is less
    # likely than the market does — the model's edge is on the AWAY side.
    # The pick must reflect that. (Pre-fix, pick_side ignored the market
    # and picked home whenever blended_home >= 0.5, which is the root cause
    # of Phil's "Recommended Plays with negative edge" bug.)
    assert e.pick == "away"
    assert math.isclose(e.pick_prob, 0.375, rel_tol=1e-6)


def test_blend_equal_weights_picks_higher_edge_side(basic_event):
    # Equal weights → blended 0.65. Market devigged home ≈ 0.639. Blended
    # 0.65 > 0.639 → edge on home, pick home.
    e = blend_event(basic_event, weights={})
    assert math.isclose(e.blended_home_prob, 0.65, rel_tol=1e-6)
    assert e.pick == "home"


def test_tie_breaker_bets_underdog(coinflip_event):
    """In the tie-breaker band, bet the underdog by moneyline.
    src-a 0.49 + src-b 0.51 → blended 0.50. Moneyline: home -115 (fav). Bet away (dog).
    """
    e = blend_event(coinflip_event, weights={})
    assert math.isclose(e.blended_home_prob, 0.50, rel_tol=1e-6)
    assert e.pick == "away"
    assert math.isclose(e.pick_prob, 0.50, rel_tol=1e-6)


def test_blend_no_sources(basic_event):
    basic_event.source_probs = []
    e = blend_event(basic_event, weights={})
    assert e.blended_home_prob is None
    assert e.pick is None
