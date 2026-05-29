"""Tests for adaptive reweighting."""

import math

from flashcat.model.reweight import softmax


def test_softmax_uniform_for_equal_inputs():
    w = softmax({"a": 1.0, "b": 1.0, "c": 1.0})
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    for v in w.values():
        assert math.isclose(v, 1 / 3, rel_tol=1e-6)


def test_softmax_concentrates_on_max():
    w = softmax({"a": 0.0, "b": 1.0}, temperature=4.0)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    assert w["b"] > w["a"]
    # temp=4, gap=1 → softmax of (0, 4) → e^4/(1+e^4) ≈ 0.982
    assert math.isclose(w["b"], math.exp(4) / (1 + math.exp(4)), rel_tol=1e-6)


def test_softmax_is_simplex():
    w = softmax({"a": -0.3, "b": -0.25, "c": -0.27})
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    for v in w.values():
        assert 0 <= v <= 1


def test_softmax_empty():
    assert softmax({}) == {}
