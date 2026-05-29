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


def test_v2_weights_load_with_per_sport_breakdown(tmp_path, monkeypatch):
    """update_weights writes a v2 payload with per-sport pools."""
    import json
    from flashcat import config as cfg
    from flashcat.model import reweight as rw

    scoreboard = tmp_path / "sb.json"
    weights_path = tmp_path / "w.json"
    scoreboard.write_text(json.dumps({
        "sources": {
            "nfl:fivethirtyeight-nfl-elo": {"n_events": 200, "brier": 0.21, "roi": 0.03},
            "nfl:market-close": {"n_events": 200, "brier": 0.20, "roi": 0.05},
            "mlb:fivethirtyeight-mlb-rating": {"n_events": 1000, "brier": 0.24, "roi": None},
            "mlb:fivethirtyeight-mlb-elo": {"n_events": 1000, "brier": 0.25, "roi": None},
            "tiny:source": {"n_events": 5, "brier": 0.10, "roi": 0.5},  # filtered out (n<50)
        },
        "per_sport": {
            "nfl": {"sources": {
                "fivethirtyeight-nfl-elo": {"n_events": 200, "brier": 0.21, "roi": 0.03},
                "market-close": {"n_events": 200, "brier": 0.20, "roi": 0.05},
            }},
            "mlb": {"sources": {
                "fivethirtyeight-mlb-rating": {"n_events": 1000, "brier": 0.24, "roi": None},
                "fivethirtyeight-mlb-elo": {"n_events": 1000, "brier": 0.25, "roi": None},
            }},
        },
    }))
    monkeypatch.setattr(cfg, "SOURCE_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("flashcat.model.blend.SOURCE_WEIGHTS_PATH", weights_path)

    payload = rw.update_weights(scoreboard_path=scoreboard, mode="brier")
    assert payload["schema"] == "v2"
    assert "nfl" in payload["by_sport"]
    assert "mlb" in payload["by_sport"]
    # Low-sample 'tiny:source' excluded.
    assert "tiny:source" not in payload["global"]
    # NFL pool: market-close has lower brier → larger weight.
    nfl = payload["by_sport"]["nfl"]
    assert nfl["market-close"] > nfl["fivethirtyeight-nfl-elo"]
    # MLB rating (lower brier) > MLB elo.
    mlb = payload["by_sport"]["mlb"]
    assert mlb["fivethirtyeight-mlb-rating"] > mlb["fivethirtyeight-mlb-elo"]


def test_hybrid_mode_blends_brier_and_roi(tmp_path, monkeypatch):
    import json
    from flashcat import config as cfg
    from flashcat.model import reweight as rw

    scoreboard = tmp_path / "sb.json"
    weights_path = tmp_path / "w.json"
    scoreboard.write_text(json.dumps({
        "sources": {},
        "per_sport": {
            "nfl": {"sources": {
                # source A: better brier, worse ROI
                "src-a": {"n_events": 100, "brier": 0.20, "roi": -0.05},
                # source B: worse brier, better ROI
                "src-b": {"n_events": 100, "brier": 0.22, "roi": 0.05},
            }},
        },
    }))
    monkeypatch.setattr(cfg, "SOURCE_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr("flashcat.model.blend.SOURCE_WEIGHTS_PATH", weights_path)

    p_brier = rw.update_weights(scoreboard_path=scoreboard, mode="brier")
    p_roi = rw.update_weights(scoreboard_path=scoreboard, mode="roi")
    p_hybrid = rw.update_weights(scoreboard_path=scoreboard, mode="brier_roi_hybrid")
    # Brier mode favors A; ROI mode favors B; hybrid sits between.
    a_brier = p_brier["by_sport"]["nfl"]["src-a"]
    a_roi = p_roi["by_sport"]["nfl"]["src-a"]
    a_hybrid = p_hybrid["by_sport"]["nfl"]["src-a"]
    assert a_roi < a_hybrid < a_brier


def test_blend_uses_per_sport_weights():
    """Blender prefers the by_sport pool when sport matches."""
    from datetime import datetime, timezone
    from flashcat.model.blend import blend_event
    from flashcat.types import Event, SourceProb

    weights = {
        "schema": "v2",
        "global": {"src-a": 0.5, "src-b": 0.5},
        "by_sport": {"nfl": {"src-a": 0.9, "src-b": 0.1}},
    }
    now = datetime.now(timezone.utc)
    ev = Event(
        event_id="t:1", sport="nfl", home="X", away="Y", commence_time=now,
        source_probs=[
            SourceProb(source="src-a", home_win_prob=0.8, captured_at=now),
            SourceProb(source="src-b", home_win_prob=0.4, captured_at=now),
        ],
    )
    blended = blend_event(ev, weights)
    # 0.9 * 0.8 + 0.1 * 0.4 = 0.76
    assert abs(blended.blended_home_prob - 0.76) < 1e-9


def test_blend_handles_v1_legacy_weights():
    """Legacy v1 (flat dict) weights still load and blend correctly."""
    from datetime import datetime, timezone
    from flashcat.model.blend import blend_event, load_weights
    from flashcat.types import Event, SourceProb

    weights = {"src-a": 0.7, "src-b": 0.3}  # v1 legacy shape
    now = datetime.now(timezone.utc)
    ev = Event(
        event_id="t:1", sport="nfl", home="X", away="Y", commence_time=now,
        source_probs=[
            SourceProb(source="src-a", home_win_prob=0.9, captured_at=now),
            SourceProb(source="src-b", home_win_prob=0.1, captured_at=now),
        ],
    )
    blended = blend_event(ev, weights)
    # 0.7*0.9 + 0.3*0.1 = 0.66
    assert abs(blended.blended_home_prob - 0.66) < 1e-9
