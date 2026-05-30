"""Regression: site stays RESEARCH MODE when any per-sport blend is negative.

Phil's rule (2026-05-29): if ANY sport's blended ROI is negative, the entire
site shows the RESEARCH-MODE badge and suppresses stake recommendations.
This test pins that behavior so future refactors can't quietly remove the
gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def test_negative_per_sport_blended_roi_triggers_research_mode(tmp_path, monkeypatch):
    from flashcat.build_site import _research_mode_state

    sb = {
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 1000, "roi": 0.10, "brier": 0.22}},
        "per_sport": {
            "atp": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": -0.05, "brier": 0.20}},
            "nfl": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": 0.08, "brier": 0.21}},
        },
        "blended_overall": {"n_events": 200, "wagered": 2000, "profit": 195, "roi": 0.10},
    }
    state = _research_mode_state(sb)
    assert state["research_mode"] is True
    assert "ATP" in (state["reason"] or "").upper()


def test_all_positive_per_sport_allows_live(tmp_path):
    from flashcat.build_site import _research_mode_state

    sb = {
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 1000, "roi": 0.10, "brier": 0.22}},
        "per_sport": {
            "atp": {"n_events": 500, "sources": {}, "blended": {"n_events": 500, "roi": 0.05, "brier": 0.20, "wins": 260, "losses": 240}},
            "nfl": {"n_events": 500, "sources": {}, "blended": {"n_events": 500, "roi": 0.08, "brier": 0.21, "wins": 270, "losses": 230}},
        },
        "blended_overall": {"n_events": 200, "wagered": 2000, "profit": 195, "roi": 0.10},
    }
    state = _research_mode_state(sb)
    assert state["research_mode"] is False


def test_grouped_layout_pushes_negative_roi_sport_into_research_bucket(tmp_path, monkeypatch):
    """Even when site clears the gate, individual sport with negative ROI
    means its picks land in the RESEARCH bucket, not RECOMMENDED."""
    from flashcat import build_site as bs
    from flashcat import config as cfg
    from flashcat.types import BookLine, Event, SourceProb

    tmp_docs = tmp_path / "docs"
    tmp_assets = tmp_docs / "assets"
    tmp_events = tmp_docs / "event"
    monkeypatch.setattr(cfg, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(cfg, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(cfg, "EVENT_PAGES_DIR", tmp_events)
    monkeypatch.setattr(bs, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(bs, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(bs, "EVENT_PAGES_DIR", tmp_events)

    sb_path = tmp_path / "data" / "source_scoreboard.json"
    sb_path.parent.mkdir(parents=True, exist_ok=True)
    sb_path.write_text(json.dumps({
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 200, "roi": -0.04, "brier": 0.22}},
        "per_sport": {
            "atp": {"n_events": 500, "sources": {}, "blended": {"n_events": 500, "roi": -0.05, "brier": 0.20, "wagered": 50000, "profit": -2500, "wins": 240, "losses": 260}},
        },
        "blended_overall": {"n_events": 100, "wagered": 1000, "profit": -50, "roi": -0.05},
    }))
    monkeypatch.setattr(cfg, "SOURCE_SCOREBOARD_PATH", sb_path)
    monkeypatch.setattr(bs, "SOURCE_SCOREBOARD_PATH", sb_path)

    now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="atp:1",
        sport="atp",
        home="Player A",
        away="Player B",
        commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.62, captured_at=now)],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=now),
            BookLine(book="dk", side="away", american=-110, captured_at=now),
        ],
        blended_home_prob=0.62,
        pick="home",
        pick_prob=0.62,
    )
    bs.build([ev])
    html = (tmp_docs / "index.html").read_text()
    # Site stays RESEARCH MODE because ATP blended ROI is negative.
    assert "RESEARCH" in html.upper()
    # The Research-Mode Picks section is rendered.
    assert "Research-Mode Picks" in html or "research-mode picks" in html.lower()


# -----------------------------------------------------------------------------
# Per-sport LIVE/RESEARCH mode resolver (PR #12)
# -----------------------------------------------------------------------------


def _make_sb(per_sport: dict) -> dict:
    return {
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 1000, "roi": 0.05, "brier": 0.22}},
        "per_sport": per_sport,
        "blended_overall": {"n_events": 1000, "wagered": 100000, "profit": 5000, "roi": 0.05},
    }


def test_resolve_sport_modes_live_sport_with_edge_clearing_threshold():
    """Sport with healthy ROI + enough scored bets → LIVE. Edge-clearing event
    routes to the RECOMMENDED bucket and carries a $ stake."""
    from datetime import datetime, timezone
    from flashcat.build_site import resolve_sport_modes, _group_by_pick_quality, _event_view
    from flashcat.types import BookLine, Event, SourceProb

    sb = _make_sb({
        "nfl": {"n_events": 540, "sources": {}, "blended": {"n_events": 540, "roi": 0.117, "brier": 0.21, "wagered": 54000, "profit": 6300, "wins": 74, "losses": 51}},
    })
    modes = resolve_sport_modes(sb)
    assert modes["nfl"]["mode"] == "live"
    assert modes["nfl"]["marginal"] is False
    assert "LIVE" in modes["nfl"]["badge_label"]

    now = datetime(2026, 9, 7, 18, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="nfl:1", sport="nfl", home="KC", away="DET", commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.62, captured_at=now)],
        lines=[BookLine(book="dk", side="home", american=-110, captured_at=now),
               BookLine(book="dk", side="away", american=-110, captured_at=now)],
        blended_home_prob=0.62, pick="home", pick_prob=0.62,
    )
    view = _event_view(ev, {}, sport_modes=modes)
    assert view["sport_research_only"] is False

    grouped = _group_by_pick_quality([view], modes, edge_min=0.03)
    assert len(grouped["recommended"]) == 1
    assert len(grouped["research"]) == 0


def test_resolve_sport_modes_live_sport_no_edge():
    """Live sport but event has no edge → NO-EDGE bucket."""
    from datetime import datetime, timezone
    from flashcat.build_site import resolve_sport_modes, _group_by_pick_quality, _event_view
    from flashcat.types import BookLine, Event, SourceProb

    sb = _make_sb({
        "nfl": {"n_events": 540, "sources": {}, "blended": {"n_events": 540, "roi": 0.117, "brier": 0.21, "wins": 74, "losses": 51}},
    })
    modes = resolve_sport_modes(sb)

    now = datetime(2026, 9, 7, 18, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="nfl:2", sport="nfl", home="KC", away="DET", commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.521, captured_at=now)],
        # Devigged market ~0.51 → edge ~0.011 < 0.03 threshold
        lines=[BookLine(book="dk", side="home", american=-105, captured_at=now),
               BookLine(book="dk", side="away", american=-105, captured_at=now)],
        blended_home_prob=0.521, pick="home", pick_prob=0.521,
    )
    view = _event_view(ev, {}, sport_modes=modes)
    grouped = _group_by_pick_quality([view], modes, edge_min=0.03)
    assert len(grouped["recommended"]) == 0
    assert len(grouped["no_edge"]) == 1


def test_resolve_sport_modes_research_sport_with_edge_routes_to_research_bucket():
    """Negative-ROI sport with an edge-clearing event → RESEARCH bucket, NO stake."""
    from datetime import datetime, timezone
    from flashcat.build_site import resolve_sport_modes, _group_by_pick_quality, _event_view
    from flashcat.types import BookLine, Event, SourceProb

    sb = _make_sb({
        "atp": {"n_events": 5155, "sources": {}, "blended": {"n_events": 5155, "roi": -0.068, "brier": 0.217, "wins": 183, "losses": 256}},
    })
    modes = resolve_sport_modes(sb)
    assert modes["atp"]["mode"] == "research"
    assert "below" in modes["atp"]["reason"].lower() or "roi" in modes["atp"]["reason"].lower()

    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="atp:1", sport="atp", home="Sinner", away="Alcaraz", commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.62, captured_at=now)],
        lines=[BookLine(book="dk", side="home", american=-110, captured_at=now),
               BookLine(book="dk", side="away", american=-110, captured_at=now)],
        blended_home_prob=0.62, pick="home", pick_prob=0.62,
    )
    view = _event_view(ev, {}, sport_modes=modes)
    assert view["sport_research_only"] is True
    assert view["recommended_stake"] == 0.0
    assert "Research only" in view["recommended_stake_str"]

    grouped = _group_by_pick_quality([view], modes, edge_min=0.03)
    assert len(grouped["recommended"]) == 0
    assert len(grouped["research"]) == 1


def test_resolve_sport_modes_research_sport_no_edge_goes_to_no_edge():
    """Research sport + no edge → NO-EDGE bucket."""
    from datetime import datetime, timezone
    from flashcat.build_site import resolve_sport_modes, _group_by_pick_quality, _event_view
    from flashcat.types import BookLine, Event, SourceProb

    sb = _make_sb({
        "atp": {"n_events": 5155, "sources": {}, "blended": {"n_events": 5155, "roi": -0.068, "brier": 0.217, "wins": 183, "losses": 256}},
    })
    modes = resolve_sport_modes(sb)

    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="atp:2", sport="atp", home="A", away="B", commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.51, captured_at=now)],
        lines=[BookLine(book="dk", side="home", american=-105, captured_at=now),
               BookLine(book="dk", side="away", american=-105, captured_at=now)],
        blended_home_prob=0.51, pick="home", pick_prob=0.51,
    )
    view = _event_view(ev, {}, sport_modes=modes)
    grouped = _group_by_pick_quality([view], modes, edge_min=0.03)
    assert len(grouped["recommended"]) == 0
    assert len(grouped["research"]) == 0
    assert len(grouped["no_edge"]) == 1


def test_marginal_live_band_flagged():
    """0% < ROI < 2.5% with adequate n_bets → LIVE but marginal."""
    from flashcat.build_site import resolve_sport_modes

    sb = _make_sb({
        "wta": {"n_events": 4639, "sources": {}, "blended": {"n_events": 4639, "roi": 0.016, "brier": 0.219, "wins": 200, "losses": 261}},
    })
    modes = resolve_sport_modes(sb)
    assert modes["wta"]["mode"] == "live"
    assert modes["wta"]["marginal"] is True
    assert "marginal" in modes["wta"]["reason"].lower()


def test_insufficient_n_bets_forces_research():
    """Positive ROI but n_bets < min → RESEARCH on sample-size grounds."""
    from flashcat.build_site import resolve_sport_modes

    sb = _make_sb({
        "mlb": {"n_events": 150, "sources": {}, "blended": {"n_events": 150, "roi": 0.05, "brier": 0.246, "wins": 80, "losses": 70}},
    })
    modes = resolve_sport_modes(sb)
    assert modes["mlb"]["mode"] == "research"
    assert "scored bets" in modes["mlb"]["reason"]


def test_no_roi_forces_research_even_with_large_n_events():
    """MLB has n_events=3488 but wins=0/losses=0 and roi=None (no bets ever
    graded by the connector). That must stay RESEARCH regardless of n_events.
    """
    from flashcat.build_site import resolve_sport_modes

    sb = _make_sb({
        "mlb": {"n_events": 3488, "sources": {}, "blended": {"n_events": 3488, "roi": None, "brier": 0.246, "wins": 0, "losses": 0}},
    })
    modes = resolve_sport_modes(sb)
    assert modes["mlb"]["mode"] == "research"


def test_negative_roi_sport_never_gets_stake_even_when_other_sport_is_live():
    """Phil's gate preserved: ATP -6.8% never shows $ stake, even when NFL +11.7% is LIVE.

    This is the regression target for PR #12. Before: site-wide research mode
    blocked NFL even though it was profitable. After: NFL is LIVE, ATP stays
    RESEARCH, no negative-ROI sport ever shows stake."""
    from datetime import datetime, timezone
    from flashcat.build_site import resolve_sport_modes, _event_view, _group_by_pick_quality
    from flashcat.types import BookLine, Event, SourceProb

    sb = _make_sb({
        "nfl": {"n_events": 540, "sources": {}, "blended": {"n_events": 540, "roi": 0.117, "brier": 0.21, "wins": 74, "losses": 51}},
        "atp": {"n_events": 5155, "sources": {}, "blended": {"n_events": 5155, "roi": -0.068, "brier": 0.217, "wins": 183, "losses": 256}},
    })
    modes = resolve_sport_modes(sb)
    assert modes["nfl"]["mode"] == "live"
    assert modes["atp"]["mode"] == "research"

    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    nfl_ev = Event(
        event_id="nfl:1", sport="nfl", home="KC", away="DET", commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.62, captured_at=now)],
        lines=[BookLine(book="dk", side="home", american=-110, captured_at=now),
               BookLine(book="dk", side="away", american=-110, captured_at=now)],
        blended_home_prob=0.62, pick="home", pick_prob=0.62,
    )
    atp_ev = Event(
        event_id="atp:1", sport="atp", home="Sinner", away="Alcaraz", commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.62, captured_at=now)],
        lines=[BookLine(book="dk", side="home", american=-110, captured_at=now),
               BookLine(book="dk", side="away", american=-110, captured_at=now)],
        blended_home_prob=0.62, pick="home", pick_prob=0.62,
    )
    nfl_view = _event_view(nfl_ev, {}, sport_modes=modes)
    atp_view = _event_view(atp_ev, {}, sport_modes=modes)

    # NFL: LIVE, carries stake
    assert nfl_view["sport_research_only"] is False
    assert nfl_view["recommended_stake"] > 0
    # ATP: RESEARCH, ZERO stake despite same edge
    assert atp_view["sport_research_only"] is True
    assert atp_view["recommended_stake"] == 0.0

    grouped = _group_by_pick_quality([nfl_view, atp_view], modes, edge_min=0.03)
    assert len(grouped["recommended"]) == 1
    assert grouped["recommended"][0]["sport"] == "nfl"
    assert len(grouped["research"]) == 1
    assert grouped["research"][0]["sport"] == "atp"
