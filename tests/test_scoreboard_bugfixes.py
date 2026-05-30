"""Regression tests for the three scoreboard bugs Phil audited on 2026-05-30:

Bug 1 — Pick-side selection had a sign error. The picker chose home whenever
        ``blended_home_prob >= 0.5`` instead of choosing the side that beat
        the devigged market. Result: all 14 Recommended Plays on the live
        site had NEGATIVE edge.

Bug 2 — Per-sport mode badge on individual event cards must come from
        ``resolve_sport_modes()[sport]['mode']``, not from a stale default
        or unrelated flag. ATP was correctly RESEARCH in the summary table
        but ATP cards were rendering 🟢 LIVE badges.

Bug 3 — Headline ROI of ``+26.6%`` was a cross-sport aggregate that did
        not reflect any single sport's reality (NFL +11.7%, WTA +1.6%,
        ATP -6.8%). Replaced with per-sport pills in the header (Option C)
        and the misleading single-number aggregate is removed.

Bug 4 — Recommended/Research routing accepted any ``|edge| >= threshold``,
        which let negative-edge picks through whenever Bug 1 produced one.
        Routing now gates on POSITIVE edge.

Bug 5 — Required regression test: every card in the Recommended Plays
        section must have positive edge AND be on a LIVE-mode sport.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flashcat.build_site import (
    _event_view,
    _group_by_pick_quality,
    build,
    resolve_sport_modes,
)
from flashcat.model.blend import blend_event
from flashcat.model.pick import pick_side
from flashcat.types import BookLine, Event, SourceProb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime(2024, 1, 7, 18, 0, tzinfo=timezone.utc)


def _event_with_market(
    *,
    event_id: str,
    sport: str,
    blended_home_prob: float,
    home_american: int,
    away_american: int,
    home: str = "A",
    away: str = "B",
) -> Event:
    now = _now()
    return Event(
        event_id=event_id,
        sport=sport,
        home=home,
        away=away,
        commence_time=now,
        source_probs=[
            SourceProb(source="src-a", home_win_prob=blended_home_prob, captured_at=now),
        ],
        lines=[
            BookLine(book="dk", side="home", american=home_american, captured_at=now),
            BookLine(book="dk", side="away", american=away_american, captured_at=now),
        ],
    )


def _write_scoreboard(path: Path, per_sport: dict[str, dict]) -> None:
    """Build a scoreboard JSON with the per_sport.blended fields set.

    ``per_sport`` keys map ``sport -> {"roi": float, "n_events": int,
    "wins": int, "losses": int}``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    per_sport_full = {}
    overall_roi = 0.0
    n_total = 0
    for sport, p in per_sport.items():
        n = int(p.get("n_events", 500))
        per_sport_full[sport] = {
            "n_events": n,
            "sources": {},
            "blended": {
                "n_events": n,
                "roi": p["roi"],
                "brier": 0.22,
                "wins": int(p.get("wins", n // 2)),
                "losses": int(p.get("losses", n // 2)),
            },
        }
        overall_roi += p["roi"] * n
        n_total += n
    overall_roi = overall_roi / n_total if n_total else 0.0
    path.write_text(json.dumps({
        "window": {"start": "2024-01-01", "end": "2024-12-31", "sport": "multi"},
        "weights": {},
        "n_events": n_total,
        "sources": {"flashcat-blended": {"n_events": n_total, "roi": overall_roi, "brier": 0.22}},
        "per_sport": per_sport_full,
        "blended_overall": {
            "n_events": n_total, "wagered": 1000 * n_total,
            "profit": int(overall_roi * 1000 * n_total),
            "wins": n_total // 2, "losses": n_total // 2, "roi": overall_roi,
        },
    }))


def _patch_paths(tmp_path, monkeypatch, sb_kind: str = "live"):
    from flashcat import build_site as bs
    from flashcat import config as cfg

    tmp_docs = tmp_path / "docs"
    monkeypatch.setattr(cfg, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(cfg, "ASSETS_DIR", tmp_docs / "assets")
    monkeypatch.setattr(cfg, "EVENT_PAGES_DIR", tmp_docs / "event")
    monkeypatch.setattr(bs, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(bs, "ASSETS_DIR", tmp_docs / "assets")
    monkeypatch.setattr(bs, "EVENT_PAGES_DIR", tmp_docs / "event")

    sb_path = tmp_path / "data" / "source_scoreboard.json"
    if sb_kind == "live":
        per_sport = {
            "nfl": {"roi": 0.05, "n_events": 500},
            "atp": {"roi": -0.07, "n_events": 14000},  # ATP negative → research
            "wta": {"roi": 0.03, "n_events": 12000},  # WTA marginal → live (post-PR-19: floor=2%, ceiling=4%)
        }
    elif sb_kind == "atp_research":
        per_sport = {
            "atp": {"roi": -0.07, "n_events": 14000},
            "nfl": {"roi": 0.05, "n_events": 500},
        }
    else:
        per_sport = {}
    _write_scoreboard(sb_path, per_sport)
    monkeypatch.setattr(cfg, "SOURCE_SCOREBOARD_PATH", sb_path)
    monkeypatch.setattr(bs, "SOURCE_SCOREBOARD_PATH", sb_path)
    return tmp_docs


# ---------------------------------------------------------------------------
# Bug 1 — pick_side aligns with market edge
# ---------------------------------------------------------------------------

def test_bug1_pick_side_picks_higher_edge_side_not_home():
    """When blended_home < devigged_market_home, the picker MUST pick away.

    Live-site repro: blended_home=0.439, market devigged home ≈ 0.477. Pre-fix
    the picker chose home because blended_home >= 0.5 was the only check;
    actually it was below 0.5 here but the OTHER common case (blended_home
    0.56 vs market 0.60) was also broken — pick=home, edge=-4pp. Either way
    the fix is the same: compare against the devigged market.
    """
    ev = _event_with_market(
        event_id="b1:1", sport="wta",
        blended_home_prob=0.56,
        home_american=-150,  # ~60% implied
        away_american=+130,
    )
    side, prob = pick_side(ev, 0.56)
    # Devigged home ≈ 0.585. Blended 0.56 < 0.585 → model's edge is on away.
    assert side == "away", (
        f"pick must reflect model's actual edge over market, got {side}"
    )
    assert prob == pytest.approx(1 - 0.56)


def test_bug1_pick_side_picks_home_when_edge_is_on_home():
    """Symmetric case: blended_home > devigged market home → pick home."""
    ev = _event_with_market(
        event_id="b1:2", sport="nfl",
        blended_home_prob=0.70,
        home_american=-150,
        away_american=+130,
    )
    side, prob = pick_side(ev, 0.70)
    assert side == "home"
    assert prob == pytest.approx(0.70)


def test_bug1_blended_event_records_positive_edge_pick():
    """End-to-end through blend_event: pick + decide_stake must agree.

    If the blender picks the side opposite the market-implied favorite, the
    staking decision's ``edge`` field must be non-negative (zero only when
    the model exactly matches the market).
    """
    from flashcat.model.staking import decide_stake

    ev = _event_with_market(
        event_id="b1:3", sport="wta",
        blended_home_prob=0.44,
        home_american=-110,
        away_american=-110,
    )
    blended = blend_event(ev, weights={})
    dec = decide_stake(blended, blended.pick, blended.pick_prob or 0.5)
    assert dec.edge >= 0.0, (
        f"after the pick-side fix, the picked side must have non-negative "
        f"edge over the devigged market; got {dec.edge:+.4f}"
    )


# ---------------------------------------------------------------------------
# Bug 2 — ATP card never carries a 🟢 LIVE badge when ATP is in RESEARCH
# ---------------------------------------------------------------------------

def test_bug2_atp_research_card_has_no_live_badge(tmp_path, monkeypatch):
    """Synthetic slate: ATP is RESEARCH in the per-sport gate. No ATP card
    rendered on the index page may carry a 🟢 LIVE badge — every ATP card
    must show 🔍 RESEARCH.

    This pins the contract that per-card badges come from the same
    ``resolve_sport_modes()`` resolver as the per-sport summary table.
    """
    tmp_docs = _patch_paths(tmp_path, monkeypatch, sb_kind="atp_research")

    # Sanity: per-sport resolver agrees ATP is research.
    sport_modes = resolve_sport_modes()
    assert sport_modes["atp"]["mode"] == "research", sport_modes["atp"]

    now = _now()
    atp_event = _event_with_market(
        event_id="b2:atp", sport="atp",
        blended_home_prob=0.62,
        home_american=-150, away_american=+130,
        home="Alcaraz", away="Sinner",
    )
    nfl_event = _event_with_market(
        event_id="b2:nfl", sport="nfl",
        blended_home_prob=0.62,
        home_american=-150, away_american=+130,
        home="Chiefs", away="Dolphins",
    )
    build([atp_event, nfl_event])
    html = (tmp_docs / "index.html").read_text()

    # Locate every event card and inspect the badge it carries.
    # The card template renders ``mode-pill {{ sport_mode_cls }}`` next to
    # the sport label inside the card header. We require: any line that
    # names "ATP" as the sport on the card must not also contain "live".
    import re
    # Per-card sport label + pill pattern: SPORT &middot; <time>...</time> <span class="mode-pill X">...</span>
    pattern = re.compile(
        r'>\s*(ATP|NFL|WTA)\s*&middot;.*?mode-pill\s+([a-z-]+)"',
        re.DOTALL,
    )
    matches = pattern.findall(html)
    assert matches, "expected at least one event card with a sport label"
    atp_cards = [cls for sport, cls in matches if sport == "ATP"]
    assert atp_cards, "expected at least one ATP card in the rendered slate"
    for cls in atp_cards:
        assert "live" not in cls, (
            f"ATP card carried a LIVE pill class {cls!r} even though ATP "
            f"is in RESEARCH mode per the per-sport gate"
        )
        assert cls == "research", f"ATP card pill class must be 'research', got {cls!r}"


# ---------------------------------------------------------------------------
# Bug 3 — Headline ROI replaced with per-sport pills (Option C)
# ---------------------------------------------------------------------------

def test_bug3_header_replaces_aggregate_roi_with_per_sport_pills(tmp_path, monkeypatch):
    """No single ``backtest ROI +X%`` aggregate appears in the header.

    Instead, the header renders a ``.site-status-pills`` row carrying one
    pill per sport with that sport's own blended ROI. The cross-sport
    aggregate (+26.6% on the audited live site) is removed because it did
    not reflect per-sport reality.
    """
    tmp_docs = _patch_paths(tmp_path, monkeypatch, sb_kind="live")
    build([])
    html = (tmp_docs / "index.html").read_text()

    # Per-sport pills row is present.
    assert "site-status-pills" in html
    # One pill per sport with its ROI string inline.
    assert "🟢 LIVE NFL" in html
    # ATP and WTA pills are present too (ATP research, WTA live-marginal).
    for sport in ("ATP", "WTA"):
        assert sport in html, f"expected per-sport pill for {sport}"

    # The misleading aggregate header string is gone.
    assert "· backtest ROI" not in html
    assert "&middot; backtest ROI" not in html


# ---------------------------------------------------------------------------
# Bug 4 + Bug 5 — Recommended Plays gating
# ---------------------------------------------------------------------------

def test_bug5_every_recommended_card_has_positive_edge_and_live_sport(tmp_path, monkeypatch):
    """The unit Phil asked for: assert every Recommended card has
    ``edge >= edge_threshold_pp`` AND ``sport_mode == 'live'``.

    Synthetic slate has events spanning positive/negative edge × LIVE/RESEARCH
    sport; only the LIVE+positive subset should land in ``recommended``.
    """
    from flashcat import config as cfg

    tmp_docs = _patch_paths(tmp_path, monkeypatch, sb_kind="live")
    sport_modes = resolve_sport_modes()
    assert sport_modes["nfl"]["mode"] == "live"
    assert sport_modes["atp"]["mode"] == "research"
    # WTA marginal-live counts as live for routing purposes.
    assert sport_modes["wta"]["mode"] == "live"

    # Build a mix:
    #  - NFL positive edge → should land in RECOMMENDED
    #  - NFL negative edge → should land in NO-EDGE (Bug 4 defence)
    #  - ATP positive edge → should land in RESEARCH (sport not live)
    #  - ATP negative edge → should land in NO-EDGE
    #  - WTA positive edge → RECOMMENDED
    events = [
        _event_with_market(
            event_id="bug5:nfl-pos", sport="nfl",
            blended_home_prob=0.65, home_american=-110, away_american=-110,
        ),
        _event_with_market(
            event_id="bug5:nfl-neg", sport="nfl",
            blended_home_prob=0.40, home_american=-150, away_american=+130,
        ),
        _event_with_market(
            event_id="bug5:atp-pos", sport="atp",
            blended_home_prob=0.65, home_american=-110, away_american=-110,
        ),
        _event_with_market(
            event_id="bug5:wta-pos", sport="wta",
            blended_home_prob=0.60, home_american=-110, away_american=-110,
        ),
    ]
    blended_events = [blend_event(e, weights={}) for e in events]
    views = [_event_view(e, weights={}, sport_modes=sport_modes) for e in blended_events]
    edge_min = cfg.EDGE_THRESHOLD_PP / 100.0 if hasattr(cfg, "EDGE_THRESHOLD_PP") else 0.03
    grouped = _group_by_pick_quality(views, sport_modes, edge_min)

    # The contract the audit asked for:
    for v in grouped["recommended"]:
        assert v["edge_value"] is not None, "recommended card without edge"
        assert v["edge_value"] >= edge_min, (
            f"recommended card has edge {v['edge_value']:+.4f} below threshold "
            f"{edge_min:+.4f} (event {v.get('slug')})"
        )
        assert v["sport_mode"] == "live", (
            f"recommended card has non-live sport_mode {v['sport_mode']!r} "
            f"(sport={v.get('sport')!r}, event={v.get('slug')})"
        )

    # And research-bucket cards must also be positive-edge — never negative.
    for v in grouped["research"]:
        assert v["edge_value"] >= edge_min, (
            f"research-bucket card has non-positive edge {v['edge_value']:+.4f}"
        )
        assert v["sport_mode"] != "live"

    # Non-empty buckets to prove the test exercised real data.
    recommended_sports = {v["sport"] for v in grouped["recommended"]}
    assert "nfl" in recommended_sports or "wta" in recommended_sports, (
        f"expected at least one LIVE-sport positive-edge card in recommended, "
        f"got {recommended_sports}"
    )


def test_bug4_negative_edge_never_routes_to_recommended():
    """Defence in depth: even if a future regression in pick_side puts a
    negative edge on a card, the bucketer must keep it out of Recommended/
    Research and route it to No-Edge.
    """
    sport_modes = {
        "nfl": {"mode": "live", "marginal": False, "badge_label": "🟢 LIVE", "badge_cls": "live"},
    }
    views = [
        {
            "edge_value": -0.05,    # negative, magnitude well above threshold
            "pick_label": "CHIEFS",
            "sport": "nfl",
            "sport_mode": "live",
            "recommended_stake": 0.0,
            "commence_iso": "2024-01-07T18:00:00Z",
        },
    ]
    grouped = _group_by_pick_quality(views, sport_modes, edge_min=0.03)
    assert grouped["recommended"] == []
    assert grouped["research"] == []
    assert len(grouped["no_edge"]) == 1
