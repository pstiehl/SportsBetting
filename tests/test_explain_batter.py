"""Per-batter MLB rationale tests.

Covers PR #16 work items 2-4 and 6:
  - Specific batter mismatches surface with name + xwOBA + handedness.
  - Top-3 deviation gating works (no fabricated specificity).
  - Falls back to generic "Statcast lineup edge" string when data is missing
    or no batter clears the deviation threshold.
  - DB load fallback works when SourceProb.metadata is absent.
  - Regression: per-sport LIVE/RESEARCH gate is unaffected by the new path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from flashcat.explain import (
    BATTER_RATIONALE_DEVIATION_THRESHOLD,
    _statcast_lines,
    _ordinal,
    explain_event,
)
from flashcat.types import BookLine, Event, SourceProb
from flashcat.sources.mlb_statcast_lineup import (
    LEAGUE_AVG_XWOBA,
    load_lineup_contributions,
    persist_lineup_contributions,
)


def _ts():
    return datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)


def _contrib(
    *,
    name: str,
    pos: int,
    xwoba: float,
    team: str = "New York Yankees",
    team_side: str = "home",
    vs_hand: str = "L",
    batter_stand: str = "R",
    observed: bool = True,
    pa_weight: float | None = None,
) -> dict:
    if pa_weight is None:
        # Match the connector's PA-weight schedule (PA_WEIGHTS_9 normalized).
        weights = [0.135, 0.126, 0.118, 0.112, 0.106, 0.101, 0.096, 0.091, 0.086]
        total = sum(weights)
        pa_weight = weights[pos - 1] / total if 1 <= pos <= 9 else 1.0 / 9
    return {
        "batter_id": 600000 + pos,
        "batter_name": name,
        "batting_order_position": pos,
        "team": team,
        "team_side": team_side,
        "batter_stand": batter_stand,
        "vs_pitcher_hand": vs_hand,
        "xwoba_vs_handedness": xwoba,
        "league_avg_xwoba": LEAGUE_AVG_XWOBA,
        "pa_weight": pa_weight,
        "opp_xwoba_allowed": LEAGUE_AVG_XWOBA,
        "contribution_to_team_score": pa_weight * xwoba * LEAGUE_AVG_XWOBA,
        "xwoba_observed": observed,
    }


def _mk_event(contributions: list[dict] | None) -> Event:
    metadata = None
    if contributions is not None:
        metadata = {
            "home_off": 0.110,
            "away_off": 0.090,
            "diff": 0.020,
            "lineup_contributions": contributions,
        }
    return Event(
        event_id="mlb:test-batter:1",
        sport="mlb",
        league="MLB",
        home="New York Yankees",
        away="Boston Red Sox",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-statcast-lineup",
                home_win_prob=0.58,
                captured_at=_ts(),
                notes="home_off=0.1100 away_off=0.0900 diff=+0.0200",
                metadata=metadata,
            ),
        ],
        blended_home_prob=0.58,
        pick="home",
        pick_prob=0.58,
    )


# -- Work item 2: specific matchup-driving batters surface ------------------


def test_specific_batter_surfaces_with_name_and_xwoba():
    contribs = [
        _contrib(name="Aaron Judge", pos=1, xwoba=0.412),  # +0.097 vs league
        _contrib(name="Juan Soto", pos=2, xwoba=0.390),    # +0.075
        _contrib(name="Anthony Volpe", pos=9, xwoba=0.290),  # -0.025 (under threshold)
    ]
    ev = _mk_event(contribs)
    lines = _statcast_lines(ev)
    assert lines, "expected at least one batter rationale line"
    text = "\n".join(lines)
    assert "Aaron Judge" in text
    assert "Juan Soto" in text
    # Under-threshold batter should NOT be surfaced.
    assert "Volpe" not in text
    # Specific xwOBA numbers should appear (3 decimal precision).
    assert "0.412" in text or "0.41" in text
    # Handedness of opposing pitcher should be named.
    assert "LHP" in text


def test_top_3_cap_on_batter_lines():
    contribs = [
        _contrib(name=f"Slugger{i}", pos=i, xwoba=0.450) for i in range(1, 7)
    ]
    ev = _mk_event(contribs)
    lines = _statcast_lines(ev)
    # Top-3 cap baked into _batter_lines.
    assert len(lines) == 3
    # All listed players had identical xwOBA so just confirm 3 distinct names.
    names = {ln.split("Statcast: ")[1].split(" (")[0] for ln in lines if "Statcast: " in ln}
    assert len(names) == 3


def test_ordinal_helper():
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(21) == "21st"


# -- Work item 4: threshold + graceful fallback -----------------------------


def test_fallback_when_no_batter_clears_threshold():
    # Every batter sits within +/-0.020 of league mean — under the 0.030 gate.
    contribs = [
        _contrib(name=f"Average{i}", pos=i, xwoba=LEAGUE_AVG_XWOBA + 0.015 * (1 if i % 2 else -1))
        for i in range(1, 10)
    ]
    ev = _mk_event(contribs)
    lines = _statcast_lines(ev)
    assert len(lines) == 1
    # Generic team-level fallback should fire when no batter clears threshold.
    assert "Statcast lineup edge" in lines[0]
    assert "Average1" not in lines[0]


def test_fallback_when_metadata_absent():
    # No metadata at all on the SourceProb — the explainer must not crash
    # and must surface the existing generic team-level rationale.
    ev = _mk_event(None)
    lines = _statcast_lines(ev)
    assert len(lines) == 1
    assert "Statcast lineup edge" in lines[0]


def test_fallback_when_metadata_is_empty_list():
    ev = _mk_event([])
    lines = _statcast_lines(ev)
    assert len(lines) == 1
    assert "Statcast lineup edge" in lines[0]


def test_unobserved_xwoba_batters_skipped():
    # An "observed=False" row means the connector filled in league average.
    # We must not name that player as a "matchup-driving" batter.
    contribs = [
        _contrib(name="Aaron Judge", pos=1, xwoba=0.412, observed=True),
        _contrib(name="Mystery Rookie", pos=2, xwoba=0.500, observed=False),
    ]
    ev = _mk_event(contribs)
    lines = _statcast_lines(ev)
    text = "\n".join(lines)
    assert "Aaron Judge" in text
    assert "Mystery Rookie" not in text


def test_threshold_exact_boundary():
    # Below the threshold: skipped. At/above: included.
    just_under = LEAGUE_AVG_XWOBA + (BATTER_RATIONALE_DEVIATION_THRESHOLD - 0.001)
    just_over = LEAGUE_AVG_XWOBA + (BATTER_RATIONALE_DEVIATION_THRESHOLD + 0.001)
    contribs = [
        _contrib(name="JustUnder", pos=1, xwoba=just_under),
        _contrib(name="JustOver", pos=2, xwoba=just_over),
    ]
    ev = _mk_event(contribs)
    lines = _statcast_lines(ev)
    text = "\n".join(lines)
    assert "JustOver" in text
    assert "JustUnder" not in text


# -- Work item 3: DB-fallback path when metadata is missing -----------------


def test_explainer_loads_contributions_from_db_when_metadata_missing(
    tmp_path, monkeypatch
):
    """When SourceProb.metadata is None the explainer falls back to the\n    persisted ``mlb_lineup_contributions`` table in source_history.db.\n    """
    db_path = tmp_path / "source_history.db"
    contribs = [
        _contrib(name="Aaron Judge", pos=1, xwoba=0.412),
        _contrib(name="Juan Soto", pos=2, xwoba=0.395),
    ]
    event_id = "mlb-statcast-lineup:777999"
    n = persist_lineup_contributions(
        event_id=event_id,
        commence_time=_ts(),
        contributions=contribs,
        db_path=db_path,
    )
    assert n == 2
    rows = load_lineup_contributions(event_id, db_path=db_path)
    assert len(rows) == 2
    # Now patch the constant used by load_lineup_contributions when called
    # from the explainer (it consults SOURCE_HISTORY_DB_PATH by default).
    import flashcat.sources.mlb_statcast_lineup as scl

    monkeypatch.setattr(scl, "SOURCE_HISTORY_DB_PATH", db_path)

    ev = Event(
        event_id=event_id,
        sport="mlb",
        league="MLB",
        home="New York Yankees",
        away="Boston Red Sox",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-statcast-lineup",
                home_win_prob=0.58,
                captured_at=_ts(),
                notes="home_off=0.1100 away_off=0.0900 diff=+0.0200",
                metadata=None,
            ),
        ],
        blended_home_prob=0.58,
        pick="home",
        pick_prob=0.58,
    )
    lines = _statcast_lines(ev)
    text = "\n".join(lines)
    assert "Aaron Judge" in text
    assert "Juan Soto" in text


def test_load_returns_empty_when_db_path_missing(tmp_path):
    rows = load_lineup_contributions(
        "nonexistent:1", db_path=tmp_path / "absent.db"
    )
    assert rows == []


def test_load_returns_empty_when_table_missing(tmp_path):
    import sqlite3

    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()
    rows = load_lineup_contributions("any:1", db_path=db_path)
    assert rows == []


# -- Work item 6: explain_event integrates batter lines first ---------------


def test_explain_event_puts_batter_lines_before_weather():
    contribs = [
        _contrib(name="Aaron Judge", pos=1, xwoba=0.412),
        _contrib(name="Juan Soto", pos=2, xwoba=0.395),
    ]
    ev = Event(
        event_id="mlb:test-batter:2",
        sport="mlb",
        league="MLB",
        home="New York Yankees",
        away="Boston Red Sox",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-statcast-lineup",
                home_win_prob=0.58,
                captured_at=_ts(),
                notes="home_off=0.1100 away_off=0.0900 diff=+0.0200",
                metadata={
                    "home_off": 0.11,
                    "away_off": 0.09,
                    "diff": 0.02,
                    "lineup_contributions": contribs,
                },
            ),
            SourceProb(
                source="mlb-weather",
                home_win_prob=0.54,
                captured_at=_ts(),
                notes="park=Yankee Stadium dome=False runs_h=5.20 runs_a=5.20 temp=82F wind=12mph dir=0deg",
            ),
        ],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=_ts()),
            BookLine(book="dk", side="away", american=-110, captured_at=_ts()),
        ],
        blended_home_prob=0.58,
        pick="home",
        pick_prob=0.58,
    )
    out = explain_event(ev, top_n=3)
    assert len(out) == 3
    assert "Aaron Judge" in out[0]
    # The third slot was either second batter or the weather line — both
    # acceptable. The top slot is the highest-deviation batter.
    assert any("Weather" in s or "Soto" in s for s in out[1:])


def test_non_mlb_event_unaffected():
    # NFL events should not pick up any MLB-specific path.
    ev = Event(
        event_id="nfl:test:1",
        sport="nfl",
        league="NFL",
        home="Kansas City Chiefs",
        away="Buffalo Bills",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="nfl-nflfastr-epa",
                home_win_prob=0.62,
                captured_at=_ts(),
                notes="pred_diff=4.5 h_off=0.150 a_off=0.100",
            ),
        ],
        blended_home_prob=0.62,
        pick="home",
        pick_prob=0.62,
    )
    out = explain_event(ev)
    assert any("EPA edge" in s for s in out)
    # No Statcast line should ever leak into NFL.
    assert not any("Statcast" in s for s in out)
