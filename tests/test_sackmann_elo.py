"""Sackmann surface-Elo unit tests using a tiny CSV fixture."""

from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path

import pytest

from flashcat.sources import sackmann_elo as se


FIXTURE_HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,loser_name,"
    "loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,minutes,"
    "w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,w_bpFaced,"
    "l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,l_bpFaced,"
    "winner_rank,winner_rank_points,loser_rank,loser_rank_points\n"
)


def _row(date_str: str, surface: str, winner: str, loser: str, match_num: int = 1) -> str:
    """Build a minimal CSV row (most fields blank — Sackmann allows it)."""
    return (
        f"x-{match_num},Test,{surface},32,A,{date_str},{match_num},"
        f"100,,,{ winner },R,,USA,25,200,,,{loser},R,,USA,25,6-0 6-0,3,F,90,"
        ",,,,,,,,,,,,,,,,,,,,,1,2000,2,1500\n"
    )


@pytest.fixture
def fake_atp_csv(tmp_path, monkeypatch):
    """Patch the per-year download to read from a temp file."""
    csv_content = (
        FIXTURE_HEADER
        + _row("20220110", "Hard", "Player A", "Player B", 1)
        + _row("20220115", "Hard", "Player A", "Player C", 2)
        + _row("20220120", "Clay", "Player B", "Player A", 3)  # B beats A on clay
        + _row("20220125", "Clay", "Player B", "Player C", 4)
        + _row("20220130", "Clay", "Player A", "Player B", 5)  # rematch on clay
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "sackmann_atp_2022.csv").write_text(csv_content)
    # Empty file for years we don't care about
    monkeypatch.setattr(se, "CACHE_DIR", cache_dir)
    return cache_dir


def test_sackmann_walk_forward_atp(fake_atp_csv):
    """Walk-forward Elo: B's clay rating should rise after each clay win."""
    eng = se._SackmannElo("atp", [2022])
    preds = eng.predictions(date(2022, 1, 1), date(2022, 12, 31))
    assert len(preds) == 5
    # match 1 (Hard): first time both seen → ~50/50
    _, _, _, home1, away1, p1, _, _ = preds[0]
    assert abs(p1 - 0.5) < 0.05
    # match 5 (Clay, A vs B): B has won 2 clay games, A has lost 1 → B favored.
    last = preds[4]
    home, away = last[3], last[4]
    home_prob = last[5]
    # The "home" team is the alphabetically-first one; B is favored, so if B is
    # home the prob should be > 0.5, else < 0.5.
    if home == "Player B":
        assert home_prob > 0.5
    else:
        assert home_prob < 0.5


def test_sackmann_emit_events(fake_atp_csv):
    """fetch_events returns Events with one Sackmann SourceProb each."""
    src = se.SackmannATPElo()
    events = src.fetch_events(date(2022, 1, 1), date(2022, 12, 31))
    assert len(events) == 5
    assert all(e.sport == "atp" for e in events)
    assert all(any(sp.source == "sackmann-atp-elo" for sp in e.source_probs) for e in events)


def test_sackmann_load_results_aligned(fake_atp_csv):
    src = se.SackmannATPElo()
    events = src.fetch_events(date(2022, 1, 1), date(2022, 12, 31))
    results = src.load_results(date(2022, 1, 1), date(2022, 12, 31))
    assert {e.event_id for e in events} == {r.event_id for r in results}
