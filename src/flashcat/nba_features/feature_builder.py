"""Walk-forward feature builder for NBA games.

Pure functions. Every feature passed to the model is computable using
ONLY data strictly BEFORE the game's date. The leakage gate is asserted
in ``build_rolling_signals``.

Data source: ``data/source_history.db.predictions`` (committed). Three
NBA prior sources are present and graded:

  fivethirtyeight-nba-raptor       2022-01-01 .. 2023-06-12
  fivethirtyeight-nba-elo-modern   2022-01-01 .. 2023-06-12
  nba-bref-srs-pace                2021-10-22 .. 2024-04-14

The three sources never share ``event_id`` (different scrapers), so we
join on the natural key ``(commence_date, home, away)`` with team-code
normalization (BRK/BKN, CHO/CHA, PHO/PHX).
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team-code normalization
# ---------------------------------------------------------------------------

# 538 uses BRK/CHO/PHO; basketball-reference uses BKN/CHA/PHX. Canonical
# = the 538 form because that's what the FlashCat NBA connector emits.
TEAM_ALIASES = {
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO",
}


def normalize_team(team: str) -> str:
    """Map a 3-letter team code to the canonical FlashCat form."""
    if not team:
        return team
    t = team.strip().upper()
    return TEAM_ALIASES.get(t, t)


# ---------------------------------------------------------------------------
# GameRow — one record per game after joining the priors
# ---------------------------------------------------------------------------


@dataclass
class GameRow:
    """One row per scheduled NBA game, after merging prior sources."""

    game_date: date
    season: int
    home: str
    away: str
    home_won: Optional[int]  # 0/1; None if not graded
    # Per-source priors (None if source didn't cover this game).
    raptor_prob_home: Optional[float] = None
    elo_modern_prob_home: Optional[float] = None
    bref_srs_prob_home: Optional[float] = None

    @property
    def priors_available(self) -> int:
        return sum(
            1
            for v in (
                self.raptor_prob_home,
                self.elo_modern_prob_home,
                self.bref_srs_prob_home,
            )
            if v is not None
        )


# ---------------------------------------------------------------------------
# Feature catalog
# ---------------------------------------------------------------------------

# Order matters — this is the column order the model fits and the
# coefficient export relies on.
FEATURE_NAMES = [
    "raptor_prob_home",
    "elo_modern_prob_home",
    "bref_srs_prob_home",
    "prior_consensus",
    "prior_dispersion",
    "raptor_vs_elo_disagree",
    "win_pct_l5_home",
    "win_pct_l5_away",
    "win_pct_l10_home",
    "win_pct_l10_away",
    "win_pct_diff_l10",
    "avg_margin_l10_home",
    "avg_margin_l10_away",
    "days_rest_home",
    "days_rest_away",
    "days_rest_diff",
    "b2b_home",
    "b2b_away",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _season_of(d: date) -> int:
    """NBA season is named for the year of the FINALS.

    Oct 2021 → 2022 season; Apr 2022 → 2022 season; Oct 2022 → 2023; etc.
    """
    return d.year if d.month <= 7 else d.year + 1


def load_nba_games_from_history(
    db_path: Path,
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> list[GameRow]:
    """Load NBA games from ``source_history.db`` and join the three priors.

    Returns one ``GameRow`` per unique ``(game_date, home, away)`` triple
    in the date window. ``home_won`` is taken from the first source that
    has it populated; we cross-check that all sources agree and log a
    warning on disagreement.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"source_history.db not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Pull every NBA prior row in the window.
    where = ["sport = 'nba'", "home_prob IS NOT NULL"]
    params: list = []
    if start is not None:
        where.append("commence_time >= ?")
        params.append(start.isoformat() + "T00:00:00+00:00")
    if end is not None:
        where.append("commence_time <= ?")
        params.append(end.isoformat() + "T23:59:59+00:00")
    sql = (
        "SELECT event_id, source, commence_time, home, away, home_prob, home_won "
        "FROM predictions WHERE " + " AND ".join(where) + " ORDER BY commence_time"
    )
    cur.execute(sql, params)

    # Pivot by (date, normalized_home, normalized_away).
    bucket: dict[tuple[date, str, str], dict] = {}
    for row in cur.fetchall():
        ct = row["commence_time"]
        try:
            gdate = datetime.fromisoformat(ct.replace("Z", "+00:00")).date()
        except Exception:
            continue
        h = normalize_team(row["home"]) if row["home"] else None
        a = normalize_team(row["away"]) if row["away"] else None
        if not h or not a:
            continue
        key = (gdate, h, a)
        entry = bucket.setdefault(
            key,
            {
                "game_date": gdate,
                "home": h,
                "away": a,
                "home_won": None,
                "raptor": None,
                "elo": None,
                "bref": None,
                "_won_sources": [],
            },
        )
        if row["home_won"] is not None:
            entry["_won_sources"].append((row["source"], int(row["home_won"])))
        p = float(row["home_prob"])
        if row["source"] == "fivethirtyeight-nba-raptor":
            entry["raptor"] = p
        elif row["source"] == "fivethirtyeight-nba-elo-modern":
            entry["elo"] = p
        elif row["source"] == "nba-bref-srs-pace":
            entry["bref"] = p

    conn.close()

    # Resolve home_won (require unanimity when multiple sources report).
    out: list[GameRow] = []
    disagreements = 0
    for key, entry in bucket.items():
        won_votes = {v for _, v in entry["_won_sources"]}
        if len(won_votes) == 1:
            entry["home_won"] = next(iter(won_votes))
        elif len(won_votes) > 1:
            disagreements += 1
            continue  # Drop ambiguous outcomes from training.
        gdate = entry["game_date"]
        out.append(
            GameRow(
                game_date=gdate,
                season=_season_of(gdate),
                home=entry["home"],
                away=entry["away"],
                home_won=entry["home_won"],
                raptor_prob_home=entry["raptor"],
                elo_modern_prob_home=entry["elo"],
                bref_srs_prob_home=entry["bref"],
            )
        )
    if disagreements:
        log.warning("dropped %d NBA games with source-disagreed outcomes", disagreements)

    out.sort(key=lambda g: (g.game_date, g.home, g.away))
    log.info(
        "loaded %d NBA games from %s; %d have all 3 priors, %d have >=1 prior",
        len(out),
        db_path,
        sum(1 for g in out if g.priors_available == 3),
        sum(1 for g in out if g.priors_available >= 1),
    )
    return out


# ---------------------------------------------------------------------------
# Rolling-rate builder (pure-functional, leakage-asserted)
# ---------------------------------------------------------------------------


@dataclass
class RollingTeamState:
    """Per-team running history used by the rolling-feature builder."""

    results: deque = field(default_factory=lambda: deque(maxlen=40))
    # Each entry is (game_date, win_flag (0/1), signed_margin_proxy)
    last_game_date: Optional[date] = None


def _signed_margin_proxy(
    game: GameRow, *, team_is_home: bool, home_won_flag: int
) -> float:
    """Return a signed margin proxy in (-1, +1) for a graded game.

    We don't have NBA box scores in source_history.db, so we proxy each
    game's margin by ``consensus_prior - 0.5``. If the team won we
    sign-correct: a heavy favorite winning is a small +; a heavy
    underdog winning is a big +.

    This is a coarse rolling-form proxy. The Phase-2 work brings real
    box-score margins.
    """
    priors = [
        game.raptor_prob_home,
        game.elo_modern_prob_home,
        game.bref_srs_prob_home,
    ]
    available = [p for p in priors if p is not None]
    if not available:
        return 0.0
    consensus = sum(available) / len(available)
    # Margin for HOME team: positive if home won easier-than-expected.
    home_margin = (1.0 - consensus) if home_won_flag == 1 else -(consensus)
    if team_is_home:
        return home_margin
    return -home_margin


def build_rolling_signals(
    games: list[GameRow],
) -> dict[int, dict[str, float]]:
    """Walk every game in chronological order; build per-game rolling features.

    Returns ``{id(game): {feature_name: value}}``. Leakage gate: the
    rolling state for a game is built from STRICTLY-EARLIER games only.

    Features produced per game:
      win_pct_l5_home, win_pct_l5_away
      win_pct_l10_home, win_pct_l10_away
      win_pct_diff_l10
      avg_margin_l10_home, avg_margin_l10_away
      days_rest_home, days_rest_away, days_rest_diff
      b2b_home, b2b_away
    """
    games_sorted = sorted(games, key=lambda g: (g.game_date, g.home, g.away))
    state: dict[str, RollingTeamState] = defaultdict(RollingTeamState)
    out: dict[int, dict[str, float]] = {}

    for g in games_sorted:
        home_state = state[g.home]
        away_state = state[g.away]

        # Compute features from CURRENT state (which only contains games
        # strictly before this one — assert it).
        for s, who in ((home_state, "home"), (away_state, "away")):
            if s.last_game_date is not None:
                assert s.last_game_date < g.game_date, (
                    f"leakage: {who} state for {g.home}-{g.away} on {g.game_date} "
                    f"already contains game on {s.last_game_date}"
                )

        def win_pct(s: RollingTeamState, n: int) -> float:
            if not s.results:
                return 0.5
            tail = list(s.results)[-n:]
            return sum(w for _, w, _ in tail) / len(tail)

        def avg_margin(s: RollingTeamState, n: int) -> float:
            if not s.results:
                return 0.0
            tail = list(s.results)[-n:]
            return sum(m for _, _, m in tail) / len(tail)

        feats: dict[str, float] = {}
        feats["win_pct_l5_home"] = win_pct(home_state, 5)
        feats["win_pct_l5_away"] = win_pct(away_state, 5)
        feats["win_pct_l10_home"] = win_pct(home_state, 10)
        feats["win_pct_l10_away"] = win_pct(away_state, 10)
        feats["win_pct_diff_l10"] = (
            feats["win_pct_l10_home"] - feats["win_pct_l10_away"]
        )
        feats["avg_margin_l10_home"] = avg_margin(home_state, 10)
        feats["avg_margin_l10_away"] = avg_margin(away_state, 10)

        # Rest features.
        def rest(s: RollingTeamState) -> float:
            if s.last_game_date is None:
                return 3.0  # season-opener proxy
            delta = (g.game_date - s.last_game_date).days
            return float(max(0, min(delta, 14)))  # cap at 14 days

        feats["days_rest_home"] = rest(home_state)
        feats["days_rest_away"] = rest(away_state)
        feats["days_rest_diff"] = (
            feats["days_rest_home"] - feats["days_rest_away"]
        )
        feats["b2b_home"] = 1.0 if feats["days_rest_home"] <= 1.0 else 0.0
        feats["b2b_away"] = 1.0 if feats["days_rest_away"] <= 1.0 else 0.0

        out[id(g)] = feats

        # Now (AFTER reading state) update state with this game's outcome.
        if g.home_won is not None:
            home_margin = _signed_margin_proxy(g, team_is_home=True, home_won_flag=g.home_won)
            away_margin = _signed_margin_proxy(g, team_is_home=False, home_won_flag=g.home_won)
            home_state.results.append((g.game_date, int(g.home_won == 1), home_margin))
            away_state.results.append((g.game_date, int(g.home_won == 0), away_margin))
        home_state.last_game_date = g.game_date
        away_state.last_game_date = g.game_date

    return out


# ---------------------------------------------------------------------------
# Per-game feature assembly
# ---------------------------------------------------------------------------


def build_features(
    game: GameRow,
    rolling_signals: dict[int, dict[str, float]],
) -> Optional[dict[str, float]]:
    """Assemble the full feature dict for one game. Returns None if it
    can't produce any prior signal at all (model has nothing to fit).
    """
    if game.priors_available == 0:
        return None

    priors = [
        game.raptor_prob_home,
        game.elo_modern_prob_home,
        game.bref_srs_prob_home,
    ]
    available = [p for p in priors if p is not None]
    consensus = sum(available) / len(available)
    dispersion = max(available) - min(available) if len(available) > 1 else 0.0

    raptor = game.raptor_prob_home
    elo = game.elo_modern_prob_home
    raptor_vs_elo = (
        (raptor - elo) if (raptor is not None and elo is not None) else 0.0
    )

    feats: dict[str, float] = {
        "raptor_prob_home": raptor if raptor is not None else consensus,
        "elo_modern_prob_home": elo if elo is not None else consensus,
        "bref_srs_prob_home": (
            game.bref_srs_prob_home
            if game.bref_srs_prob_home is not None
            else consensus
        ),
        "prior_consensus": consensus,
        "prior_dispersion": dispersion,
        "raptor_vs_elo_disagree": raptor_vs_elo,
    }
    rolling = rolling_signals.get(id(game), {})
    for k in (
        "win_pct_l5_home",
        "win_pct_l5_away",
        "win_pct_l10_home",
        "win_pct_l10_away",
        "win_pct_diff_l10",
        "avg_margin_l10_home",
        "avg_margin_l10_away",
        "days_rest_home",
        "days_rest_away",
        "days_rest_diff",
        "b2b_home",
        "b2b_away",
    ):
        feats[k] = float(rolling.get(k, 0.0))
    return feats


def feature_vector(feats: dict[str, float], *, fill_value: float = 0.0) -> list[float]:
    return [float(feats.get(name, fill_value)) for name in FEATURE_NAMES]
