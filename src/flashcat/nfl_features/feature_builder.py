"""Walk-forward feature builder for NFL games — Phase 1.

Pure functions only. Every feature is computable using ONLY data strictly
BEFORE the game's date. The leakage gate is asserted on every call.

Two data sources:
  * ``nfl_data_py.import_schedules`` — one row per game with the closing
    home/away moneyline, spread, total, rest days, and final score.
  * ``nfl_data_py.import_pbp_data``  — every regular-season play.
    Aggregated into per-team per-game off/def EPA-per-play, success
    rate, and pass/rush EPA splits.

Priors:
  * 538 NFL Elo  — already persisted into ``data/source_history.db`` by
    ``scripts/backfill_nfl_historical.py``. We re-read from the DB so we
    inherit the existing leakage gate.
  * nflfastR-EPA — same: already persisted via the existing backfill.
  * Market moneyline — devigged from the closing moneyline column.

Required-feature gate:
  At minimum the market prob must be present (every game in the schedule
  has it). The Elo and EPA priors are present for 2022+; the rolling EPA
  features need at least 4 prior games (~5 weeks). Games that fail any
  required feature are excluded from the backtest — they show up in
  ``n_games_loaded`` minus ``n_games_with_features`` in the report.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Divisional matchups — stable since 2002 realignment.
NFC_EAST = {"DAL", "NYG", "PHI", "WAS"}
NFC_NORTH = {"CHI", "DET", "GB", "MIN"}
NFC_SOUTH = {"ATL", "CAR", "NO", "TB"}
NFC_WEST = {"ARI", "LA", "LAR", "SF", "SEA"}
AFC_EAST = {"BUF", "MIA", "NE", "NYJ"}
AFC_NORTH = {"BAL", "CIN", "CLE", "PIT"}
AFC_SOUTH = {"HOU", "IND", "JAX", "TEN"}
AFC_WEST = {"DEN", "KC", "LAC", "LV", "OAK"}

DIVISIONS = [
    NFC_EAST, NFC_NORTH, NFC_SOUTH, NFC_WEST,
    AFC_EAST, AFC_NORTH, AFC_SOUTH, AFC_WEST,
]


def _same_division(a: str, b: str) -> bool:
    a = (a or "").upper()
    b = (b or "").upper()
    for div in DIVISIONS:
        if a in div and b in div:
            return True
    return False


# ---------------------------------------------------------------------------
# GameRow — canonical normalized record consumed by the feature builder
# ---------------------------------------------------------------------------


@dataclass
class GameRow:
    """One row per NFL regular-season game, after merging sources."""

    game_id: str
    game_date: date
    season: int
    week: int
    home: str
    away: str
    home_score: Optional[int]
    away_score: Optional[int]
    # Closing moneyline (American). When missing, we can't devig the market.
    home_moneyline: Optional[float] = None
    away_moneyline: Optional[float] = None
    spread_line: Optional[float] = None
    # Rest days (nflverse computes; 7 is the modal value).
    home_rest: Optional[int] = None
    away_rest: Optional[int] = None
    # Priors (loaded from source_history.db by backfill_nfl_historical).
    elo_prob_home: Optional[float] = None
    qbelo_prob_home: Optional[float] = None
    epa_prob_home: Optional[float] = None

    @property
    def home_won(self) -> Optional[bool]:
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score == self.away_score:
            return None  # ties; drop
        return self.home_score > self.away_score


# ---------------------------------------------------------------------------
# Schedule loader
# ---------------------------------------------------------------------------

def _normalize_team(code: str) -> str:
    """nflverse uses post-2020 codes. Map a few historical shifts so divisional
    detection and rolling state stay coherent.
    """
    c = (code or "").upper()
    # Oakland → Las Vegas
    if c == "OAK":
        return "LV"
    # St Louis Rams → LA Rams (Rams in nflverse are "LA")
    if c in ("STL",):
        return "LA"
    # San Diego Chargers → LA Chargers
    if c == "SD":
        return "LAC"
    # nflverse uses "LA" for Rams in modern; some old code uses "LAR"
    if c == "LAR":
        return "LA"
    return c


def load_games_from_schedules(seasons: list[int]) -> list[GameRow]:
    """Pull the nflverse schedule and convert to ``GameRow``s.

    Regular-season only (``game_type == 'REG'``).
    """
    try:
        import nfl_data_py as nfl  # type: ignore
    except ImportError:
        log.error("nfl_data_py not installed")
        return []
    sched = nfl.import_schedules(seasons)
    sched = sched[sched["game_type"] == "REG"].copy()
    out: list[GameRow] = []
    for _, row in sched.iterrows():
        gameday = row.get("gameday")
        if gameday is None or (isinstance(gameday, float) and math.isnan(gameday)):
            continue
        try:
            d = datetime.strptime(str(gameday), "%Y-%m-%d").date()
        except ValueError:
            continue
        home = _normalize_team(row.get("home_team"))
        away = _normalize_team(row.get("away_team"))
        if not home or not away:
            continue
        out.append(GameRow(
            game_id=str(row.get("game_id")),
            game_date=d,
            season=int(row.get("season")),
            week=int(row.get("week") or 0),
            home=home,
            away=away,
            home_score=_safe_int(row.get("home_score")),
            away_score=_safe_int(row.get("away_score")),
            home_moneyline=_safe_float(row.get("home_moneyline")),
            away_moneyline=_safe_float(row.get("away_moneyline")),
            spread_line=_safe_float(row.get("spread_line")),
            home_rest=_safe_int(row.get("home_rest")),
            away_rest=_safe_int(row.get("away_rest")),
        ))
    return out


def _safe_int(v) -> Optional[int]:
    try:
        if v is None: return None
        if isinstance(v, float) and math.isnan(v): return None
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None: return None
        f = float(v)
        if math.isnan(f): return None
        return f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Prior loader — re-uses backfill_nfl_historical's source_history.db rows
# ---------------------------------------------------------------------------


def attach_priors_from_db(games: list[GameRow], db_path: Path) -> None:
    """Mutates ``games`` in place, populating elo_prob_home / epa_prob_home /
    qbelo_prob_home from the source_history.db predictions table.

    The backfill keys rows by ``nfl:YYYY-MM-DD:AWAY@HOME`` (built by
    ``scripts/backfill_nfl_historical.py``). We index games by the same
    format so the join hits.
    """
    if not db_path.exists():
        log.warning("source_history.db not found at %s", db_path)
        return

    def _key(g: GameRow) -> str:
        return f"nfl:{g.game_date.isoformat()}:{g.away}@{g.home}"

    by_id = {_key(g): g for g in games}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT event_id, source, home_prob FROM predictions WHERE source IN "
            "('fivethirtyeight-nfl-elo','fivethirtyeight-nfl-qbelo','nfl-nflfastr-epa','market-close','market-consensus')"
        )
        hit = 0
        for event_id, source, home_prob in cur:
            g = by_id.get(str(event_id))
            if g is None or home_prob is None:
                continue
            hit += 1
            if source == "fivethirtyeight-nfl-elo":
                g.elo_prob_home = float(home_prob)
            elif source == "fivethirtyeight-nfl-qbelo":
                g.qbelo_prob_home = float(home_prob)
            elif source == "nfl-nflfastr-epa":
                g.epa_prob_home = float(home_prob)
        log.info("attached %d prior rows from source_history.db", hit)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Market devigging
# ---------------------------------------------------------------------------


def _moneyline_to_prob(ml: float) -> float:
    """American moneyline → implied probability (NO devig — just raw implied)."""
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return -ml / (-ml + 100.0)


def market_prob_home(g: GameRow) -> Optional[float]:
    """Devigged closing implied probability on the home side.

    Standard two-way devig: implied_home / (implied_home + implied_away).
    """
    if g.home_moneyline is None or g.away_moneyline is None:
        return None
    ph = _moneyline_to_prob(g.home_moneyline)
    pa = _moneyline_to_prob(g.away_moneyline)
    s = ph + pa
    if s <= 0:
        return None
    return ph / s


# ---------------------------------------------------------------------------
# PBP rollups — per-team per-game EPA stats
# ---------------------------------------------------------------------------


@dataclass
class TeamGameStats:
    """Per-(team, game) PBP-derived stats for the rolling builder."""
    game_id: str
    team: str
    season: int
    week: int
    game_date: date
    off_epa_per_play: float
    def_epa_per_play: float  # signed from defense perspective: negative = good defense
    success_rate: float
    pass_epa_per_play: Optional[float]
    rush_epa_per_play: Optional[float]
    n_off_plays: int
    n_def_plays: int


def load_pbp_rollups(seasons: list[int]) -> dict[str, list[TeamGameStats]]:
    """Aggregate PBP into per-(team, game) stats.

    Returns ``{team_code: [TeamGameStats sorted by game_date]}``.
    Aggregates BOTH the offense view (when posteam == team) AND the
    defense view (when defteam == team) in a single pass.
    """
    try:
        import nfl_data_py as nfl  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError:
        log.error("nfl_data_py/pandas not installed")
        return {}
    pbp = nfl.import_pbp_data(seasons, downcast=False, cache=False)
    # Keep only standard offensive plays with non-null EPA
    mask = pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()
    pbp = pbp.loc[mask].copy()
    # Aggregate per (game_id, posteam) for offense
    off = pbp.groupby(["game_id", "posteam", "season", "week"]).agg(
        off_epa=("epa", "mean"),
        off_plays=("epa", "size"),
        success=("success", "mean"),
        pass_epa=("epa", lambda s: s[pbp.loc[s.index, "play_type"] == "pass"].mean()),
        rush_epa=("epa", lambda s: s[pbp.loc[s.index, "play_type"] == "run"].mean()),
        game_date=("game_date", "first"),
    ).reset_index()
    # Aggregate per (game_id, defteam) for defense — EPA allowed on each play
    deff = pbp.groupby(["game_id", "defteam", "season", "week"]).agg(
        def_epa=("epa", "mean"),
        def_plays=("epa", "size"),
    ).reset_index().rename(columns={"defteam": "team"})
    off = off.rename(columns={"posteam": "team"})
    merged = off.merge(deff, on=["game_id", "team", "season", "week"], how="outer")
    out: dict[str, list[TeamGameStats]] = defaultdict(list)
    for _, row in merged.iterrows():
        team = _normalize_team(row.get("team"))
        if not team:
            continue
        gd = row.get("game_date")
        if gd is None or (isinstance(gd, float) and math.isnan(gd)):
            continue
        try:
            if hasattr(gd, "date"):
                d = gd.date() if not isinstance(gd, date) else gd
            else:
                d = datetime.strptime(str(gd), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        stat = TeamGameStats(
            game_id=str(row.get("game_id")),
            team=team,
            season=int(row.get("season")),
            week=int(row.get("week") or 0),
            game_date=d,
            off_epa_per_play=_safe_float(row.get("off_epa")) or 0.0,
            def_epa_per_play=_safe_float(row.get("def_epa")) or 0.0,
            success_rate=_safe_float(row.get("success")) or 0.0,
            pass_epa_per_play=_safe_float(row.get("pass_epa")),
            rush_epa_per_play=_safe_float(row.get("rush_epa")),
            n_off_plays=_safe_int(row.get("off_plays")) or 0,
            n_def_plays=_safe_int(row.get("def_plays")) or 0,
        )
        out[team].append(stat)
    for team in out:
        out[team].sort(key=lambda s: s.game_date)
    return dict(out)


# ---------------------------------------------------------------------------
# Rolling team rates — strict walk-forward
# ---------------------------------------------------------------------------


@dataclass
class RollingNFLFeatures:
    """Per-team rolling features at a moment in time."""

    off_epa: deque = field(default_factory=lambda: deque(maxlen=20))
    def_epa: deque = field(default_factory=lambda: deque(maxlen=20))
    success: deque = field(default_factory=lambda: deque(maxlen=20))
    pass_epa: deque = field(default_factory=lambda: deque(maxlen=20))
    rush_epa: deque = field(default_factory=lambda: deque(maxlen=20))
    last_game_date: Optional[date] = None
    last_season: Optional[int] = None

    def update(self, s: TeamGameStats) -> None:
        self.off_epa.append(s.off_epa_per_play)
        self.def_epa.append(s.def_epa_per_play)
        self.success.append(s.success_rate)
        if s.pass_epa_per_play is not None:
            self.pass_epa.append(s.pass_epa_per_play)
        if s.rush_epa_per_play is not None:
            self.rush_epa.append(s.rush_epa_per_play)
        self.last_game_date = s.game_date
        self.last_season = s.season

    def _avg(self, dq: deque, n: int) -> Optional[float]:
        if len(dq) < n:
            return None
        recent = list(dq)[-n:]
        return sum(recent) / n

    def off_epa_n(self, n: int) -> Optional[float]:
        return self._avg(self.off_epa, n)

    def def_epa_n(self, n: int) -> Optional[float]:
        return self._avg(self.def_epa, n)

    def success_n(self, n: int) -> Optional[float]:
        return self._avg(self.success, n)

    def pass_epa_n(self, n: int) -> Optional[float]:
        return self._avg(self.pass_epa, n)

    def rush_epa_n(self, n: int) -> Optional[float]:
        return self._avg(self.rush_epa, n)


def fit_rolling_rates(
    team_stats: dict[str, list[TeamGameStats]],
    *,
    season_reset: bool = True,
) -> dict[tuple[date, str], RollingNFLFeatures]:
    """Build a snapshot of every team's rolling features just BEFORE each game.

    Returns ``{(game_date, team_code): RollingNFLFeatures}``. The snapshot
    at key ``(d, t)`` reflects ALL games of team ``t`` strictly before
    ``d`` (in the same season, if ``season_reset=True``).
    """
    snapshots: dict[tuple[date, str], RollingNFLFeatures] = {}
    for team, stats in team_stats.items():
        state = RollingNFLFeatures()
        for s in stats:
            # Season reset BEFORE snapshot so the snapshot reflects fresh
            # season state (not last season's carry-over).
            if season_reset and state.last_season is not None and state.last_season < s.season:
                state = RollingNFLFeatures()
            # Snapshot AS-OF this game's date (reflects only prior games).
            snapshots[(s.game_date, team)] = _freeze(state)
            # Then update with this game's stats so subsequent snapshots see it.
            state.update(s)
    return snapshots


def _freeze(f: RollingNFLFeatures) -> RollingNFLFeatures:
    out = RollingNFLFeatures()
    out.off_epa = deque(list(f.off_epa), maxlen=20)
    out.def_epa = deque(list(f.def_epa), maxlen=20)
    out.success = deque(list(f.success), maxlen=20)
    out.pass_epa = deque(list(f.pass_epa), maxlen=20)
    out.rush_epa = deque(list(f.rush_epa), maxlen=20)
    out.last_game_date = f.last_game_date
    out.last_season = f.last_season
    return out


# ---------------------------------------------------------------------------
# Bye-week detection
# ---------------------------------------------------------------------------


def compute_bye_status(games: list[GameRow]) -> dict[tuple[date, str], bool]:
    """For each (date, team) determine whether the team was coming off a bye.

    A team is "off a bye" if their previous game in the same season was
    > 10 days ago (modal rest day = 7; bye-week rest = 14).
    """
    last_game: dict[str, tuple[date, int]] = {}
    by_team_dates: dict[str, list[tuple[date, str, int]]] = defaultdict(list)
    for g in games:
        by_team_dates[g.home].append((g.game_date, "home", g.season))
        by_team_dates[g.away].append((g.game_date, "away", g.season))
    out: dict[tuple[date, str], bool] = {}
    for team, dates in by_team_dates.items():
        dates.sort()
        prev_date: Optional[date] = None
        prev_season: Optional[int] = None
        for d, _role, season in dates:
            if prev_date is None or prev_season != season:
                out[(d, team)] = False  # season opener; no bye
            else:
                out[(d, team)] = (d - prev_date).days > 10
            prev_date = d
            prev_season = season
    return out


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


FEATURE_NAMES: list[str] = [
    # Priors (3)
    "elo_prob_home",
    "epa_prob_home",
    "market_prob_home",
    # Prior-stack signals (3)
    "elo_minus_market_pp",
    "epa_minus_market_pp",
    "priors_avg",
    # Rolling form (6)
    "off_epa_l4_diff",
    "def_epa_l4_diff",
    "success_rate_l4_diff",
    "off_epa_l8_diff",
    "pass_epa_l4_diff",
    "rush_epa_l4_diff",
    # Schedule / rest (5)
    "rest_diff",
    "home_off_bye",
    "away_off_bye",
    "divisional",
    "week_number",
]


def build_features(
    game: GameRow,
    rolling: dict[tuple[date, str], RollingNFLFeatures],
    bye_status: dict[tuple[date, str], bool],
) -> Optional[dict]:
    """Build the per-game feature dict, strictly walk-forward.

    Returns ``None`` if any **required** feature is missing.
    Required: market_prob_home and either L4 rolling EPA for both teams.
    """
    # Leakage gate
    home_f = rolling.get((game.game_date, game.home), RollingNFLFeatures())
    away_f = rolling.get((game.game_date, game.away), RollingNFLFeatures())
    assert home_f.last_game_date is None or home_f.last_game_date < game.game_date, (
        f"leakage: home {game.home} last_game_date {home_f.last_game_date} >= {game.game_date}"
    )
    assert away_f.last_game_date is None or away_f.last_game_date < game.game_date, (
        f"leakage: away {game.away} last_game_date {away_f.last_game_date} >= {game.game_date}"
    )

    mkt = market_prob_home(game)

    def _diff(fn_h, fn_a, n: int) -> Optional[float]:
        a = fn_h(n)
        b = fn_a(n)
        if a is None or b is None:
            return None
        return a - b

    feats: dict = {
        "elo_prob_home": game.elo_prob_home,
        "epa_prob_home": game.epa_prob_home,
        "market_prob_home": mkt,
        "elo_minus_market_pp": (
            (game.elo_prob_home - mkt) if (game.elo_prob_home is not None and mkt is not None) else None
        ),
        "epa_minus_market_pp": (
            (game.epa_prob_home - mkt) if (game.epa_prob_home is not None and mkt is not None) else None
        ),
        "priors_avg": None,
        "off_epa_l4_diff": _diff(home_f.off_epa_n, away_f.off_epa_n, 4),
        "def_epa_l4_diff": _diff(home_f.def_epa_n, away_f.def_epa_n, 4),
        "success_rate_l4_diff": _diff(home_f.success_n, away_f.success_n, 4),
        "off_epa_l8_diff": _diff(home_f.off_epa_n, away_f.off_epa_n, 8),
        "pass_epa_l4_diff": _diff(home_f.pass_epa_n, away_f.pass_epa_n, 4),
        "rush_epa_l4_diff": _diff(home_f.rush_epa_n, away_f.rush_epa_n, 4),
        "rest_diff": (
            float(game.home_rest - game.away_rest)
            if (game.home_rest is not None and game.away_rest is not None) else None
        ),
        "home_off_bye": 1.0 if bye_status.get((game.game_date, game.home), False) else 0.0,
        "away_off_bye": 1.0 if bye_status.get((game.game_date, game.away), False) else 0.0,
        "divisional": 1.0 if _same_division(game.home, game.away) else 0.0,
        "week_number": float(game.week) / 18.0 if game.week else 0.0,
    }
    # Composite prior — mean of (elo, epa, market) where each is present.
    parts = [p for p in (game.elo_prob_home, game.epa_prob_home, mkt) if p is not None]
    if parts:
        feats["priors_avg"] = sum(parts) / len(parts)

    # Required: market_prob_home (always present for any game with closing odds)
    # AND L4 rolling EPA for both teams.
    if feats["market_prob_home"] is None:
        return None
    if feats["off_epa_l4_diff"] is None or feats["def_epa_l4_diff"] is None:
        return None
    return feats


def feature_vector(feats: dict, fill_value: float = 0.0) -> list[float]:
    """Convert a feature dict to a positional vector in ``FEATURE_NAMES`` order."""
    return [
        float(feats[name]) if feats.get(name) is not None else fill_value
        for name in FEATURE_NAMES
    ]
