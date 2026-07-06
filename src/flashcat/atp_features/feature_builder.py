"""Walk-forward feature builder for ATP matches — Phase 1.

Pure functions only. Every feature is computable using ONLY data strictly
BEFORE the match's date. The leakage gate is asserted on every call.

Data spine
----------
tennis-data.co.uk season xlsx (one row per completed main-tour singles
match) supplies: winner/loser, WRank/LRank, WPts/LPts, surface, round,
series, per-set scores, and closing decimal odds (Pinnacle preferred).

We canonicalize the alphabetically-first player name as "home" to match
the convention used by ``scripts/backfill_tennis_historical.py`` and
``sources/tennis_history.py`` — this makes the persisted ``tennis-rank-bt``
and ``market-close`` prior probabilities join cleanly by event_id.

Priors
------
* ``market-close``  — devigged closing two-way moneyline. Read from
  ``source_history.db`` (populated by the backfill). Also our CLV proxy.
* ``tennis-rank-bt``— Bradley-Terry on ATP rank points. Read from the DB.

Required-feature gate
---------------------
At minimum ``market_prob_home`` must be present (every archived match has
closing odds). The rolling form features need at least 10 prior matches
per player; matches that fail any required feature are excluded and show
up as ``n_loaded`` minus ``n_with_features`` in the report.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonicalization — must match tennis_history / backfill conventions
# ---------------------------------------------------------------------------

def _canonical_pair(p1: str, p2: str) -> tuple[str, str, bool]:
    """Return (home, away, swap) with home alphabetically first.

    ``swap`` indicates whether the original (winner, loser) pair was
    flipped relative to our canonical ordering. Mirrors
    ``sources/tennis_history.py::_canonical_pair``.
    """
    if p1 <= p2:
        return p1, p2, False
    return p2, p1, True


def _norm_name(name: str) -> str:
    """Loose name normalizer (lastname + first-initial, lower-cased).

    Mirrors ``scripts/backfill_tennis_historical.py::_norm`` so event_ids
    built here join the DB prior rows.
    """
    s = (name or "").strip()
    if not s:
        return ""
    tokens = s.split()
    if tokens and tokens[-1].endswith(".") and len(tokens[-1].replace(".", "")) <= 2:
        initial = tokens[-1].replace(".", "").strip().lower()[:1]
        last = " ".join(tokens[:-1])
        return f"{last.lower()} {initial}".strip()
    if len(tokens) >= 2:
        first = tokens[0]
        last = " ".join(tokens[1:])
        return f"{last.lower()} {first[:1].lower()}".strip()
    return s.lower()


def event_id(tour: str, match_day: date, p1: str, p2: str) -> str:
    """Canonical event_id — matches backfill_tennis_historical._event_id."""
    home, away, _ = _canonical_pair(p1, p2)
    return (
        f"tennis:{tour}:{match_day.isoformat()}:"
        f"{_norm_name(home)}-vs-{_norm_name(away)}"
    ).replace(" ", "_")


# ---------------------------------------------------------------------------
# MatchRow — canonical normalized record consumed by the feature builder
# ---------------------------------------------------------------------------

_SURFACES = ("Hard", "Clay", "Grass", "Carpet")


@dataclass
class MatchRow:
    """One completed ATP singles match, canonicalized (home = alpha-first)."""

    event_id: str
    match_date: date
    season: int
    tour: str
    home: str
    away: str
    home_won: Optional[bool]
    surface: str  # Hard / Clay / Grass / Carpet
    series: str   # ATP250 / ATP500 / Masters 1000 / Grand Slam / ...
    round: str
    best_of: int
    # Ranking snapshots (as published at match time).
    home_rank: Optional[int] = None
    away_rank: Optional[int] = None
    home_pts: Optional[float] = None
    away_pts: Optional[float] = None
    # Per-set games won (home perspective) for the games/sets share proxies.
    home_games: Optional[int] = None
    away_games: Optional[int] = None
    home_sets: Optional[int] = None
    away_sets: Optional[int] = None
    # Real closing decimal odds (Pinnacle preferred) — for true payout/CLV.
    home_decimal: Optional[float] = None
    away_decimal: Optional[float] = None
    # Priors (loaded from source_history.db by attach_priors_from_db).
    market_prob_home: Optional[float] = None
    rank_bt_prob_home: Optional[float] = None


# ---------------------------------------------------------------------------
# xlsx loader
# ---------------------------------------------------------------------------

def _safe_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return int(v)
    except (ValueError, TypeError):
        return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _parse_date(d) -> Optional[date]:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        return datetime.fromisoformat(str(d)).date()
    except Exception:
        return None


def load_matches_from_cache(
    seasons: list[int],
    *,
    tour: str = "atp",
    cache_dir: Optional[Path] = None,
) -> list[MatchRow]:
    """Load canonicalized ``MatchRow``s from the cached tennis-data xlsx files.

    Only completed matches with valid dates are returned. Requires the
    xlsx to have been downloaded by ``scripts/backfill_tennis_historical.py``.
    """
    import openpyxl  # local import — keeps base install light

    if cache_dir is None:
        from ..config import CACHE_DIR
        cache_dir = CACHE_DIR

    out: list[MatchRow] = []
    for year in seasons:
        path = cache_dir / f"tennis_data_{tour}_{year}.xlsx"
        if not path.exists():
            log.warning("tennis-data cache missing for %s %s: %s", tour, year, path)
            continue
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb.active
        header = None
        for raw in ws.iter_rows(values_only=True):
            if header is None:
                header = list(raw)
                continue
            row = dict(zip(header, raw))
            mr = _parse_match(row, year, tour)
            if mr is not None:
                out.append(mr)
        wb.close()
    out.sort(key=lambda m: m.match_date)
    return out


def _parse_match(row: dict, year: int, tour: str) -> Optional[MatchRow]:
    winner = (row.get("Winner") or "").strip()
    loser = (row.get("Loser") or "").strip()
    if not winner or not loser:
        return None
    if (row.get("Comment") or "").strip().lower() != "completed":
        return None
    match_day = _parse_date(row.get("Date"))
    if match_day is None:
        return None

    home, away, swap = _canonical_pair(winner, loser)
    home_won = home == winner

    wrank = _safe_int(row.get("WRank"))
    lrank = _safe_int(row.get("LRank"))
    wpts = _safe_float(row.get("WPts"))
    lpts = _safe_float(row.get("LPts"))
    wsets = _safe_int(row.get("Wsets"))
    lsets = _safe_int(row.get("Lsets"))

    # Games won across all sets (winner / loser perspective).
    w_games = 0
    l_games = 0
    any_games = False
    for i in range(1, 6):
        wg = _safe_int(row.get(f"W{i}"))
        lg = _safe_int(row.get(f"L{i}"))
        if wg is not None:
            w_games += wg
            any_games = True
        if lg is not None:
            l_games += lg
            any_games = True

    if swap:
        home_rank, away_rank = lrank, wrank
        home_pts, away_pts = lpts, wpts
        home_sets, away_sets = lsets, wsets
        home_games, away_games = (l_games, w_games) if any_games else (None, None)
    else:
        home_rank, away_rank = wrank, lrank
        home_pts, away_pts = wpts, lpts
        home_sets, away_sets = wsets, lsets
        home_games, away_games = (w_games, l_games) if any_games else (None, None)

    # Closing decimal odds — Pinnacle > Bet365 > market-average.
    def _f(x):
        return _safe_float(x)

    psw, psl = _f(row.get("PSW")), _f(row.get("PSL"))
    b365w, b365l = _f(row.get("B365W")), _f(row.get("B365L"))
    avgw, avgl = _f(row.get("AvgW")), _f(row.get("AvgL"))
    dec_w = next((x for x in (psw, b365w, avgw) if x is not None and x > 1.0), None)
    dec_l = next((x for x in (psl, b365l, avgl) if x is not None and x > 1.0), None)
    if dec_w is not None and dec_l is not None:
        home_decimal = dec_l if swap else dec_w
        away_decimal = dec_w if swap else dec_l
    else:
        home_decimal = away_decimal = None

    surface = (row.get("Surface") or "").strip() or "Hard"
    series = (row.get("Series") or "").strip()
    round_ = (row.get("Round") or "").strip()
    best_of = _safe_int(row.get("Best of")) or 3

    return MatchRow(
        event_id=event_id(tour, match_day, winner, loser),
        match_date=match_day,
        season=year,
        tour=tour,
        home=home,
        away=away,
        home_won=home_won,
        surface=surface,
        series=series,
        round=round_,
        best_of=best_of,
        home_rank=home_rank,
        away_rank=away_rank,
        home_pts=home_pts,
        away_pts=away_pts,
        home_games=home_games,
        away_games=away_games,
        home_sets=home_sets,
        away_sets=away_sets,
        home_decimal=home_decimal,
        away_decimal=away_decimal,
    )


# ---------------------------------------------------------------------------
# Prior loader — re-uses backfill_tennis_historical's source_history.db rows
# ---------------------------------------------------------------------------

def attach_priors_from_db(matches: list[MatchRow], db_path: Path) -> int:
    """Mutates ``matches`` in place, populating market_prob_home and
    rank_bt_prob_home from the source_history.db predictions table.

    Returns the number of prior rows attached. The backfill keys rows by
    the same ``event_id`` we build in ``event_id()``.
    """
    if not db_path.exists():
        log.warning("source_history.db not found at %s", db_path)
        return 0
    by_id = {m.event_id: m for m in matches}
    conn = sqlite3.connect(str(db_path))
    hit = 0
    try:
        cur = conn.execute(
            "SELECT event_id, source, home_prob FROM predictions "
            "WHERE sport IN ('atp') AND source IN "
            "('market-close','tennis-rank-bt')"
        )
        for eid, source, home_prob in cur:
            m = by_id.get(str(eid))
            if m is None or home_prob is None:
                continue
            if source == "market-close":
                m.market_prob_home = float(home_prob)
                hit += 1
            elif source == "tennis-rank-bt":
                m.rank_bt_prob_home = float(home_prob)
                hit += 1
    finally:
        conn.close()
    log.info("attached %d prior rows from source_history.db", hit)
    return hit


# ---------------------------------------------------------------------------
# Rolling per-player form — strict walk-forward
# ---------------------------------------------------------------------------

@dataclass
class RollingATPFeatures:
    """Per-player rolling form at a moment in time.

    Each deque stores per-match outcomes/ratios in chronological order.
    """

    wins: deque = field(default_factory=lambda: deque(maxlen=25))          # 1/0 win flags
    games_share: deque = field(default_factory=lambda: deque(maxlen=25))   # games won / total games
    sets_share: deque = field(default_factory=lambda: deque(maxlen=25))    # sets won / total sets
    # Surface-specific win flags, keyed by surface.
    surface_wins: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=20)))
    last_match_date: Optional[date] = None
    match_dates: deque = field(default_factory=lambda: deque(maxlen=40))   # for match-load counting

    def update(self, won: bool, games_share: Optional[float],
               sets_share: Optional[float], surface: str, d: date) -> None:
        self.wins.append(1 if won else 0)
        if games_share is not None:
            self.games_share.append(games_share)
        if sets_share is not None:
            self.sets_share.append(sets_share)
        self.surface_wins[surface].append(1 if won else 0)
        self.last_match_date = d
        self.match_dates.append(d)

    # ---- readers (return None when insufficient sample) ----

    def win_pct(self, n: int) -> Optional[float]:
        if len(self.wins) < n:
            return None
        recent = list(self.wins)[-n:]
        return sum(recent) / n

    def games_pct(self, n: int) -> Optional[float]:
        if len(self.games_share) < n:
            return None
        recent = list(self.games_share)[-n:]
        return sum(recent) / len(recent)

    def sets_pct(self, n: int) -> Optional[float]:
        if len(self.sets_share) < n:
            return None
        recent = list(self.sets_share)[-n:]
        return sum(recent) / len(recent)

    def surface_win_pct(self, surface: str, n: int) -> Optional[float]:
        dq = self.surface_wins.get(surface)
        if not dq or len(dq) < n:
            return None
        recent = list(dq)[-n:]
        return sum(recent) / n

    def rest_days(self, d: date) -> Optional[int]:
        if self.last_match_date is None:
            return None
        return (d - self.last_match_date).days

    def matches_in_trailing(self, d: date, days: int) -> int:
        return sum(1 for md in self.match_dates if 0 <= (d - md).days <= days)


def _freeze(f: RollingATPFeatures) -> RollingATPFeatures:
    out = RollingATPFeatures()
    out.wins = deque(list(f.wins), maxlen=25)
    out.games_share = deque(list(f.games_share), maxlen=25)
    out.sets_share = deque(list(f.sets_share), maxlen=25)
    out.surface_wins = defaultdict(lambda: deque(maxlen=20))
    for s, dq in f.surface_wins.items():
        out.surface_wins[s] = deque(list(dq), maxlen=20)
    out.last_match_date = f.last_match_date
    out.match_dates = deque(list(f.match_dates), maxlen=40)
    return out


def _match_shares(m: MatchRow) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (home_games_share, away_games_share, home_sets_share, away_sets_share)."""
    hg, ag = m.home_games, m.away_games
    hs, as_ = m.home_sets, m.away_sets
    games_home = games_away = None
    if hg is not None and ag is not None and (hg + ag) > 0:
        games_home = hg / (hg + ag)
        games_away = ag / (hg + ag)
    sets_home = sets_away = None
    if hs is not None and as_ is not None and (hs + as_) > 0:
        sets_home = hs / (hs + as_)
        sets_away = as_ / (hs + as_)
    return games_home, games_away, sets_home, sets_away


def fit_rolling_rates(
    matches: list[MatchRow],
) -> dict[tuple[str, str], RollingATPFeatures]:
    """Build a snapshot of every player's rolling form just BEFORE each match.

    Returns ``{(event_id, player_name): RollingATPFeatures}``. The snapshot
    reflects ALL of that player's matches strictly before the snapshot match.
    Keyed by (event_id, player) so home/away lookups are unambiguous even
    when the same two players meet twice.
    """
    state: dict[str, RollingATPFeatures] = defaultdict(RollingATPFeatures)
    snapshots: dict[tuple[str, str], RollingATPFeatures] = {}
    # Group matches by date. tennis-data has no intra-day match times, so we
    # cannot order same-day matches. To stay strictly walk-forward (and keep
    # the leakage gate honest at day granularity), every match on date D is
    # snapshotted against state reflecting only matches STRICTLY BEFORE D;
    # all of D's outcomes are then applied together AFTER snapshotting.
    by_date: dict[date, list[MatchRow]] = defaultdict(list)
    for m in matches:
        by_date[m.match_date].append(m)
    for d in sorted(by_date):
        day_matches = by_date[d]
        # 1) Snapshot every match on this date against pre-date state.
        for m in day_matches:
            snapshots[(m.event_id, m.home)] = _freeze(state[m.home])
            snapshots[(m.event_id, m.away)] = _freeze(state[m.away])
        # 2) Apply all of this date's outcomes.
        for m in day_matches:
            gh, ga, sh, sa = _match_shares(m)
            home_won = bool(m.home_won)
            state[m.home].update(home_won, gh, sh, m.surface, m.match_date)
            state[m.away].update(not home_won, ga, sa, m.surface, m.match_date)
    return snapshots


# ---------------------------------------------------------------------------
# Head-to-head — strict walk-forward
# ---------------------------------------------------------------------------

def compute_h2h(matches: list[MatchRow]) -> dict[str, float]:
    """For each match event_id, the home player's H2H win share vs this
    opponent BEFORE this match (0.5 prior when they've never met).
    """
    h2h: dict[tuple[str, str], list[int]] = defaultdict(list)  # (a,b) sorted -> home-perspective flags
    out: dict[str, float] = {}
    # Date-group so a same-day rematch reads prior (< D) history only.
    by_date: dict[date, list[MatchRow]] = defaultdict(list)
    for m in matches:
        by_date[m.match_date].append(m)
    for d in sorted(by_date):
        day_matches = by_date[d]
        for m in day_matches:
            key = (m.home, m.away)  # home is alpha-first, so ordering is stable
            prior = h2h.get(key, [])
            out[m.event_id] = (sum(prior) / len(prior)) if prior else 0.5
        for m in day_matches:
            h2h[(m.home, m.away)].append(1 if m.home_won else 0)
    return out


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # Priors / market (3)
    "market_prob_home",
    "rank_bt_prob_home",
    "rank_bt_minus_market_pp",
    # Ranking (2)
    "rank_log_ratio",
    "rank_points_log_ratio",
    # Rolling form (5)
    "win_pct_l10_diff",
    "win_pct_l25_diff",
    "surface_win_pct_l20_diff",
    "games_won_pct_l10_diff",
    "sets_won_pct_l10_diff",
    # Schedule / fatigue (2)
    "rest_days_diff",
    "matches_l14_diff",
    # H2H + context (2)
    "h2h_home_share",
    "best_of_5",
]


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def build_features(
    match: MatchRow,
    rolling: dict[tuple[str, str], RollingATPFeatures],
    h2h: dict[str, float],
) -> Optional[dict]:
    """Build the per-match feature dict, strictly walk-forward.

    Returns ``None`` if any **required** feature is missing.
    Required: ``market_prob_home`` AND win_pct_l10 for both players.
    """
    home_f = rolling.get((match.event_id, match.home), RollingATPFeatures())
    away_f = rolling.get((match.event_id, match.away), RollingATPFeatures())

    # Leakage gate — the frozen snapshot must not include this match's date
    # or any later match.
    assert home_f.last_match_date is None or home_f.last_match_date < match.match_date, (
        f"leakage: home {match.home} last_match {home_f.last_match_date} >= {match.match_date}"
    )
    assert away_f.last_match_date is None or away_f.last_match_date < match.match_date, (
        f"leakage: away {match.away} last_match {away_f.last_match_date} >= {match.match_date}"
    )

    mkt = match.market_prob_home
    bt = match.rank_bt_prob_home

    # Ranking ratios (lower rank number = better player → invert for a
    # "home strength" sign: away_rank/home_rank > 1 means home ranked better).
    rank_log_ratio = None
    if match.home_rank and match.away_rank and match.home_rank > 0 and match.away_rank > 0:
        rank_log_ratio = math.log(match.away_rank / match.home_rank)
    rank_pts_log_ratio = None
    if match.home_pts and match.away_pts and match.home_pts > 0 and match.away_pts > 0:
        rank_pts_log_ratio = math.log(match.home_pts / match.away_pts)

    # Rest days — capped to a sane range so a 200-day layoff doesn't dominate.
    def _rest(f: RollingATPFeatures) -> Optional[float]:
        r = f.rest_days(match.match_date)
        if r is None:
            return None
        return float(min(r, 60))

    rest_home = _rest(home_f)
    rest_away = _rest(away_f)
    rest_diff = _diff(rest_home, rest_away)

    matches_l14_diff = float(
        home_f.matches_in_trailing(match.match_date, 14)
        - away_f.matches_in_trailing(match.match_date, 14)
    )

    feats: dict = {
        "market_prob_home": mkt,
        "rank_bt_prob_home": bt,
        "rank_bt_minus_market_pp": (bt - mkt) if (bt is not None and mkt is not None) else None,
        "rank_log_ratio": rank_log_ratio,
        "rank_points_log_ratio": rank_pts_log_ratio,
        "win_pct_l10_diff": _diff(home_f.win_pct(10), away_f.win_pct(10)),
        "win_pct_l25_diff": _diff(home_f.win_pct(25), away_f.win_pct(25)),
        "surface_win_pct_l20_diff": _diff(
            home_f.surface_win_pct(match.surface, 20),
            away_f.surface_win_pct(match.surface, 20),
        ),
        "games_won_pct_l10_diff": _diff(home_f.games_pct(10), away_f.games_pct(10)),
        "sets_won_pct_l10_diff": _diff(home_f.sets_pct(10), away_f.sets_pct(10)),
        "rest_days_diff": rest_diff,
        "matches_l14_diff": matches_l14_diff,
        "h2h_home_share": h2h.get(match.event_id, 0.5),
        "best_of_5": 1.0 if match.best_of == 5 else 0.0,
    }

    # Required-feature gate.
    if feats["market_prob_home"] is None:
        return None
    if feats["win_pct_l10_diff"] is None:
        return None
    return feats


def feature_vector(feats: dict, fill_value: float = 0.0) -> list[float]:
    """Convert a feature dict to a positional vector in ``FEATURE_NAMES`` order."""
    return [
        float(feats[name]) if feats.get(name) is not None else fill_value
        for name in FEATURE_NAMES
    ]
