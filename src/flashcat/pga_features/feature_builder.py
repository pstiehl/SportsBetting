"""Walk-forward feature builder for PGA head-to-head matchups — Phase 1.

Pure functions only. Every feature is computable using ONLY data strictly
BEFORE the matchup's tournament date. The leakage gate is asserted on every
call.

This is a PORT of ``wta_features/feature_builder.py`` (Week 6) to golf.
The event shape is the same H2H matchup the tennis harness expects — two
players, ``home`` = alphabetically-first player, ``home_won`` = whether the
alpha-first player finished ahead over the tournament. The analogies:

    tennis surface        -> PGA course difficulty tier
    tennis ranking points -> DataGolf skill estimate / pre-tournament win%
    tennis rank-BT prior  -> datagolf-sg Bradley-Terry matchup prior
    tennis market-close   -> matchup closing moneyline (CLV proxy)

Data spine
----------
The production ``sources/pga_datagolf.py`` connector already synthesizes
H2H matchup Events with ``home_win_prob`` = Bradley-Terry on DataGolf
skill. Historical graded rows (player finished-ahead outcomes) plus a
matchup closing-line prior are read from ``source_history.db`` under the
sources ``datagolf-sg`` (BT prior) and ``market-close`` (devigged closing
matchup moneyline — also our CLV proxy), keyed by the ``event_id`` this
module builds.

CRITICAL DATA NOTE (Week 7, 2026-07-20)
---------------------------------------
As of this Phase-1 landing there are **no PGA rows in source_history.db**.
DataGolf's historical matchup-odds archive (the only known source of
genuine closing H2H matchup prices) is a PAID-tier endpoint and is OFF
LIMITS per Phil's standing constraint; ``DATAGOLF_API_KEY`` is also unset
in this environment. No free source pairing (player finish results +
closing matchup probability) was found. So this harness ships
leakage-gated and unit-tested against a fixture, ready to run the moment a
data source or key is provided — but the real walk-forward backtest is
BLOCKED on data access. See the PR body and playbook §6 (honest
evaluation over green dashboards).

Required-feature gate
---------------------
At minimum ``market_prob_home`` (the matchup closing prior / CLV proxy)
must be present, plus each player's rolling finish form over the last 10
starts. Matchups failing any required feature are excluded and show up as
``n_loaded`` minus ``n_with_features`` in the report.
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
# Canonicalization — must match sources/pga_datagolf conventions
# ---------------------------------------------------------------------------

def _canonical_pair(p1: str, p2: str) -> tuple[str, str, bool]:
    """Return (home, away, swap) with home the alphabetically-first player.

    ``swap`` indicates whether the original (p1, p2) pair was flipped
    relative to our canonical ordering. Mirrors
    ``sources/pga_datagolf.py::_canonical_pair`` (which lower-cases for the
    comparison but returns the original casing).
    """
    na = (p1 or "").strip().lower()
    nb = (p2 or "").strip().lower()
    if na <= nb:
        return p1, p2, False
    return p2, p1, True


def _norm_name(name: str) -> str:
    """Loose player-name normalizer (lower-cased, punctuation stripped).

    Mirrors ``sources/pga_datagolf.py::_normalize_player`` so event_ids
    built here join DataGolf-sourced prior rows.
    """
    return (
        (name or "")
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace("  ", " ")
        .strip()
    )


def event_id(tour: str, event_label: str, p1: str, p2: str) -> str:
    """Canonical event_id for a PGA H2H matchup.

    Mirrors ``sources/pga_datagolf.py::_event_id_for_pair`` so the harness's
    ids join the connector's persisted prior rows. ``tour`` is normally
    "pga"; ``event_label`` is the tournament name.
    """
    home, away, _ = _canonical_pair(p1, p2)
    slug = (
        f"{tour}:{event_label}:"
        f"{_norm_name(home).replace(' ', '_')}_vs_"
        f"{_norm_name(away).replace(' ', '_')}"
    )
    return f"datagolf:{slug}"


# ---------------------------------------------------------------------------
# MatchupRow — canonical normalized record consumed by the feature builder
# ---------------------------------------------------------------------------

# Course difficulty tiers (analog of tennis surface). "hard" here means a
# hard/penal course setup (majors, tough par-70s), NOT a hard tennis court.
_COURSE_TIERS = ("easy", "standard", "hard", "major")


@dataclass
class MatchupRow:
    """One completed PGA head-to-head matchup, canonicalized (home = alpha-first).

    ``home_won`` is True when the alphabetically-first player finished
    strictly ahead of the other on the tournament leaderboard (ties are
    graded as ``None`` — a push — and dropped from the sample).
    """

    event_id: str
    match_date: date          # tournament start date
    season: int
    tour: str                 # "pga"
    event_label: str          # tournament name
    home: str
    away: str
    home_won: Optional[bool]
    course_tier: str          # easy / standard / hard / major
    # Pre-tournament skill snapshots (DataGolf win% or skill estimate).
    home_win_pct: Optional[float] = None   # pre-tournament outright win prob [0,1]
    away_win_pct: Optional[float] = None
    home_skill: Optional[float] = None     # raw DG skill estimate (SG total)
    away_skill: Optional[float] = None
    # Finishing positions for form updates (lower = better; 1 = win).
    home_finish: Optional[int] = None
    away_finish: Optional[int] = None
    # Real closing decimal odds for the matchup, when archived.
    home_decimal: Optional[float] = None
    away_decimal: Optional[float] = None
    # Priors (loaded from source_history.db by attach_priors_from_db).
    market_prob_home: Optional[float] = None
    skill_bt_prob_home: Optional[float] = None


# ---------------------------------------------------------------------------
# safe casters
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


# ---------------------------------------------------------------------------
# Prior loader — reuses source_history.db rows written by a PGA backfill
# ---------------------------------------------------------------------------

def attach_priors_from_db(matchups: list[MatchupRow], db_path: Path) -> int:
    """Mutate ``matchups`` in place, populating ``market_prob_home`` and
    ``skill_bt_prob_home`` from the source_history.db predictions table.

    Returns the number of prior rows attached. Rows are keyed by the same
    ``event_id`` we build in ``event_id()``. ``datagolf-sg`` supplies the
    Bradley-Terry matchup prior; ``market-close`` the devigged closing
    matchup moneyline (also the CLV proxy).
    """
    if not db_path.exists():
        log.warning("source_history.db not found at %s", db_path)
        return 0
    by_id = {m.event_id: m for m in matchups}
    conn = sqlite3.connect(str(db_path))
    hit = 0
    try:
        cur = conn.execute(
            "SELECT event_id, source, home_prob, market_close_decimal, "
            "closing_implied_prob FROM predictions "
            "WHERE sport = 'pga' AND source IN ('market-close', 'datagolf-sg')"
        )
        for eid, source, home_prob, close_dec, close_prob in cur:
            m = by_id.get(str(eid))
            if m is None:
                continue
            if source == "market-close":
                if close_prob is not None:
                    m.market_prob_home = float(close_prob)
                elif home_prob is not None:
                    m.market_prob_home = float(home_prob)
                if close_dec is not None and m.home_decimal is None:
                    m.home_decimal = float(close_dec)
                hit += 1
            elif source == "datagolf-sg" and home_prob is not None:
                m.skill_bt_prob_home = float(home_prob)
                hit += 1
    finally:
        conn.close()
    log.info("attached %d prior rows from source_history.db", hit)
    return hit


# ---------------------------------------------------------------------------
# Rolling per-player form — strict walk-forward
# ---------------------------------------------------------------------------

@dataclass
class RollingPGAFeatures:
    """Per-player rolling tournament form at a moment in time.

    Deques store per-start outcomes in chronological order.
    """

    # H2H-outcome flags (1 = finished ahead of that week's paired opponent).
    h2h_results: deque = field(default_factory=lambda: deque(maxlen=25))
    # Made-cut flags (1 = made the cut).
    made_cut: deque = field(default_factory=lambda: deque(maxlen=25))
    # Finishing-position "quality": 1.0 for a win, decaying to 0 for missed cut.
    finish_quality: deque = field(default_factory=lambda: deque(maxlen=25))
    # Course-tier-specific finish quality.
    tier_quality: dict = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=20))
    )
    last_start_date: Optional[date] = None
    start_dates: deque = field(default_factory=lambda: deque(maxlen=40))

    def update(
        self,
        finished_ahead: bool,
        made_cut: Optional[bool],
        finish_quality: Optional[float],
        course_tier: str,
        d: date,
    ) -> None:
        self.h2h_results.append(1 if finished_ahead else 0)
        if made_cut is not None:
            self.made_cut.append(1 if made_cut else 0)
        if finish_quality is not None:
            self.finish_quality.append(finish_quality)
            self.tier_quality[course_tier].append(finish_quality)
        self.last_start_date = d
        self.start_dates.append(d)

    # ---- readers (return None when insufficient sample) ----

    def h2h_win_pct(self, n: int) -> Optional[float]:
        if len(self.h2h_results) < n:
            return None
        recent = list(self.h2h_results)[-n:]
        return sum(recent) / n

    def made_cut_pct(self, n: int) -> Optional[float]:
        if len(self.made_cut) < n:
            return None
        recent = list(self.made_cut)[-n:]
        return sum(recent) / n

    def finish_quality_avg(self, n: int) -> Optional[float]:
        if len(self.finish_quality) < n:
            return None
        recent = list(self.finish_quality)[-n:]
        return sum(recent) / len(recent)

    def tier_quality_avg(self, tier: str, n: int) -> Optional[float]:
        dq = self.tier_quality.get(tier)
        if not dq or len(dq) < n:
            return None
        recent = list(dq)[-n:]
        return sum(recent) / len(recent)

    def rest_days(self, d: date) -> Optional[int]:
        if self.last_start_date is None:
            return None
        return (d - self.last_start_date).days

    def starts_in_trailing(self, d: date, days: int) -> int:
        return sum(1 for sd in self.start_dates if 0 <= (d - sd).days <= days)


def _freeze(f: RollingPGAFeatures) -> RollingPGAFeatures:
    out = RollingPGAFeatures()
    out.h2h_results = deque(list(f.h2h_results), maxlen=25)
    out.made_cut = deque(list(f.made_cut), maxlen=25)
    out.finish_quality = deque(list(f.finish_quality), maxlen=25)
    out.tier_quality = defaultdict(lambda: deque(maxlen=20))
    for t, dq in f.tier_quality.items():
        out.tier_quality[t] = deque(list(dq), maxlen=20)
    out.last_start_date = f.last_start_date
    out.start_dates = deque(list(f.start_dates), maxlen=40)
    return out


def _finish_quality(finish: Optional[int], field_size: int = 156) -> Optional[float]:
    """Map a finishing position to a [0, 1] quality score.

    1st = 1.0, missed cut / no finish (None) = 0.0, linear-ish in between
    using a soft rank transform so a T5 is much better than a T50 but the
    gap between T80 and T120 is small.
    """
    if finish is None:
        return 0.0
    if finish <= 0:
        return None
    # Soft transform: exp decay on rank keeps top finishes well-separated.
    return math.exp(-(finish - 1) / 25.0)


def fit_rolling_rates(
    matchups: list[MatchupRow],
) -> dict[tuple[str, str], RollingPGAFeatures]:
    """Build a snapshot of every player's rolling form just BEFORE each matchup.

    Returns ``{(event_id, player_name): RollingPGAFeatures}``. The snapshot
    reflects ALL of that player's tournaments strictly before the snapshot
    matchup's date. Keyed by (event_id, player) so home/away lookups are
    unambiguous even when the same two players meet twice.

    Matchups are grouped by tournament date. All matchups on date D are
    snapshotted against state reflecting only tournaments STRICTLY BEFORE D;
    then all of D's outcomes are applied together after snapshotting. This
    keeps the leakage gate honest at day granularity (all of a tournament's
    H2H pairings share a start date).
    """
    state: dict[str, RollingPGAFeatures] = defaultdict(RollingPGAFeatures)
    snapshots: dict[tuple[str, str], RollingPGAFeatures] = {}
    by_date: dict[date, list[MatchupRow]] = defaultdict(list)
    for m in matchups:
        by_date[m.match_date].append(m)
    for d in sorted(by_date):
        day = by_date[d]
        # 1) Snapshot every matchup on this date against pre-date state.
        for m in day:
            snapshots[(m.event_id, m.home)] = _freeze(state[m.home])
            snapshots[(m.event_id, m.away)] = _freeze(state[m.away])
        # 2) Apply all of this date's outcomes. Each player appears in at
        #    most one matchup per tournament (adjacent pairing), so a single
        #    update per player per date is correct.
        seen: set[str] = set()
        for m in day:
            home_ahead = bool(m.home_won)
            hq = _finish_quality(m.home_finish)
            aq = _finish_quality(m.away_finish)
            h_cut = None if m.home_finish is None else (m.home_finish > 0)
            a_cut = None if m.away_finish is None else (m.away_finish > 0)
            if m.home not in seen:
                state[m.home].update(home_ahead, h_cut, hq, m.course_tier, d)
                seen.add(m.home)
            if m.away not in seen:
                state[m.away].update(not home_ahead, a_cut, aq, m.course_tier, d)
                seen.add(m.away)
    return snapshots


# ---------------------------------------------------------------------------
# Head-to-head — strict walk-forward
# ---------------------------------------------------------------------------

def compute_h2h(matchups: list[MatchupRow]) -> dict[str, float]:
    """For each matchup event_id, the home player's H2H finish-ahead share
    vs this specific opponent BEFORE this matchup (0.5 prior on first meet).
    """
    h2h: dict[tuple[str, str], list[int]] = defaultdict(list)
    out: dict[str, float] = {}
    by_date: dict[date, list[MatchupRow]] = defaultdict(list)
    for m in matchups:
        by_date[m.match_date].append(m)
    for d in sorted(by_date):
        day = by_date[d]
        for m in day:
            key = (m.home, m.away)  # home is alpha-first, stable ordering
            prior = h2h.get(key, [])
            out[m.event_id] = (sum(prior) / len(prior)) if prior else 0.5
        for m in day:
            h2h[(m.home, m.away)].append(1 if m.home_won else 0)
    return out


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
# PGA catalog: 12 features. Adapts the WTA 13-feature catalog to golf. The
# WTA "sets_won_pct" feature has no golf analog and is dropped; "games_won"
# maps to made-cut rate; surface maps to course tier; ranking maps to the
# DataGolf pre-tournament win% / skill estimate.

FEATURE_NAMES: list[str] = [
    # Priors / market (3)
    "market_prob_home",
    "skill_bt_prob_home",
    "skill_bt_minus_market_pp",
    # Skill / rating (2)
    "win_pct_log_ratio",       # log(home_win_pct / away_win_pct)
    "skill_diff",              # home_skill - away_skill (SG total delta)
    # Rolling form (4)
    "h2h_form_l10_diff",       # rolling finish-ahead rate over last 10 starts
    "finish_quality_l10_diff", # avg finish quality over last 10 starts
    "made_cut_pct_l10_diff",   # made-cut rate over last 10 starts
    "course_tier_quality_l10_diff",  # finish quality on this course tier
    # Schedule / fatigue (2)
    "rest_days_diff",          # days since last start (home - away), capped
    "starts_l28_diff",         # starts in trailing 28 days (home - away)
    # H2H (1)
    "h2h_home_share",
]


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def build_features(
    matchup: MatchupRow,
    rolling: dict[tuple[str, str], RollingPGAFeatures],
    h2h: dict[str, float],
) -> Optional[dict]:
    """Build the per-matchup feature dict, strictly walk-forward.

    Returns ``None`` if any **required** feature is missing.
    Required: ``market_prob_home`` AND rolling H2H form L10 for both players.
    """
    home_f = rolling.get((matchup.event_id, matchup.home), RollingPGAFeatures())
    away_f = rolling.get((matchup.event_id, matchup.away), RollingPGAFeatures())

    # Leakage gate — the frozen snapshot must not include this matchup's date
    # or any later tournament.
    assert home_f.last_start_date is None or home_f.last_start_date < matchup.match_date, (
        f"leakage: home {matchup.home} last_start {home_f.last_start_date} "
        f">= {matchup.match_date}"
    )
    assert away_f.last_start_date is None or away_f.last_start_date < matchup.match_date, (
        f"leakage: away {matchup.away} last_start {away_f.last_start_date} "
        f">= {matchup.match_date}"
    )

    mkt = matchup.market_prob_home
    bt = matchup.skill_bt_prob_home

    # Win-prob log ratio — higher home win% => positive.
    win_pct_log_ratio = None
    if (
        matchup.home_win_pct and matchup.away_win_pct
        and matchup.home_win_pct > 0 and matchup.away_win_pct > 0
    ):
        win_pct_log_ratio = math.log(matchup.home_win_pct / matchup.away_win_pct)

    skill_diff = _diff(matchup.home_skill, matchup.away_skill)

    # Rest days — capped so a 200-day off-season layoff doesn't dominate.
    def _rest(f: RollingPGAFeatures) -> Optional[float]:
        r = f.rest_days(matchup.match_date)
        if r is None:
            return None
        return float(min(r, 90))

    rest_diff = _diff(_rest(home_f), _rest(away_f))

    starts_l28_diff = float(
        home_f.starts_in_trailing(matchup.match_date, 28)
        - away_f.starts_in_trailing(matchup.match_date, 28)
    )

    feats: dict = {
        "market_prob_home": mkt,
        "skill_bt_prob_home": bt,
        "skill_bt_minus_market_pp": (bt - mkt) if (bt is not None and mkt is not None) else None,
        "win_pct_log_ratio": win_pct_log_ratio,
        "skill_diff": skill_diff,
        "h2h_form_l10_diff": _diff(home_f.h2h_win_pct(10), away_f.h2h_win_pct(10)),
        "finish_quality_l10_diff": _diff(
            home_f.finish_quality_avg(10), away_f.finish_quality_avg(10)
        ),
        "made_cut_pct_l10_diff": _diff(
            home_f.made_cut_pct(10), away_f.made_cut_pct(10)
        ),
        "course_tier_quality_l10_diff": _diff(
            home_f.tier_quality_avg(matchup.course_tier, 10),
            away_f.tier_quality_avg(matchup.course_tier, 10),
        ),
        "rest_days_diff": rest_diff,
        "starts_l28_diff": starts_l28_diff,
        "h2h_home_share": h2h.get(matchup.event_id, 0.5),
    }

    # Required-feature gate.
    if feats["market_prob_home"] is None:
        return None
    if feats["h2h_form_l10_diff"] is None:
        return None
    return feats


def feature_vector(feats: dict, fill_value: float = 0.0) -> list[float]:
    """Convert a feature dict to a positional vector in ``FEATURE_NAMES`` order."""
    return [
        float(feats[name]) if feats.get(name) is not None else fill_value
        for name in FEATURE_NAMES
    ]
