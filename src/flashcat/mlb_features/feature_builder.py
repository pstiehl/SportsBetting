"""Walk-forward feature builder for MLB games.

Pure functions only — no I/O outside the explicit loader entry points.
Every feature passed to the model has to be computable using ONLY data
strictly BEFORE the game's date. This file makes that contract explicit
and asserts it in the rolling-rate builder.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import sqlite3
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

try:  # network optional — degrades to None features when offline
    import httpx  # type: ignore
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore

from ..config import CACHE_DIR, DATA_DIR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 538 MLB Elo CSV (committed in data/cache/538_mlb_elo.csv).
# Source: https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv (now 404,
# replaced by ABC News redirect on 2023-10-01 when 538 was shuttered). The
# committed cache covers 1871-2023; we only ever consume 2022+.
# ---------------------------------------------------------------------------

ELO_538_CACHE = CACHE_DIR / "538_mlb_elo.csv"
ELO_538_URL = "https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv"

# Retrosheet game logs — one ZIP per year, covers 1871-present, free.
# ``https://www.retrosheet.org/gamelogs/gl{YYYY}.zip`` returns gl{YYYY}.txt
# (CSV-shaped but uses non-RFC quoting; csv module handles it).
RETROSHEET_BASE = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
RETROSHEET_CACHE = CACHE_DIR / "retrosheet"

# Static park table (lat/lon/orientation/baseline run env).
PARKS_PATH = DATA_DIR / "mlb_parks.json"

# Open-Meteo historical archive (no API key, free for archive). Best for
# back-tests because it returns hourly weather by lat/lon for any past
# date. The connector caches per (date, lat, lon).
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CACHE = CACHE_DIR / "open_meteo"


# ---------------------------------------------------------------------------
# GameRow — canonical normalized record consumed by the feature builder
# ---------------------------------------------------------------------------


@dataclass
class GameRow:
    """One row per scheduled game, after merging sources.

    All teams use 3-letter Retrosheet codes (LAN, NYA, BOS...). 538 uses
    its own 3-letter scheme that's largely identical; we map a handful of
    historic exceptions in ``_normalize_team``.
    """

    game_date: date
    season: int
    home: str
    away: str
    home_score: Optional[int]
    away_score: Optional[int]
    # Pre-game 538 columns (None if game not in 538 archive).
    elo_prob_home: Optional[float] = None
    rating_prob_home: Optional[float] = None
    pitcher_rgs_home: Optional[float] = None
    pitcher_rgs_away: Optional[float] = None
    pitcher_adj_home: Optional[float] = None
    pitcher_adj_away: Optional[float] = None
    # Game-log extras (Retrosheet).
    park_id: Optional[str] = None
    day_night: Optional[str] = None
    home_pitcher_id: Optional[str] = None
    away_pitcher_id: Optional[str] = None
    plate_umpire_id: Optional[str] = None
    # Free-text weather (Retrosheet "weather" column when present).
    weather_text: Optional[str] = None
    # Retrosheet box-score extras (Week-8 expansion). All optional; None
    # when the source row lacks them (e.g. 538-only rows). These power the
    # pitcher-form / bullpen-fatigue / empirical-park features.
    home_pitchers_used: Optional[int] = None   # total pitchers the HOME team used
    away_pitchers_used: Optional[int] = None   # total pitchers the AWAY team used
    home_team_er: Optional[int] = None          # earned runs charged to HOME staff
    away_team_er: Optional[int] = None          # earned runs charged to AWAY staff
    home_so_pitching: Optional[int] = None      # strikeouts recorded by HOME staff
    away_so_pitching: Optional[int] = None      # strikeouts recorded by AWAY staff
    home_bb_pitching: Optional[int] = None       # walks issued by HOME staff
    away_bb_pitching: Optional[int] = None       # walks issued by AWAY staff
    home_hr_allowed: Optional[int] = None        # HR allowed by HOME staff
    away_hr_allowed: Optional[int] = None        # HR allowed by AWAY staff

    @property
    def home_won(self) -> Optional[bool]:
        if self.home_score is None or self.away_score is None:
            return None
        return self.home_score > self.away_score

    def as_dict(self) -> dict:
        return {
            "date": self.game_date.isoformat(),
            "season": self.season,
            "home": self.home,
            "away": self.away,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "elo_prob_home": self.elo_prob_home,
            "rating_prob_home": self.rating_prob_home,
            "pitcher_rgs_home": self.pitcher_rgs_home,
            "pitcher_rgs_away": self.pitcher_rgs_away,
            "pitcher_adj_home": self.pitcher_adj_home,
            "pitcher_adj_away": self.pitcher_adj_away,
            "park_id": self.park_id,
            "day_night": self.day_night,
            "home_pitcher_id": self.home_pitcher_id,
            "away_pitcher_id": self.away_pitcher_id,
            "plate_umpire_id": self.plate_umpire_id,
            "weather_text": self.weather_text,
            "home_pitchers_used": self.home_pitchers_used,
            "away_pitchers_used": self.away_pitchers_used,
            "home_team_er": self.home_team_er,
            "away_team_er": self.away_team_er,
            "home_so_pitching": self.home_so_pitching,
            "away_so_pitching": self.away_so_pitching,
            "home_bb_pitching": self.home_bb_pitching,
            "away_bb_pitching": self.away_bb_pitching,
            "home_hr_allowed": self.home_hr_allowed,
            "away_hr_allowed": self.away_hr_allowed,
            "home_won": None if self.home_won is None else int(self.home_won),
        }


# Cross-source team name harmonization. 538 mostly matches Retrosheet's
# 3-letter codes but a few legacy codes diverge.
_TEAM_ALIASES = {
    # 538 → Retrosheet
    "ANA": "ANA",  # Angels (Retrosheet uses "ANA" then "ANA" - same)
    "CHC": "CHN",
    "CHW": "CHA",
    "KCR": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "NYM": "NYN",
    "NYY": "NYA",
    "SDP": "SDN",
    "SFG": "SFN",
    "STL": "SLN",
    "TBD": "TBA",
    "TBR": "TBA",
    "WSN": "WAS",
    # Retrosheet → canonical (we use Retrosheet codes everywhere)
    "ANA": "ANA",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHA": "CHA",
    "CHN": "CHN",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCA": "KCA",
    "LAN": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYA": "NYA",
    "NYN": "NYN",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDN": "SDN",
    "SEA": "SEA",
    "SFN": "SFN",
    "SLN": "SLN",
    "TBA": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WAS": "WAS",
}


def _normalize_team(code: str) -> str:
    """Canonicalize team code to Retrosheet 3-letter scheme."""
    if not code:
        return ""
    code = code.strip().upper()
    return _TEAM_ALIASES.get(code, code)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def load_538_mlb_games(
    path: Path | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[GameRow]:
    """Load 538 MLB Elo rows as ``GameRow``s.

    The 538 archive has one row per game (NOT one per team-perspective).
    We assert that by uniqueness on (date, team1, team2).
    """
    p = path or ELO_538_CACHE
    if not p.exists():
        log.warning("538 MLB cache not present at %s — skipping", p)
        return []
    out: list[GameRow] = []
    with open(p) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            t1 = _normalize_team(row.get("team1") or "")
            t2 = _normalize_team(row.get("team2") or "")
            if not t1 or not t2:
                continue
            # 538 lists team1 as home (per their schema docs).
            out.append(
                GameRow(
                    game_date=d,
                    season=_safe_int(row.get("season")) or d.year,
                    home=t1,
                    away=t2,
                    home_score=_safe_int(row.get("score1")),
                    away_score=_safe_int(row.get("score2")),
                    elo_prob_home=_safe_float(row.get("elo_prob1")),
                    rating_prob_home=_safe_float(row.get("rating_prob1")),
                    pitcher_rgs_home=_safe_float(row.get("pitcher1_rgs")),
                    pitcher_rgs_away=_safe_float(row.get("pitcher2_rgs")),
                    pitcher_adj_home=_safe_float(row.get("pitcher1_adj")),
                    pitcher_adj_away=_safe_float(row.get("pitcher2_adj")),
                )
            )
    return out


def _download_retrosheet_zip(year: int, *, timeout: float = 30.0) -> bytes | None:
    """Fetch retrosheet gl{year}.zip with on-disk cache. Returns raw bytes."""
    RETROSHEET_CACHE.mkdir(parents=True, exist_ok=True)
    cached = RETROSHEET_CACHE / f"gl{year}.zip"
    if cached.exists():
        return cached.read_bytes()
    if httpx is None:
        log.warning("httpx unavailable; cannot fetch retrosheet")
        return None
    url = RETROSHEET_BASE.format(year=year)
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code != 200 or len(r.content) < 1000:
            log.warning("retrosheet %d returned %s (%d bytes)", year, r.status_code, len(r.content))
            return None
        cached.write_bytes(r.content)
        return r.content
    except Exception as e:  # noqa: BLE001
        log.warning("retrosheet %d fetch failed: %s", year, e)
        return None


# Retrosheet game log fields (0-indexed). See
# https://www.retrosheet.org/gamelogs/glfields.txt for the full spec.
RS_DATE = 0
RS_VISITING_TEAM = 3
RS_HOME_TEAM = 6
RS_VISITING_SCORE = 9
RS_HOME_SCORE = 10
RS_LENGTH_OUTS = 11
RS_DAY_NIGHT = 12
RS_PARK_ID = 16
RS_ATTENDANCE = 17
RS_GAME_DURATION = 18
RS_PLATE_UMP = 77  # umpire id, plate
RS_VISITING_STARTING_PITCHER_ID = 101
RS_HOME_STARTING_PITCHER_ID = 103

# Box-score offense/pitching fields (0-indexed). Verified empirically against
# gl2022.txt (see docs/mlb_feature_audit.md "Retrosheet field map"). The two
# line-score strings occupy 0idx 19 (visitor) and 0idx 20 (home); visitor
# batting/pitching runs 0idx 21..48, home 0idx 49..76.
RS_V_HR = 25             # HR hit by visiting batters (== HR allowed by HOME staff)
RS_V_BB = 30             # BB drawn by visiting batters (== BB issued by HOME staff)
RS_V_SO = 32             # SO by visiting batters (== SO recorded by HOME staff)
RS_V_PITCHERS_USED = 38  # number of pitchers the VISITING team used
RS_V_TEAM_ER = 40        # team earned runs charged to VISITING staff
RS_H_HR = 53             # HR hit by home batters (== HR allowed by AWAY staff)
RS_H_BB = 58             # BB drawn by home batters (== BB issued by AWAY staff)
RS_H_SO = 60             # SO by home batters (== SO recorded by AWAY staff)
RS_H_PITCHERS_USED = 66  # number of pitchers the HOME team used
RS_H_TEAM_ER = 68        # team earned runs charged to HOME staff


def load_retrosheet_games(
    year: int,
    *,
    timeout: float = 30.0,
    raw_zip: bytes | None = None,
) -> list[GameRow]:
    """Load all games for a single year from Retrosheet game logs.

    Returns one ``GameRow`` per game (deduped). ``home_pitcher_id`` and
    ``plate_umpire_id`` are the Retrosheet player-id strings (8-char).
    """
    raw = raw_zip if raw_zip is not None else _download_retrosheet_zip(year, timeout=timeout)
    if not raw:
        return []
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return []
    names = zf.namelist()
    if not names:
        return []
    txt = zf.read(names[0]).decode("latin-1", errors="ignore")
    out: list[GameRow] = []
    rdr = csv.reader(io.StringIO(txt))
    seen: set[tuple] = set()
    for fields in rdr:
        if not fields or len(fields) < RS_HOME_SCORE + 1:
            continue
        try:
            d = datetime.strptime(fields[RS_DATE], "%Y%m%d").date()
        except (ValueError, IndexError):
            continue
        home = _normalize_team(fields[RS_HOME_TEAM])
        away = _normalize_team(fields[RS_VISITING_TEAM])
        key = (d, home, away)
        if key in seen:
            # Doubleheader — keep both but disambiguate by score combination.
            continue
        seen.add(key)
        out.append(
            GameRow(
                game_date=d,
                season=year,
                home=home,
                away=away,
                home_score=_safe_int(fields[RS_HOME_SCORE]),
                away_score=_safe_int(fields[RS_VISITING_SCORE]),
                park_id=fields[RS_PARK_ID] if len(fields) > RS_PARK_ID else None,
                day_night=fields[RS_DAY_NIGHT] if len(fields) > RS_DAY_NIGHT else None,
                home_pitcher_id=(
                    fields[RS_HOME_STARTING_PITCHER_ID]
                    if len(fields) > RS_HOME_STARTING_PITCHER_ID
                    else None
                ),
                away_pitcher_id=(
                    fields[RS_VISITING_STARTING_PITCHER_ID]
                    if len(fields) > RS_VISITING_STARTING_PITCHER_ID
                    else None
                ),
                plate_umpire_id=(
                    fields[RS_PLATE_UMP].strip("\"") if len(fields) > RS_PLATE_UMP else None
                ),
                # A team's PITCHING allowed = the OPPONENT's batting produced.
                # Home staff faced visiting batters: home_* allowed == V_* fields.
                home_pitchers_used=(
                    _safe_int(fields[RS_H_PITCHERS_USED]) if len(fields) > RS_H_PITCHERS_USED else None
                ),
                away_pitchers_used=(
                    _safe_int(fields[RS_V_PITCHERS_USED]) if len(fields) > RS_V_PITCHERS_USED else None
                ),
                home_team_er=(
                    _safe_int(fields[RS_H_TEAM_ER]) if len(fields) > RS_H_TEAM_ER else None
                ),
                away_team_er=(
                    _safe_int(fields[RS_V_TEAM_ER]) if len(fields) > RS_V_TEAM_ER else None
                ),
                home_so_pitching=(
                    _safe_int(fields[RS_V_SO]) if len(fields) > RS_V_SO else None
                ),
                away_so_pitching=(
                    _safe_int(fields[RS_H_SO]) if len(fields) > RS_H_SO else None
                ),
                home_bb_pitching=(
                    _safe_int(fields[RS_V_BB]) if len(fields) > RS_V_BB else None
                ),
                away_bb_pitching=(
                    _safe_int(fields[RS_H_BB]) if len(fields) > RS_H_BB else None
                ),
                home_hr_allowed=(
                    _safe_int(fields[RS_V_HR]) if len(fields) > RS_V_HR else None
                ),
                away_hr_allowed=(
                    _safe_int(fields[RS_H_HR]) if len(fields) > RS_H_HR else None
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Rolling team rates — strict walk-forward
# ---------------------------------------------------------------------------


@dataclass
class RollingTeamFeatures:
    """Per-team rolling features at a moment in time.

    Lists hold the most recent N completed-game runs. Helper methods
    return windowed averages safely (None when sample too small).
    """

    runs_scored: deque = field(default_factory=lambda: deque(maxlen=30))
    runs_allowed: deque = field(default_factory=lambda: deque(maxlen=30))
    won: deque = field(default_factory=lambda: deque(maxlen=30))
    # Week-8 expansion: rolling bullpen usage (pitchers used per game) as a
    # fatigue proxy. A team that has burned its pen over the last few games
    # goes into the next game short-handed.
    pitchers_used: deque = field(default_factory=lambda: deque(maxlen=30))
    last_game_date: Optional[date] = None

    def update(
        self,
        *,
        rs: int,
        ra: int,
        won: bool,
        on_date: date,
        pitchers_used: Optional[int] = None,
    ) -> None:
        self.runs_scored.append(rs)
        self.runs_allowed.append(ra)
        self.won.append(1 if won else 0)
        if pitchers_used is not None:
            self.pitchers_used.append(int(pitchers_used))
        self.last_game_date = on_date

    def _avg(self, dq: deque, n: int) -> Optional[float]:
        if len(dq) < n:
            return None
        # Most recent n entries
        recent = list(dq)[-n:]
        return sum(recent) / n

    def rs(self, n: int) -> Optional[float]:
        return self._avg(self.runs_scored, n)

    def ra(self, n: int) -> Optional[float]:
        return self._avg(self.runs_allowed, n)

    def win_pct(self, n: int) -> Optional[float]:
        return self._avg(self.won, n)

    def bullpen_load(self, n: int) -> Optional[float]:
        """Avg pitchers used over the last ``n`` games (bullpen-fatigue proxy)."""
        return self._avg(self.pitchers_used, n)


def fit_rolling_rates(
    games: list[GameRow],
    *,
    season_reset: bool = True,
) -> dict[date, dict[str, RollingTeamFeatures]]:
    """Build a snapshot of every team's rolling features just BEFORE each game date.

    Returns a mapping ``{date → {team_code → RollingTeamFeatures}}``. The
    features at key ``d`` reflect ALL games strictly before ``d``. To get
    features for a game on date ``d`` look up ``snapshots[d][team]``.

    Important: this is single-pass O(n_games). Each date snapshot is a
    shallow copy of the per-team state — copies use a frozen snapshot of
    the deque contents so callers can safely keep references.
    """
    state: dict[str, RollingTeamFeatures] = defaultdict(RollingTeamFeatures)
    last_season_seen: dict[str, int] = {}
    snapshots: dict[date, dict[str, RollingTeamFeatures]] = {}
    last_date: Optional[date] = None

    # Sort defensively — caller may not have.
    for g in sorted(games, key=lambda r: r.game_date):
        # Season-reset BEFORE snapshotting so the snapshot reflects the
        # fresh-season state, not last season's carry-over.
        if season_reset:
            for team in (g.home, g.away):
                if last_season_seen.get(team) is not None and last_season_seen[team] < g.season:
                    state[team] = RollingTeamFeatures()
                last_season_seen[team] = g.season

        # When we cross a date boundary, freeze the snapshot for the new date.
        if g.game_date != last_date:
            # Snapshot all teams' state AS-OF this date.
            snapshots[g.game_date] = _freeze_snapshot(state)
            last_date = g.game_date

        if g.home_score is None or g.away_score is None:
            continue

        state[g.home].update(
            rs=g.home_score, ra=g.away_score, won=g.home_score > g.away_score,
            on_date=g.game_date, pitchers_used=g.home_pitchers_used,
        )
        state[g.away].update(
            rs=g.away_score, ra=g.home_score, won=g.away_score > g.home_score,
            on_date=g.game_date, pitchers_used=g.away_pitchers_used,
        )

    return snapshots


def _freeze_snapshot(state: dict[str, RollingTeamFeatures]) -> dict[str, RollingTeamFeatures]:
    """Shallow copy of per-team state with deques materialized to tuples."""
    out: dict[str, RollingTeamFeatures] = {}
    for team, f in state.items():
        # Build a fresh RollingTeamFeatures with deques copied.
        c = RollingTeamFeatures()
        c.runs_scored = deque(list(f.runs_scored), maxlen=30)
        c.runs_allowed = deque(list(f.runs_allowed), maxlen=30)
        c.won = deque(list(f.won), maxlen=30)
        c.pitchers_used = deque(list(f.pitchers_used), maxlen=30)
        c.last_game_date = f.last_game_date
        out[team] = c
    return out


# ---------------------------------------------------------------------------
# Pitcher days-rest
# ---------------------------------------------------------------------------


def compute_pitcher_rest(games: list[GameRow]) -> dict[tuple[date, str], int]:
    """Days since a starting pitcher's previous appearance.

    Returns ``{(game_date, pitcher_id) → days_rest}``. Pitcher IDs from
    Retrosheet (8-char). ``None`` pitcher_ids are skipped. A pitcher's
    first observed appearance gets days_rest = 6 (league-average season
    opener).
    """
    last_start: dict[str, date] = {}
    rest: dict[tuple[date, str], int] = {}
    for g in sorted(games, key=lambda r: r.game_date):
        for pid in (g.home_pitcher_id, g.away_pitcher_id):
            if not pid:
                continue
            prior = last_start.get(pid)
            rest[(g.game_date, pid)] = (g.game_date - prior).days if prior else 6
            last_start[pid] = g.game_date
    return rest


# ---------------------------------------------------------------------------
# Starting-pitcher rolling form  (Week-8 expansion)
# ---------------------------------------------------------------------------

# We don't have per-pitcher line-level splits from the free game logs, so we
# attribute the team's staff box-score line to the STARTER as a proxy for
# recent form. This is deliberately coarse — it captures "was the team's run
# prevention good in the games this guy started", which is a strictly-prior,
# no-leakage signal that correlates with starter quality. Documented as a
# proxy in the audit; a Statcast per-pitcher pull is the Phase-2 upgrade.


@dataclass
class PitcherForm:
    """Rolling box-score-attributed form for one starting pitcher."""

    er: deque = field(default_factory=lambda: deque(maxlen=10))
    so: deque = field(default_factory=lambda: deque(maxlen=10))
    bb: deque = field(default_factory=lambda: deque(maxlen=10))
    hr: deque = field(default_factory=lambda: deque(maxlen=10))
    last_start: Optional[date] = None

    def _avg(self, dq: deque, n: int) -> Optional[float]:
        if len(dq) < n:
            return None
        recent = list(dq)[-n:]
        return sum(recent) / n

    def er_l(self, n: int) -> Optional[float]:
        return self._avg(self.er, n)

    def k_minus_bb_l(self, n: int) -> Optional[float]:
        """Avg (SO - BB) over last n starts — a coarse command/stuff proxy."""
        if len(self.so) < n or len(self.bb) < n:
            return None
        so = list(self.so)[-n:]
        bb = list(self.bb)[-n:]
        return sum(so[i] - bb[i] for i in range(n)) / n

    def hr_l(self, n: int) -> Optional[float]:
        return self._avg(self.hr, n)


def fit_pitcher_form(
    games: list[GameRow],
    *,
    window: int = 5,
) -> dict[tuple[date, str], PitcherForm]:
    """Build strictly-prior rolling form snapshots keyed by (game_date, pitcher_id).

    The snapshot at ``(d, pid)`` reflects only starts strictly before ``d``.
    We attribute the team's staff pitching line (ER/SO/BB/HR) to the game's
    starter as a coarse form proxy. Games missing box-score fields or a
    pitcher id are skipped for that side.
    """
    state: dict[str, PitcherForm] = defaultdict(PitcherForm)
    last_season: dict[str, int] = {}
    out: dict[tuple[date, str], PitcherForm] = {}

    for g in sorted(games, key=lambda r: r.game_date):
        sides = (
            (g.home_pitcher_id, g.home_team_er, g.home_so_pitching,
             g.home_bb_pitching, g.home_hr_allowed),
            (g.away_pitcher_id, g.away_team_er, g.away_so_pitching,
             g.away_bb_pitching, g.away_hr_allowed),
        )
        for pid, er, so, bb, hr in sides:
            if not pid:
                continue
            # Season reset so we don't carry last year's form.
            if last_season.get(pid) is not None and last_season[pid] < g.season:
                state[pid] = PitcherForm()
            last_season[pid] = g.season
            # Snapshot BEFORE updating (strictly prior).
            f = state[pid]
            out[(g.game_date, pid)] = _freeze_pitcher_form(f)
            if er is not None:
                f.er.append(int(er))
            if so is not None:
                f.so.append(int(so))
            if bb is not None:
                f.bb.append(int(bb))
            if hr is not None:
                f.hr.append(int(hr))
            f.last_start = g.game_date
    return out


def _freeze_pitcher_form(f: PitcherForm) -> PitcherForm:
    c = PitcherForm()
    c.er = deque(list(f.er), maxlen=10)
    c.so = deque(list(f.so), maxlen=10)
    c.bb = deque(list(f.bb), maxlen=10)
    c.hr = deque(list(f.hr), maxlen=10)
    c.last_start = f.last_start
    return c


# ---------------------------------------------------------------------------
# Empirical park run environment  (Week-8 expansion)
# ---------------------------------------------------------------------------


def fit_empirical_park_factor(
    games: list[GameRow],
    *,
    prior_runs: float = 4.5,
    prior_weight: int = 40,
) -> dict[tuple[date, str], float]:
    """Leakage-safe expanding-window park run environment.

    Returns ``{(game_date, park_id) -> total_runs_per_game}`` where the value
    at ``(d, park)`` uses ONLY games at that park strictly before ``d``.
    Smoothed toward a league prior (``prior_runs`` per team => 2*prior_runs
    total) with ``prior_weight`` pseudo-games so early-season parks aren't
    noisy. This replaces / augments the static data/mlb_parks.json table with
    an in-sample-honest measurement.
    """
    from collections import defaultdict as _dd

    running_total: dict[str, float] = _dd(float)
    running_n: dict[str, int] = _dd(int)
    out: dict[tuple[date, str], float] = {}
    prior_total = 2.0 * prior_runs
    for g in sorted(games, key=lambda r: r.game_date):
        if not g.park_id:
            continue
        n = running_n[g.park_id]
        tot = running_total[g.park_id]
        est = (tot + prior_total * prior_weight) / (n + prior_weight)
        out[(g.game_date, g.park_id)] = est
        if g.home_score is not None and g.away_score is not None:
            running_total[g.park_id] += g.home_score + g.away_score
            running_n[g.park_id] += 1
    return out


# ---------------------------------------------------------------------------
# Rolling strength prior  (Week-8 expansion) — replaces defunct 538 Elo
# ---------------------------------------------------------------------------


# Home-field advantage baseline (pp above 50%) and run-diff -> log-odds slope.
# ~0.083 maps a +1 run/game rolling edge to roughly +2pp win prob, in line
# with pythagorean run-to-win conversion at MLB scoring levels.
HOME_FIELD_PP = 0.035
RUN_DIFF_COEF = 0.083


def strength_prior_home(
    home_rd10: Optional[float],
    away_rd10: Optional[float],
    *,
    home_field_pp: float = HOME_FIELD_PP,
    run_diff_coef: float = RUN_DIFF_COEF,
) -> Optional[float]:
    """Cheap, leakage-safe pre-game win-prob prior from rolling run diff.

    538's Elo/rating columns are gone (the archive 404s since 2023-10-01), so
    the Week-8 pipeline synthesizes an honest, strictly-prior baseline prob
    from each team's rolling L10 run differential + a fixed home-field bump.
    This is NOT a market line and NOT a real closing price — it is a
    model-internal prior. The simulator labels the resulting market proxy
    loudly as a Retrosheet-derived prior, not a book close.

    Returns ``None`` when either team's rolling sample is too small.
    """
    if home_rd10 is None or away_rd10 is None:
        return None
    base = math.log((0.5 + home_field_pp) / (0.5 - home_field_pp))
    z = base + run_diff_coef * (home_rd10 - away_rd10)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Park run environment — static lookup from data/mlb_parks.json
# ---------------------------------------------------------------------------


def load_park_run_env(path: Path | None = None) -> dict[str, float]:
    """Park ID → baseline runs-per-team-game."""
    p = path or PARKS_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    out: dict[str, float] = {}
    for k, v in (data.get("venues") or {}).items():
        rg = v.get("run_env_baseline")
        if rg is None:
            continue
        # Index by both the venue display name AND by short codes the
        # park may also be known under (Retrosheet park_id, etc).
        out[k] = float(rg)
        for alias in v.get("aliases") or []:
            out[alias] = float(rg)
        rs_pid = v.get("retrosheet_park_id")
        if rs_pid:
            out[rs_pid] = float(rg)
    return out


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


# All numeric features the model consumes, in canonical order. Order is
# stable so persisted coefficient arrays can be matched up positionally.
FEATURE_NAMES: list[str] = [
    # Baseline prior (replaces defunct 538 elo_prob_home / rating_prob_home).
    "prior_prob_home",
    # Legacy 538 pitcher-rating diffs (None on Retrosheet-only rows; kept in
    # the vector for backward compat, fill to 0 when absent).
    "pitcher_rgs_diff",
    "pitcher_adj_diff",
    # Rolling team offense / defense.
    "rs_l3_diff",
    "rs_l5_diff",
    "rs_l10_diff",
    "rs_l20_diff",
    "ra_l3_diff",
    "ra_l5_diff",
    "ra_l10_diff",
    "ra_l20_diff",
    "win_pct_l10_diff",
    "run_diff_l10_diff",
    "pitcher_rest_diff",
    "is_day_game",
    # ---- Week-8 expansion: pitcher / bullpen / park levers ----
    "sp_er_l3_diff",        # starter rolling earned-runs form (lower = better)
    "sp_kbb_l5_diff",       # starter rolling (SO-BB) command/stuff
    "sp_hr_l5_diff",        # starter rolling HR-allowed tendency
    "bullpen_load_l3_diff", # recent pitchers-used (fatigue) home - away
    "park_run_env_emp",     # empirical (expanding-window) park run env
]


def build_features(
    game: GameRow,
    snapshots: dict[date, dict[str, RollingTeamFeatures]],
    pitcher_rest: dict[tuple[date, str], int],
    park_run_env: dict[str, float],
    *,
    pitcher_form: dict[tuple[date, str], "PitcherForm"] | None = None,
    park_factor_emp: dict[tuple[date, str], float] | None = None,
) -> Optional[dict]:
    """Build the per-game feature dict, strictly walk-forward.

    Returns ``None`` if any **required** feature is missing (no rolling
    strength prior, no sufficient rolling sample). Optional features (park,
    day/night, pitcher form, bullpen) default to neutral values.
    """
    pitcher_form = pitcher_form or {}
    park_factor_emp = park_factor_emp or {}
    # Leakage gate — snapshot keyed at the game's date should reflect
    # only PRIOR games. ``fit_rolling_rates`` enforces this.
    snap = snapshots.get(game.game_date) or {}
    home_f = snap.get(game.home) or RollingTeamFeatures()
    away_f = snap.get(game.away) or RollingTeamFeatures()
    assert home_f.last_game_date is None or home_f.last_game_date < game.game_date, (
        f"leakage: home team {game.home} last_game_date {home_f.last_game_date} "
        f">= game date {game.game_date}"
    )
    assert away_f.last_game_date is None or away_f.last_game_date < game.game_date, (
        f"leakage: away team {game.away} last_game_date {away_f.last_game_date} "
        f">= game date {game.game_date}"
    )

    # Rolling diffs — home minus away. None when sample size too small.
    def _diff(fn_h, fn_a, n: int) -> Optional[float]:
        a = fn_h(n)
        b = fn_a(n)
        if a is None or b is None:
            return None
        return a - b

    # Rolling L10 run differential per team (used both as a feature and to
    # synthesize the strength prior that replaces the defunct 538 Elo).
    h_rs10 = home_f.rs(10)
    h_ra10 = home_f.ra(10)
    a_rs10 = away_f.rs(10)
    a_ra10 = away_f.ra(10)
    home_rd10 = (h_rs10 - h_ra10) if (h_rs10 is not None and h_ra10 is not None) else None
    away_rd10 = (a_rs10 - a_ra10) if (a_rs10 is not None and a_ra10 is not None) else None
    prior = strength_prior_home(home_rd10, away_rd10)

    feats: dict = {
        "prior_prob_home": prior,
        "pitcher_rgs_diff": (
            (game.pitcher_rgs_home or 0) - (game.pitcher_rgs_away or 0)
            if game.pitcher_rgs_home is not None and game.pitcher_rgs_away is not None
            else None
        ),
        "pitcher_adj_diff": (
            (game.pitcher_adj_home or 0) - (game.pitcher_adj_away or 0)
            if game.pitcher_adj_home is not None and game.pitcher_adj_away is not None
            else None
        ),
        "rs_l3_diff": _diff(home_f.rs, away_f.rs, 3),
        "rs_l5_diff": _diff(home_f.rs, away_f.rs, 5),
        "rs_l10_diff": _diff(home_f.rs, away_f.rs, 10),
        "rs_l20_diff": _diff(home_f.rs, away_f.rs, 20),
        "ra_l3_diff": _diff(home_f.ra, away_f.ra, 3),
        "ra_l5_diff": _diff(home_f.ra, away_f.ra, 5),
        "ra_l10_diff": _diff(home_f.ra, away_f.ra, 10),
        "ra_l20_diff": _diff(home_f.ra, away_f.ra, 20),
        "win_pct_l10_diff": _diff(home_f.win_pct, away_f.win_pct, 10),
        "run_diff_l10_diff": (
            (home_rd10 - away_rd10)
            if (home_rd10 is not None and away_rd10 is not None) else None
        ),
        "pitcher_rest_diff": None,
        "is_day_game": 1.0 if (game.day_night or "").upper().startswith("D") else 0.0,
        # Week-8 expansion features (default None -> filled to 0 in vector).
        "sp_er_l3_diff": None,
        "sp_kbb_l5_diff": None,
        "sp_hr_l5_diff": None,
        "bullpen_load_l3_diff": None,
        "park_run_env_emp": None,
    }

    # Pitcher rest diff (home_rest - away_rest)
    h_rest = pitcher_rest.get((game.game_date, game.home_pitcher_id or ""))
    a_rest = pitcher_rest.get((game.game_date, game.away_pitcher_id or ""))
    if h_rest is not None and a_rest is not None:
        feats["pitcher_rest_diff"] = float(h_rest - a_rest)

    # ---- Week-8: starting-pitcher rolling form diffs (home - away) ----
    hp = pitcher_form.get((game.game_date, game.home_pitcher_id or ""))
    ap = pitcher_form.get((game.game_date, game.away_pitcher_id or ""))
    if hp is not None and ap is not None:
        h_er3, a_er3 = hp.er_l(3), ap.er_l(3)
        if h_er3 is not None and a_er3 is not None:
            # Negate so a POSITIVE feature means the home starter has been
            # allowing FEWER earned runs (i.e. is in better form).
            feats["sp_er_l3_diff"] = a_er3 - h_er3
        h_kbb5, a_kbb5 = hp.k_minus_bb_l(5), ap.k_minus_bb_l(5)
        if h_kbb5 is not None and a_kbb5 is not None:
            feats["sp_kbb_l5_diff"] = h_kbb5 - a_kbb5
        h_hr5, a_hr5 = hp.hr_l(5), ap.hr_l(5)
        if h_hr5 is not None and a_hr5 is not None:
            # Negate: positive => home starter allows fewer HR.
            feats["sp_hr_l5_diff"] = a_hr5 - h_hr5

    # ---- Week-8: bullpen fatigue (recent pitchers used, home - away) ----
    h_bp = home_f.bullpen_load(3)
    a_bp = away_f.bullpen_load(3)
    if h_bp is not None and a_bp is not None:
        # Positive => home pen has been MORE taxed recently (a negative for
        # home). We keep the raw home-minus-away sign; the model learns it.
        feats["bullpen_load_l3_diff"] = h_bp - a_bp

    # ---- Week-8: empirical park run environment ----
    pf = park_factor_emp.get((game.game_date, game.park_id or ""))
    if pf is not None:
        feats["park_run_env_emp"] = pf
    elif game.park_id and game.park_id in park_run_env:
        # Fall back to the static table if no empirical value yet.
        feats["park_run_env_emp"] = park_run_env[game.park_id]

    # Required features: the rolling strength prior (needs L10 rates on both
    # teams). Without it we have no baseline and no market proxy.
    if feats["prior_prob_home"] is None:
        return None
    # Need at least L10 rolling rates so the rolling features carry weight.
    if feats["rs_l10_diff"] is None:
        return None

    return feats


def feature_vector(feats: dict, fill_value: float = 0.0) -> list[float]:
    """Convert a feature dict to a positional vector in ``FEATURE_NAMES`` order."""
    return [
        float(feats[name]) if feats.get(name) is not None else fill_value
        for name in FEATURE_NAMES
    ]
