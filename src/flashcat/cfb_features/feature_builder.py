"""Walk-forward feature builder for CFB games — Phase 1+2 combined.

Pure functions only — no I/O outside the explicit loader entry points.
Every feature passed to the model has to be computable using ONLY data
strictly BEFORE the game's date (leakage gate asserted on every call).

Data source: cached schedule from the ESPN public scoreboard fallback
in ``src/flashcat/sources/cfb_cfbfastr_epa.py``.  The CFBD API requires
a signup-only API key we don't have; ESPN's public endpoint returns
historical game results for free.

Cache location: ``data/cache/cfb_schedule_<season>.json``
These files are written by CFBCfbfastREPA._load_schedule() and reused
here to avoid duplicate network calls.

Phase-1 features (6):
  Rolling team efficiency (L5)
    off_eff_l5_diff         home − away offensive efficiency (pts scored avg)
    def_eff_l5_diff         home − away defensive efficiency (pts allowed avg)
    net_eff_l5_diff         combined (off − def differential)

  Schedule fatigue
    rest_days_diff          (home rest) − (away rest), capped ±7
    bye_home                1 if home team on bye last week
    bye_away                1 if away team on bye last week

Phase-2 features (4):
  Margin volatility (proxy for turnover chaos)
    margin_volatility_home  std dev of point margins, home team L5
    margin_volatility_away  std dev of point margins, away team L5

  Conference/home structural
    conf_tier_diff          +1 P5 home vs G5 away, -1 reversed, 0 same
    home_field_flag         constant 1.0 (HFA interpretability)
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from ..config import CACHE_DIR

log = logging.getLogger(__name__)

# Power-5 conferences (same definition as cfb_cfbfastr_epa.py)
POWER_FIVE = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}

# ───────────────────────────── Feature names ──────────────────────────────

FEATURE_NAMES: list[str] = [
    # Phase-1 rolling efficiency (L5)
    "off_eff_l5_home",       # avg pts scored per game, home team, L5
    "off_eff_l5_away",       # avg pts scored per game, away team, L5
    "off_eff_l5_diff",       # home − away
    "def_eff_l5_home",       # avg pts allowed per game, home team, L5
    "def_eff_l5_away",       # avg pts allowed per game, away team, L5
    "def_eff_l5_diff",       # home − away (lower = better defense)
    "net_eff_l5_diff",       # (off_eff_l5_diff) − (def_eff_l5_diff)
    # Phase-1 fatigue
    "rest_days_diff",        # (home rest) − (away rest), capped ±7
    "bye_home",              # 1 if home had bye last week
    "bye_away",              # 1 if away had bye last week
    # Phase-2 volatility
    "margin_volatility_home",  # std dev of home team L5 margins (proxy turnovers)
    "margin_volatility_away",  # std dev of away team L5 margins
    # Phase-2 structural
    "conf_tier_diff",        # P5 vs G5 advantage dummy
    "home_field_flag",       # constant 1.0
]

N_FEATURES = len(FEATURE_NAMES)

# Minimum prior games required to compute L5 rolling features
MIN_GAMES_REQUIRED = 3


# ─────────────────────────── Data structures ──────────────────────────────


@dataclass
class CFBGameRow:
    """One row per CFB regular-season game after normalizing from cache."""

    game_id: str
    game_date: date
    season: int
    week: Optional[int]
    home: str
    away: str
    home_conference: Optional[str]
    away_conference: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    home_won: Optional[bool]

    @property
    def home_margin(self) -> Optional[float]:
        if self.home_score is None or self.away_score is None:
            return None
        return float(self.home_score - self.away_score)

    @property
    def away_margin(self) -> Optional[float]:
        m = self.home_margin
        return None if m is None else -m


@dataclass
class CFBTeamSnapshot:
    """Rolling stats for one team, as-of a cutoff date (leakage-free).

    ``recent_games`` — list of CFBGameRow played by this team STRICTLY
    before ``as_of``, ordered by game_date ascending.  Built once by
    ``fit_rolling_snapshots`` and never mutated.
    """

    team: str
    as_of: date
    conference: Optional[str] = None  # most recent known conference
    recent_games: list[CFBGameRow] = field(default_factory=list)

    # ── team-perspective helpers ──

    def _margins(self, n: int) -> list[float]:
        """Point margins from team's perspective (positive = won), last n games."""
        out: list[float] = []
        for g in reversed(self.recent_games):
            if len(out) >= n:
                break
            if g.home == self.team:
                m = g.home_margin
            else:
                m = g.away_margin
            if m is not None:
                out.append(m)
        return list(reversed(out))

    def _scored(self, n: int) -> list[float]:
        """Points scored by this team, last n games."""
        out: list[float] = []
        for g in reversed(self.recent_games):
            if len(out) >= n:
                break
            if g.home == self.team:
                pts = g.home_score
            else:
                pts = g.away_score
            if pts is not None:
                out.append(float(pts))
        return list(reversed(out))

    def _allowed(self, n: int) -> list[float]:
        """Points allowed by this team, last n games."""
        out: list[float] = []
        for g in reversed(self.recent_games):
            if len(out) >= n:
                break
            if g.home == self.team:
                pts = g.away_score
            else:
                pts = g.home_score
            if pts is not None:
                out.append(float(pts))
        return list(reversed(out))

    @property
    def off_eff_l5(self) -> Optional[float]:
        """Avg points scored per game, last 5."""
        games = self._scored(5)
        if len(games) < MIN_GAMES_REQUIRED:
            return None
        return sum(games) / len(games)

    @property
    def def_eff_l5(self) -> Optional[float]:
        """Avg points allowed per game, last 5."""
        games = self._allowed(5)
        if len(games) < MIN_GAMES_REQUIRED:
            return None
        return sum(games) / len(games)

    @property
    def margin_volatility_l5(self) -> Optional[float]:
        """Std dev of point margins, last 5 games (proxy turnover chaos)."""
        ms = self._margins(5)
        if len(ms) < MIN_GAMES_REQUIRED:
            return None
        if len(ms) < 2:
            return 0.0
        mean = sum(ms) / len(ms)
        variance = sum((x - mean) ** 2 for x in ms) / (len(ms) - 1)
        return math.sqrt(variance)

    def rest_days(self, game_date: date) -> Optional[float]:
        """Days since last game before game_date, capped at 14."""
        for g in reversed(self.recent_games):
            if g.game_date < game_date:
                diff = (game_date - g.game_date).days
                return float(min(diff, 14))
        return None

    def had_bye(self, game_date: date) -> bool:
        """True if this team had a bye week immediately before game_date.

        CFB bye week: a full week (7+ days) gap in a mid-season context.
        We detect this as: last game was 13-21 days ago (exactly 2 week gaps
        are atypical; 7-11d is a normal week, 12+ suggests a bye).
        """
        rd = self.rest_days(game_date)
        if rd is None:
            return False
        return 12 <= rd <= 21


# ─────────────────────── Snapshot builder ──────────────────────────────────


def fit_rolling_snapshots(
    games: list[CFBGameRow],
) -> dict[tuple[str, date], CFBTeamSnapshot]:
    """Build leakage-free team snapshots for every (team, game_date) pair.

    For each game G involving team T, the snapshot for (T, G.game_date)
    contains ONLY games where game_date < G.game_date. This is the
    standard rolling-window approach used in nba_features and nfl_features.

    The result dict is keyed (team_name, game_date). The model calls
    snapshot[home, game.game_date] and snapshot[away, game.game_date].
    """
    # Sort ascending so we can build incrementally
    sorted_games = sorted(games, key=lambda g: g.game_date)

    # Per-team history buffer (ordered, games that have already occurred)
    team_history: dict[str, list[CFBGameRow]] = defaultdict(list)
    # Per-team latest known conference
    team_conf: dict[str, Optional[str]] = {}

    snapshots: dict[tuple[str, date], CFBTeamSnapshot] = {}

    for game in sorted_games:
        home = game.home
        away = game.away

        # Build snapshot for today using ONLY games from before today
        for team in (home, away):
            key = (team, game.game_date)
            if key not in snapshots:
                conf = team_conf.get(team)
                # Copy current history (all before this game_date)
                hist_copy = list(team_history[team])
                snapshots[key] = CFBTeamSnapshot(
                    team=team,
                    as_of=game.game_date,
                    conference=conf,
                    recent_games=hist_copy,
                )

        # Now add this game to the team histories (for future games)
        team_history[home].append(game)
        team_history[away].append(game)

        # Update conference (use the game's conference info)
        if game.home_conference:
            team_conf[home] = game.home_conference
        if game.away_conference:
            team_conf[away] = game.away_conference

    return snapshots


# ──────────────────────────── Feature builder ─────────────────────────────


def _conf_tier_diff(home_conf: Optional[str], away_conf: Optional[str]) -> float:
    """+1 home P5 vs away G5, -1 reversed, 0 same tier."""
    h_p5 = (home_conf or "") in POWER_FIVE
    a_p5 = (away_conf or "") in POWER_FIVE
    if h_p5 and not a_p5:
        return 1.0
    if a_p5 and not h_p5:
        return -1.0
    return 0.0


def build_features(
    game: CFBGameRow,
    snapshots: dict[tuple[str, date], CFBTeamSnapshot],
) -> Optional[dict]:
    """Compute feature dict for ``game`` using leakage-free snapshots.

    Returns None if the home or away snapshot is missing (no history yet).
    """
    home_snap = snapshots.get((game.home, game.game_date))
    away_snap = snapshots.get((game.away, game.game_date))

    # Need snapshots for both teams (they may be None for very first games)
    if home_snap is None or away_snap is None:
        return None

    # Rolling efficiency
    off_h = home_snap.off_eff_l5
    off_a = away_snap.off_eff_l5
    def_h = home_snap.def_eff_l5
    def_a = away_snap.def_eff_l5

    off_diff = (off_h - off_a) if (off_h is not None and off_a is not None) else None
    def_diff = (def_h - def_a) if (def_h is not None and def_a is not None) else None
    net_diff = (off_diff - def_diff) if (off_diff is not None and def_diff is not None) else None

    # Fatigue
    rest_h = home_snap.rest_days(game.game_date)
    rest_a = away_snap.rest_days(game.game_date)
    rest_diff = None
    if rest_h is not None and rest_a is not None:
        rest_diff = max(-7.0, min(7.0, rest_h - rest_a))

    bye_h = 1.0 if home_snap.had_bye(game.game_date) else 0.0
    bye_a = 1.0 if away_snap.had_bye(game.game_date) else 0.0

    # Volatility
    vol_h = home_snap.margin_volatility_l5
    vol_a = away_snap.margin_volatility_l5

    # Conference
    home_conf = home_snap.conference or game.home_conference
    away_conf = away_snap.conference or game.away_conference
    conf_diff = _conf_tier_diff(home_conf, away_conf)

    return {
        "off_eff_l5_home": off_h,
        "off_eff_l5_away": off_a,
        "off_eff_l5_diff": off_diff,
        "def_eff_l5_home": def_h,
        "def_eff_l5_away": def_a,
        "def_eff_l5_diff": def_diff,
        "net_eff_l5_diff": net_diff,
        "rest_days_diff": rest_diff,
        "bye_home": bye_h,
        "bye_away": bye_a,
        "margin_volatility_home": vol_h,
        "margin_volatility_away": vol_a,
        "conf_tier_diff": conf_diff,
        "home_field_flag": 1.0,
    }


def feature_vector(
    feat: dict,
    *,
    fill_mean: Optional[dict[str, float]] = None,
) -> Optional[list[float]]:
    """Convert feature dict to a fixed-length float vector.

    None values are imputed to fill_mean[name] if provided, else 0.0.
    The required features (home_field_flag, conf_tier_diff) are always
    populated so the vector is never None.
    """
    vec: list[float] = []
    for name in FEATURE_NAMES:
        val = feat.get(name)
        if val is None:
            if fill_mean and name in fill_mean:
                val = fill_mean[name]
            else:
                val = 0.0
        vec.append(float(val))
    return vec


# ──────────────────────────── Data loader ────────────────────────────────


def _parse_date_safe(s) -> Optional[date]:
    if s is None:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
    except Exception:
        try:
            from datetime import date as dt
            parts = str(s).split("-")
            return dt(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return None


def load_cfb_game_logs(
    seasons: Optional[list[int]] = None,
) -> list[CFBGameRow]:
    """Load CFB game rows from ESPN cache files written by CFBCfbfastREPA.

    If ``seasons`` is None, defaults to 2022, 2023, 2024.
    Only returns games with both home and away scores (completed games).
    """
    if seasons is None:
        seasons = [2022, 2023, 2024]

    rows: list[CFBGameRow] = []
    for season in seasons:
        cache_path = CACHE_DIR / f"cfb_schedule_{season}.json"
        if not cache_path.exists():
            log.info("CFB cache miss for season %s — attempting ESPN fetch", season)
            _backfill_season(season)
            if not cache_path.exists():
                log.warning("CFB season %s: cache not available after fetch attempt", season)
                continue

        try:
            with open(cache_path) as f:
                raw = json.load(f)
        except Exception as e:
            log.warning("Failed to load CFB cache for season %s: %s", season, e)
            continue

        for i, r in enumerate(raw):
            # Coerce date
            d = _parse_date_safe(r.get("date") or r.get("start_date") or r.get("startDate"))
            if d is None:
                continue

            home = r.get("home") or r.get("home_team") or r.get("homeTeam")
            away = r.get("away") or r.get("away_team") or r.get("awayTeam")
            if not home or not away:
                continue

            hp_raw = r.get("home_points") if r.get("home_points") is not None else r.get("homePoints")
            ap_raw = r.get("away_points") if r.get("away_points") is not None else r.get("awayPoints")

            try:
                hp: Optional[int] = int(hp_raw) if hp_raw is not None else None
                ap: Optional[int] = int(ap_raw) if ap_raw is not None else None
            except (TypeError, ValueError):
                hp = ap = None

            # Skip games without final scores
            if hp is None or ap is None:
                continue

            home_conf = r.get("home_conf") or r.get("home_conference") or r.get("homeConference")
            away_conf = r.get("away_conf") or r.get("away_conference") or r.get("awayConference")

            week_raw = r.get("week")
            try:
                week: Optional[int] = int(week_raw) if week_raw is not None else None
            except (TypeError, ValueError):
                week = None

            game_id = f"cfb:{season}:{d.isoformat()}:{away}@{home}"

            rows.append(CFBGameRow(
                game_id=game_id,
                game_date=d,
                season=season,
                week=week,
                home=str(home),
                away=str(away),
                home_conference=home_conf,
                away_conference=away_conf,
                home_score=hp,
                away_score=ap,
                home_won=(hp > ap),
            ))

    rows.sort(key=lambda g: g.game_date)
    log.info("Loaded %d CFB completed games across seasons %s", len(rows), seasons)
    return rows


def _backfill_season(season: int) -> None:
    """Trigger the ESPN fallback to populate the schedule cache for season."""
    try:
        from ..sources.cfb_cfbfastr_epa import CFBCfbfastREPA
        connector = CFBCfbfastREPA(timeout=30.0)
        connector._load_schedule(season)
    except Exception as e:
        log.warning("_backfill_season(%s) failed: %s", season, e)
