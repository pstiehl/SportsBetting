"""Walk-forward feature builder for NBA games — Phase 2.

Pure functions only — no I/O outside the explicit loader entry points.
Every feature passed to the model has to be computable using ONLY data
strictly BEFORE the game's date (leakage gate asserted on every call).

Data source: cached game logs from
  ``data/cache/nba_historical/gamelogs/games_<season>.json``
  (produced by ``scripts/backfill_nba_historical.py`` via nba_api).
  Each row: {game_id, date, home, away, home_score, away_score, home_won}

Phase-2 feature rationale:
  The dominant Phase-1 loss bucket was ``line_moved_against`` (43%).
  That bucket fires when the model picks correctly on direction but has
  no real edge over the market implied probability (< 1pp). Season-level
  SRS gives structural team quality but not short-term form. We add:
    * Rolling pt-differential L5 — captures hot/cold streaks
    * Rolling win% L10          — stability of form, regime-change signal
    * Back-to-back / rest-days  — most predictive of soft lines in NBA
  These three categories are the primary channels through which market
  prices move against us: B2B games and form divergences are exactly
  where sharp money flows. Capturing them in the model reduces "lost to
  vig without real signal" bets.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from ..config import CACHE_DIR, DATA_DIR

log = logging.getLogger(__name__)

NBA_GAMELOG_CACHE = CACHE_DIR / "nba_historical" / "gamelogs"

# ───────────────────────────── Feature names ──────────────────────────────

FEATURE_NAMES: list[str] = [
    "pt_diff_l5_home",   # rolling point-diff per game, home team, L5
    "pt_diff_l5_away",   # rolling point-diff per game, away team, L5
    "pt_diff_l5_diff",   # home − away (primary form signal)
    "win_pct_l10_home",  # win% home team, last 10
    "win_pct_l10_away",  # win% away team, last 10
    "win_pct_l10_diff",  # home − away win%
    "b2b_home",          # 1 if home team played yesterday
    "b2b_away",          # 1 if away team played yesterday
    "b2b_diff",          # b2b_away − b2b_home (positive = away at B2B disadvantage)
    "rest_days_diff",    # (home rest) − (away rest), capped ±7
    "srs_diff",          # season SRS home − away (pre-loaded from source_history.db)
    "home_court_flag",   # constant 1.0 (isolates HCA in logistic coeff)
]

N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────── Data structures ──────────────────────────────


@dataclass
class NBAGameRow:
    """One row per NBA regular-season game from the cached game log."""

    game_id: str
    game_date: date
    season: str          # e.g. "2022-23"
    home: str            # 3-letter team abbrev (nba_api format)
    away: str
    home_score: Optional[int]
    away_score: Optional[int]
    home_won: Optional[bool]

    @property
    def pt_diff_home(self) -> Optional[float]:
        """Home team point differential (+home score lead)."""
        if self.home_score is None or self.away_score is None:
            return None
        return float(self.home_score - self.away_score)

    @property
    def pt_diff_away(self) -> Optional[float]:
        if self.home_score is None or self.away_score is None:
            return None
        return float(self.away_score - self.home_score)


@dataclass
class NBATeamSnapshot:
    """Rolling stats for one team, as-of a cutoff date (leakage-free)."""

    team: str
    as_of: date

    # Last N games (in order, oldest → newest), strictly before as_of
    recent_games: list[NBAGameRow] = field(default_factory=list)

    @property
    def pt_diff_l5(self) -> Optional[float]:
        """Rolling avg point differential (from team's perspective) over last 5 games."""
        games = [g for g in self.recent_games][-5:]
        diffs = []
        for g in games:
            if g.home == self.team and g.pt_diff_home is not None:
                diffs.append(g.pt_diff_home)
            elif g.away == self.team and g.pt_diff_away is not None:
                diffs.append(g.pt_diff_away)
        if len(diffs) < 3:
            return None
        return sum(diffs) / len(diffs)

    @property
    def win_pct_l10(self) -> Optional[float]:
        """Win percentage over last 10 games."""
        games = [g for g in self.recent_games][-10:]
        wins = 0
        total = 0
        for g in games:
            if g.home == self.team:
                if g.home_won is not None:
                    total += 1
                    if g.home_won:
                        wins += 1
            elif g.away == self.team:
                if g.home_won is not None:
                    total += 1
                    if not g.home_won:
                        wins += 1
        if total < 5:
            return None
        return wins / total

    @property
    def last_game_date(self) -> Optional[date]:
        if not self.recent_games:
            return None
        return self.recent_games[-1].game_date

    def rest_days(self, as_of: date) -> Optional[int]:
        """Days since last game, capped at 14."""
        if not self.last_game_date:
            return None
        delta = (as_of - self.last_game_date).days
        return min(delta, 14)

    def is_b2b(self, as_of: date) -> bool:
        """True if this team played exactly 1 day ago."""
        if not self.last_game_date:
            return False
        return (as_of - self.last_game_date).days == 1


# ─────────────────────────── Loaders ──────────────────────────────────────


def _season_label(d: date) -> str:
    """Convert a date to its NBA season label (e.g. '2022-23')."""
    if d.month >= 10:
        yr = d.year
    else:
        yr = d.year - 1
    return f"{yr}-{str(yr + 1)[2:]}"


def load_nba_game_logs(seasons: list[str] | None = None) -> list[NBAGameRow]:
    """Load cached NBA game logs from disk.

    ``seasons`` — list of season labels like ["2021-22", "2022-23"]. If None,
    loads all available season files.

    Returns rows sorted by game_date ascending.
    """
    if not NBA_GAMELOG_CACHE.exists():
        log.warning("NBA gamelog cache missing: %s", NBA_GAMELOG_CACHE)
        return []

    rows: list[NBAGameRow] = []
    found_files = sorted(NBA_GAMELOG_CACHE.glob("games_*.json"))
    if not found_files:
        log.warning("No NBA gamelog files in %s", NBA_GAMELOG_CACHE)
        return []

    for fpath in found_files:
        # Extract season label from filename: games_2022-23.json → "2022-23"
        stem = fpath.stem  # "games_2022-23"
        label = stem.replace("games_", "")
        if seasons is not None and label not in seasons:
            continue
        try:
            raw = json.loads(fpath.read_text())
        except Exception as e:
            log.warning("Failed to parse %s: %s", fpath, e)
            continue
        for r in raw:
            try:
                gd = datetime.strptime(r["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            hw = r.get("home_won")
            rows.append(
                NBAGameRow(
                    game_id=str(r.get("game_id", "")),
                    game_date=gd,
                    season=label,
                    home=r["home"],
                    away=r["away"],
                    home_score=r.get("home_score"),
                    away_score=r.get("away_score"),
                    home_won=bool(hw) if hw is not None else None,
                )
            )

    rows.sort(key=lambda g: g.game_date)
    log.info("Loaded %d NBA games across %d season files", len(rows), len(found_files))
    return rows


def fit_rolling_snapshots(
    games: list[NBAGameRow],
) -> dict[tuple[str, date], NBATeamSnapshot]:
    """Build per-team rolling snapshots keyed by (team, game_date).

    For each game G, the snapshot for team T as-of G.game_date contains
    all games team T played STRICTLY BEFORE G.game_date. This is the
    leakage gate — the snapshot never includes G itself.

    Returns dict: (team, game_date) → NBATeamSnapshot
    """
    # Collect all dates each team played
    team_games: dict[str, list[NBAGameRow]] = defaultdict(list)
    for g in sorted(games, key=lambda x: x.game_date):
        team_games[g.home].append(g)
        team_games[g.away].append(g)

    snapshots: dict[tuple[str, date], NBATeamSnapshot] = {}

    # All unique game dates in chronological order
    all_dates = sorted({g.game_date for g in games})

    for d in all_dates:
        # Which teams play today?
        today_teams: set[str] = set()
        for g in games:
            if g.game_date == d:
                today_teams.add(g.home)
                today_teams.add(g.away)

        for team in today_teams:
            prior = [
                g for g in team_games[team]
                if g.game_date < d  # STRICT: no leakage
            ]
            # Keep only the most recent 20 games (enough for L10 + L5)
            prior_trimmed = sorted(prior, key=lambda g: g.game_date)[-20:]
            snap = NBATeamSnapshot(
                team=team,
                as_of=d,
                recent_games=prior_trimmed,
            )
            # Leakage assertion
            for g in snap.recent_games:
                assert g.game_date < d, (
                    f"LEAKAGE: snapshot for {team} on {d} contains game from {g.game_date}"
                )
            snapshots[(team, d)] = snap

    return snapshots


def load_srs_from_db(
    db_path: Path | None = None,
) -> dict[tuple[str, str], float]:
    """Load season SRS proxy from source_history.db.

    The bref connector stores predictions as home_prob = diff_to_home_prob(srs_diff).
    We invert: srs_diff ≈ norminv(home_prob) * 11.0 - 2.5.

    Returns dict: (home_team, season_label) → srs_diff (home − away)
    Note: we store this as (home, away, date) → srs_diff for exact game lookup.
    """
    from ..sources.nba_brefer import NBA_MARGIN_SIGMA, NBA_HFA_POINTS

    if db_path is None:
        from ..config import SOURCE_HISTORY_DB_PATH
        db_path = SOURCE_HISTORY_DB_PATH

    if not db_path.exists():
        log.warning("source_history.db not found at %s", db_path)
        return {}

    result: dict[tuple[str, str, str], float] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute(
            "SELECT home, away, commence_time, home_prob "
            "FROM predictions WHERE sport='nba' AND source='nba-bref-srs-pace'"
        )
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        log.warning("Failed to load NBA SRS from db: %s", e)
        return {}

    def _norminv(p: float) -> float:
        # Approximate normal quantile via rational approx (Abramowitz & Stegun)
        p = max(0.001, min(0.999, p))
        if p > 0.5:
            sign = 1.0
            p = 1.0 - p
        else:
            sign = -1.0
        t = math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
        return sign * z

    for home, away, ct, home_prob in rows:
        if home_prob is None:
            continue
        try:
            z = _norminv(float(home_prob))
            # home_prob = Φ(srs_diff / sigma); srs_diff = z * sigma
            # srs_diff = (h_srs - a_srs + HFA)
            srs_diff = z * NBA_MARGIN_SIGMA
        except Exception:
            continue
        # Key: (home, away, game_date_str)
        gd = ct[:10] if ct else ""
        result[(home, away, gd)] = srs_diff

    log.info("Loaded %d NBA SRS entries from DB", len(result))
    return result


# ─────────────────────────── Feature builder ──────────────────────────────


def build_features(
    game: NBAGameRow,
    snapshots: dict[tuple[str, date], NBATeamSnapshot],
    srs_lookup: dict[tuple[str, str, str], float],
) -> dict[str, Optional[float]] | None:
    """Build the Phase-2 feature dict for one game.

    Returns None if required features cannot be populated (e.g. not enough
    prior games). Partial features (nullable) are allowed — the model uses
    mean imputation for nullable features.

    Leakage gate: all data derived exclusively from games strictly before
    ``game.game_date``.
    """
    gd = game.game_date
    home = game.home
    away = game.away

    snap_h = snapshots.get((home, gd))
    snap_a = snapshots.get((away, gd))

    if snap_h is None or snap_a is None:
        # No snapshot means no prior games for this team — too early in season
        return None

    # Rolling point diff L5
    pd_h = snap_h.pt_diff_l5
    pd_a = snap_a.pt_diff_l5

    # Win % L10
    wp_h = snap_h.win_pct_l10
    wp_a = snap_a.win_pct_l10

    # Back-to-back
    b2b_h = 1.0 if snap_h.is_b2b(gd) else 0.0
    b2b_a = 1.0 if snap_a.is_b2b(gd) else 0.0

    # Rest days
    rest_h = snap_h.rest_days(gd)
    rest_a = snap_a.rest_days(gd)
    rest_diff: Optional[float] = None
    if rest_h is not None and rest_a is not None:
        rest_diff = float(max(-7, min(7, rest_h - rest_a)))

    # SRS diff from pre-loaded DB lookup
    gd_str = gd.isoformat()
    srs_diff = srs_lookup.get((home, away, gd_str))

    return {
        "pt_diff_l5_home":  pd_h,
        "pt_diff_l5_away":  pd_a,
        "pt_diff_l5_diff":  (pd_h - pd_a) if (pd_h is not None and pd_a is not None) else None,
        "win_pct_l10_home": wp_h,
        "win_pct_l10_away": wp_a,
        "win_pct_l10_diff": (wp_h - wp_a) if (wp_h is not None and wp_a is not None) else None,
        "b2b_home":         b2b_h,
        "b2b_away":         b2b_a,
        "b2b_diff":         b2b_a - b2b_h,
        "rest_days_diff":   rest_diff,
        "srs_diff":         srs_diff,
        "home_court_flag":  1.0,
    }


def feature_vector(
    feat: dict[str, Optional[float]],
    *,
    fill_mean: dict[str, float] | None = None,
) -> list[float] | None:
    """Convert a feature dict to a fixed-length list[float] for sklearn.

    ``fill_mean`` maps feature name → training-set mean, used for
    mean-imputation of None values. If None, features with missing values
    are filled with 0.0 (a safe default before training-mean is available).

    Returns None if any required features (b2b_diff, home_court_flag) are
    None — those should never be None, and if they are it signals a bug.
    """
    # home_court_flag is the only feature that must never be None.
    # b2b_home/away/diff are always 0.0 for real data but may be None
    # in ablation runs — they degrade gracefully to 0.0 imputation.
    required = {"home_court_flag"}
    for name in required:
        if feat.get(name) is None:
            log.warning("Required feature %s is None — dropping game", name)
            return None

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
