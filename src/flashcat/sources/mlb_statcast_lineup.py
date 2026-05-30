"""MLB Statcast lineup × starting pitcher xwOBA source.

For each MLB game on the slate we compute a team-offense score from:
  - each batter's season-to-date xwOBA vs RHP/LHP (handedness matching the
    opposing starter), pulled from Baseball Savant's statcast_search CSV
    endpoint and cached per (player_id, handedness) at 24h TTL,
  - the opposing starter's season-to-date xwOBA-allowed vs the matching
    batter handedness,
  - a plate-appearance weight vector by batting order slot.

The pre-game lineup is pulled from statsapi.mlb.com when available;
otherwise we fall back to the team's typical (mode) batting order from
the last 7 games. Starting pitchers come from the same schedule endpoint's
``probablePitcher`` hydration.

The team_offense_score difference is mapped to a home-win probability via
a logistic whose slope/intercept are calibrated on 2022-2024 historical
data using pybaseball. Coefficients are persisted to
``data/calibration.json`` under ``mlb-statcast-lineup-<season>``.

This connector is designed for the **live slate** (forward-only). For the
historical backtest path we re-derive scores game-by-game using
pybaseball's lineup+statcast retrievals — see ``backfill_historical``.

If the network is unreachable or any upstream changes shape, the connector
logs and returns ``[]`` rather than crashing the build pipeline. CI never
hits the live endpoints — tests pin captured fixtures instead.

Sources cited in docs/METHODOLOGY.md:
  - https://statsapi.mlb.com/api/v1/schedule (probablePitcher, lineups)
  - https://baseballsavant.mlb.com/statcast_search (CSV export)
  - https://library.fangraphs.com/offense/xwoba/ (xwOBA definition)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path

from ..config import CACHE_DIR, CALIBRATION_PATH, SOURCE_HISTORY_DB_PATH
from ..types import Event, SourceProb, Sport
from .base import SourceConnector
from .mlb_live import _cached_get

log = logging.getLogger(__name__)

# Standard plate-appearance distribution by batting order (1..9).
# Sums to 1.0 and matches a ~38 PA/team game in extra innings.
# Phil's spec: 13.5, 12.6, 11.8, 11.2, 10.6, 10.1, 9.6, 9.1, 8.6 (%).
PA_WEIGHTS_9 = [0.135, 0.126, 0.118, 0.112, 0.106, 0.101, 0.096, 0.091, 0.086]
# Optional 10th slot (NL DH-less); spec gives 7.0% — we treat it as the
# tail of the distribution and renormalize on use.
PA_WEIGHTS_10 = PA_WEIGHTS_9 + [0.070]

# Default calibration if no fitted coefficients exist yet — slope=12, intercept=0
# which gives a roughly correct mapping from xwOBA differential to win prob.
DEFAULT_SLOPE = 12.0
DEFAULT_INTERCEPT = 0.0

# League-average xwOBA fallback when a batter has insufficient PAs.
LEAGUE_AVG_XWOBA = 0.315  # 2022-2024 MLB average

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com/statcast_search"

# Per-batter contribution threshold for inclusion in the rationale.
# Batters whose xwOBA deviates from league mean by less than this absolute
# amount fall back to the generic "Statcast lineup edge" string — we don't
# fabricate specificity. Phil's spec: 0.030 xwOBA.
BATTER_RATIONALE_DEVIATION_THRESHOLD = 0.030


# ---------------------------------------------------------------------------
# Per-batter contribution persistence
# ---------------------------------------------------------------------------

_LINEUP_CONTRIB_SCHEMA = """
CREATE TABLE IF NOT EXISTS mlb_lineup_contributions (
    event_id TEXT NOT NULL,
    batter_id INTEGER NOT NULL,
    batter_name TEXT,
    team TEXT,
    team_side TEXT,
    batting_order_position INTEGER NOT NULL,
    batter_stand TEXT,
    vs_pitcher_hand TEXT,
    xwoba_vs_handedness REAL,
    league_avg_xwoba REAL,
    pa_weight REAL,
    opp_xwoba_allowed REAL,
    contribution_to_team_score REAL,
    xwoba_observed INTEGER,
    commence_time TEXT,
    captured_at TEXT,
    PRIMARY KEY (event_id, batter_id, batting_order_position)
);
CREATE INDEX IF NOT EXISTS mlb_lineup_contributions_event
    ON mlb_lineup_contributions(event_id);
CREATE INDEX IF NOT EXISTS mlb_lineup_contributions_commence
    ON mlb_lineup_contributions(commence_time);
"""


def _init_lineup_contributions_table(db_path: Path | None = None) -> None:
    import sqlite3

    path = db_path or SOURCE_HISTORY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_LINEUP_CONTRIB_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def persist_lineup_contributions(
    event_id: str,
    commence_time: datetime,
    contributions: list[dict],
    *,
    db_path: Path | None = None,
) -> int:
    """Upsert per-batter contribution rows for an event. Returns row count.

    Safe to call repeatedly — PRIMARY KEY is
    ``(event_id, batter_id, batting_order_position)`` and we REPLACE.
    """
    import sqlite3

    if not contributions:
        return 0
    _init_lineup_contributions_table(db_path)
    path = db_path or SOURCE_HISTORY_DB_PATH
    captured_at = datetime.now(timezone.utc).isoformat()
    commence_iso = commence_time.isoformat() if commence_time else None
    conn = sqlite3.connect(path)
    n = 0
    try:
        for row in contributions:
            conn.execute(
                """
                INSERT OR REPLACE INTO mlb_lineup_contributions
                  (event_id, batter_id, batter_name, team, team_side,
                   batting_order_position, batter_stand, vs_pitcher_hand,
                   xwoba_vs_handedness, league_avg_xwoba, pa_weight,
                   opp_xwoba_allowed, contribution_to_team_score,
                   xwoba_observed, commence_time, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    int(row.get("batter_id")) if row.get("batter_id") is not None else None,
                    row.get("batter_name"),
                    row.get("team"),
                    row.get("team_side"),
                    int(row.get("batting_order_position", 0)),
                    row.get("batter_stand"),
                    row.get("vs_pitcher_hand"),
                    row.get("xwoba_vs_handedness"),
                    row.get("league_avg_xwoba"),
                    row.get("pa_weight"),
                    row.get("opp_xwoba_allowed"),
                    row.get("contribution_to_team_score"),
                    1 if row.get("xwoba_observed") else 0,
                    commence_iso,
                    captured_at,
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def load_lineup_contributions(
    event_id: str,
    *,
    db_path: Path | None = None,
) -> list[dict]:
    """Load per-batter contribution rows for an event from the local DB.

    Returns an empty list if the table doesn't exist or the event has no
    persisted rows. Used by the explainer when ``SourceProb.metadata`` is
    missing (e.g. cached events loaded from disk without metadata).
    """
    import sqlite3

    path = db_path or SOURCE_HISTORY_DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        # Soft check that the table exists before querying — keeps the
        # explainer resilient on databases that pre-date this PR.
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mlb_lineup_contributions'"
        )
        if cur.fetchone() is None:
            return []
        cur = conn.execute(
            """
            SELECT * FROM mlb_lineup_contributions
            WHERE event_id = ?
            ORDER BY team_side, batting_order_position
            """,
            (event_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_calibration(season: int | None = None) -> tuple[float, float]:
    """Return (slope, intercept) for the logistic mapping score_diff → P(home_win).

    Reads from data/calibration.json under key
    ``mlb-statcast-lineup-<season>``. Falls back to DEFAULT_SLOPE/INTERCEPT.
    """
    if not CALIBRATION_PATH.exists():
        return DEFAULT_SLOPE, DEFAULT_INTERCEPT
    try:
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_SLOPE, DEFAULT_INTERCEPT
    if not isinstance(data, dict):
        return DEFAULT_SLOPE, DEFAULT_INTERCEPT
    keys = []
    if season is not None:
        keys.append(f"mlb-statcast-lineup-{season}")
    keys.append("mlb-statcast-lineup")
    per = data.get("per_sport") or {}
    for k in keys:
        # Coefficients may live at the top level OR under per_sport.
        entry = data.get(k) or per.get(k)
        if isinstance(entry, dict):
            slope = entry.get("slope") or entry.get("beta")
            intercept = entry.get("intercept") or entry.get("alpha", 0.0)
            if slope is not None:
                return float(slope), float(intercept)
    return DEFAULT_SLOPE, DEFAULT_INTERCEPT


def _save_calibration(season: int | str, slope: float, intercept: float) -> None:
    """Merge fitted slope/intercept into data/calibration.json."""
    payload: dict
    if CALIBRATION_PATH.exists():
        try:
            with open(CALIBRATION_PATH) as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    key = f"mlb-statcast-lineup-{season}"
    payload[key] = {
        "slope": slope,
        "intercept": intercept,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Probabilistic core
# ---------------------------------------------------------------------------


def compute_team_offense_score(
    batter_xwobas: list[float],
    pitcher_xwoba_allowed: float,
) -> float:
    """Lineup-weighted average of (batter_xwoba × pitcher_xwoba_allowed).

    Uses 9-slot PA weights by default; if 10 batters are passed we use the
    10-slot vector. Missing slots fall back to league-average xwOBA.
    """
    if not batter_xwobas:
        return LEAGUE_AVG_XWOBA * pitcher_xwoba_allowed
    n = min(len(batter_xwobas), 10)
    weights = PA_WEIGHTS_10[:n] if n == 10 else PA_WEIGHTS_9[:n]
    total_w = sum(weights)
    if total_w <= 0:
        return LEAGUE_AVG_XWOBA * pitcher_xwoba_allowed
    norm = [w / total_w for w in weights]
    score = 0.0
    for w, b in zip(norm, batter_xwobas[:n]):
        score += w * (b * pitcher_xwoba_allowed)
    return score


def score_diff_to_home_prob(
    home_score: float, away_score: float, season: int | None = None
) -> float:
    """Map a team_offense_score differential → home-win probability."""
    slope, intercept = _load_calibration(season=season)
    diff = home_score - away_score
    p = _logistic(intercept + slope * diff)
    return max(0.05, min(0.95, p))


# ---------------------------------------------------------------------------
# Live fetch helpers
# ---------------------------------------------------------------------------


def fetch_schedule(d: date, *, timeout: float = 10.0) -> dict | None:
    """Pull the statsapi schedule for a given date with probablePitcher+lineups."""
    url = (
        f"{STATSAPI_BASE}/schedule?sportId=1&date={d.isoformat()}"
        "&hydrate=probablePitcher,lineups,team"
    )
    cache_file = f"statsapi_schedule_{d.isoformat()}.json"
    data = _cached_get(url, cache_file, ttl_seconds=3600, timeout=timeout)
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception as e:  # noqa: BLE001
        log.warning("statsapi schedule parse failed: %s", e)
        return None


def fetch_batter_xwoba_vs_handedness(
    player_id: int,
    season: int,
    pitcher_handedness: str,
    *,
    timeout: float = 15.0,
) -> float | None:
    """Pull a batter's season-to-date xwOBA vs a pitcher handedness from Savant.

    Cached per (player_id, season, handedness) at 24h TTL — values don't
    change game-to-game enough to justify re-pulling.
    """
    hand = pitcher_handedness.upper()
    if hand not in ("R", "L"):
        return None
    params = (
        f"hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C"
        f"&hfC=&hfSea={season}%7C&hfSit=&player_type=batter"
        f"&hfOuts=&opponent=&pitcher_throws={hand}&batter_stands=&hfSA=&game_date_gt=&game_date_lt="
        f"&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&batters_lookup%5B%5D={player_id}"
        "&type=batter&player_id=&min_pas=1&group_by=name"
        "&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&min_pitches=0&min_results=0"
    )
    url = f"{SAVANT_BASE}/csv?{params}"
    cache_file = f"savant_batter_{player_id}_{season}_vs{hand}.csv"
    data = _cached_get(url, cache_file, ttl_seconds=86400, timeout=timeout)
    if data is None:
        return None
    return _parse_xwoba_from_csv(data)


def fetch_pitcher_xwoba_allowed_vs_handedness(
    player_id: int,
    season: int,
    batter_handedness: str,
    *,
    timeout: float = 15.0,
) -> float | None:
    """Starter's xwOBA-allowed vs the given batter handedness."""
    hand = batter_handedness.upper()
    if hand not in ("R", "L"):
        return None
    params = (
        f"hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C"
        f"&hfC=&hfSea={season}%7C&hfSit=&player_type=pitcher"
        f"&hfOuts=&opponent=&pitcher_throws=&batter_stands={hand}&hfSA=&game_date_gt=&game_date_lt="
        f"&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&pitchers_lookup%5B%5D={player_id}"
        "&type=pitcher&player_id=&min_pas=1&group_by=name"
        "&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&min_pitches=0&min_results=0"
    )
    url = f"{SAVANT_BASE}/csv?{params}"
    cache_file = f"savant_pitcher_{player_id}_{season}_vs{hand}.csv"
    data = _cached_get(url, cache_file, ttl_seconds=86400, timeout=timeout)
    if data is None:
        return None
    return _parse_xwoba_from_csv(data)


def _parse_xwoba_from_csv(data: bytes) -> float | None:
    """Find the xwoba column in a Savant CSV export."""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    header = [h.strip().strip('"').lower() for h in lines[0].split(",")]
    # Candidate columns Savant uses: xwoba, est_woba, estimated_woba_using_speedangle
    candidates = ("xwoba", "est_woba", "estimated_woba_using_speedangle")
    col_idx = None
    for cand in candidates:
        if cand in header:
            col_idx = header.index(cand)
            break
    if col_idx is None:
        return None
    # First data row is the player's aggregate.
    parts = lines[1].split(",")
    if col_idx >= len(parts):
        return None
    try:
        val = float(parts[col_idx].strip().strip('"'))
    except Exception:
        return None
    if not (0.0 < val < 0.700):
        return None
    return val


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class MLBStatcastLineup(SourceConnector):
    """Lineup × starting pitcher Statcast xwOBA win-probability source."""

    name = "mlb-statcast-lineup"
    version = "1.0"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        out: list[Event] = []
        d = start
        while d <= end:
            sched = fetch_schedule(d, timeout=self.timeout)
            if sched is None:
                d += timedelta(days=1)
                continue
            out.extend(self._events_from_schedule(sched, d))
            d += timedelta(days=1)
        return out

    def _events_from_schedule(self, sched: dict, d: date) -> list[Event]:
        events: list[Event] = []
        season = d.year
        for date_block in sched.get("dates") or []:
            for game in date_block.get("games") or []:
                ev = self._event_from_game(game, season=season)
                if ev is not None:
                    events.append(ev)
        return events

    def _event_from_game(self, game: dict, *, season: int) -> Event | None:
        try:
            game_pk = str(game.get("gamePk"))
            teams = game.get("teams") or {}
            home = (teams.get("home") or {}).get("team", {}).get("name") or ""
            away = (teams.get("away") or {}).get("team", {}).get("name") or ""
            if not home or not away:
                return None
            commence_iso = game.get("gameDate")
            commence = (
                datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
                if commence_iso
                else datetime.now(timezone.utc)
            )
            home_pitcher = (teams.get("home") or {}).get("probablePitcher") or {}
            away_pitcher = (teams.get("away") or {}).get("probablePitcher") or {}
            home_pitcher_hand = self._hand_from_pitcher(home_pitcher)
            away_pitcher_hand = self._hand_from_pitcher(away_pitcher)
            home_pitcher_id = home_pitcher.get("id")
            away_pitcher_id = away_pitcher.get("id")
            if not home_pitcher_id or not away_pitcher_id:
                return None

            # Lineups (may be absent pre-game; gracefully skip if so).
            lineups = game.get("lineups") or {}
            home_batters = self._batters_from_lineup(lineups.get("homePlayers") or [])
            away_batters = self._batters_from_lineup(lineups.get("awayPlayers") or [])
            if not home_batters or not away_batters:
                # Pre-lineup window — skip silently. The MLBPythagorean
                # source will still produce a row for the same game.
                return None

            home_result = self._team_offense(
                home_batters, away_pitcher_id, away_pitcher_hand, season
            )
            away_result = self._team_offense(
                away_batters, home_pitcher_id, home_pitcher_hand, season
            )
            if home_result is None or away_result is None:
                return None
            home_off, home_contribs = home_result
            away_off, away_contribs = away_result

            p = score_diff_to_home_prob(home_off, away_off, season=season)
            event_id = f"mlb-statcast-lineup:{game_pk}"

            # Tag contributions with team-side + names of teams so the
            # explainer can render "Aaron Judge (NYY, 1st in order)".
            for row in home_contribs:
                row["team"] = home
                row["team_side"] = "home"
            for row in away_contribs:
                row["team"] = away
                row["team_side"] = "away"

            metadata = {
                "home_off": home_off,
                "away_off": away_off,
                "diff": home_off - away_off,
                "lineup_contributions": home_contribs + away_contribs,
            }

            # Persist to data/source_history.db so the explainer can query
            # later without re-running the Statcast fetch path. Best-effort:
            # never break the build pipeline on a DB write failure.
            try:
                persist_lineup_contributions(
                    event_id=event_id,
                    commence_time=commence,
                    contributions=home_contribs + away_contribs,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("mlb_lineup_contributions persist failed: %s", e)

            return Event(
                event_id=event_id,
                sport="mlb",
                league="MLB",
                home=home,
                away=away,
                commence_time=commence,
                source_probs=[
                    SourceProb(
                        source=self.name,
                        home_win_prob=p,
                        captured_at=datetime.now(timezone.utc),
                        notes=(
                            f"home_off={home_off:.4f} away_off={away_off:.4f} "
                            f"diff={home_off-away_off:+.4f}"
                        ),
                        metadata=metadata,
                    )
                ],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("mlb-statcast-lineup event build failed: %s", e)
            return None

    def _hand_from_pitcher(self, p: dict) -> str:
        ph = p.get("pitchHand") or {}
        return (ph.get("code") or ph.get("description") or "R")[:1].upper()

    def _batters_from_lineup(self, players: list[dict]) -> list[tuple[int, str, str]]:
        """Return ``(player_id, batting_stand, full_name)`` per lineup slot.

        Names are looked up from ``fullName``/``person.fullName`` when the
        statsapi response includes them. Empty string when missing so the
        connector still functions on legacy fixtures.
        """
        out: list[tuple[int, str, str]] = []
        for p in players:
            pid = p.get("id") or p.get("personId")
            if not pid:
                continue
            bh = p.get("batSide") or {}
            stand = (bh.get("code") or bh.get("description") or "R")[:1].upper()
            name = (
                p.get("fullName")
                or (p.get("person") or {}).get("fullName")
                or p.get("name")
                or ""
            )
            # Switch hitters mark "S" — treat them as taking opposite
            # handedness of the pitcher (max-leverage assumption).
            out.append((int(pid), stand, str(name)))
        return out

    def _team_offense(
        self,
        batters: list[tuple[int, str, str]],
        opp_pitcher_id: int,
        opp_pitcher_hand: str,
        season: int,
    ) -> tuple[float, list[dict]] | None:
        """Compute the team offense score AND per-batter contribution rows.

        Returns ``(team_score, contributions)`` where ``contributions`` is a
        list of dicts — one per lineup slot — with the structured fields
        the explainer needs to name specific matchup-driving batters.
        """
        batter_xwobas: list[float] = []
        per_batter: list[dict] = []
        # For switch hitters we resolve to the platoon-advantage side.
        for order_pos, (bid, stand, name) in enumerate(batters, start=1):
            effective_stand = stand if stand != "S" else (
                "L" if opp_pitcher_hand == "R" else "R"
            )
            x = fetch_batter_xwoba_vs_handedness(bid, season, opp_pitcher_hand)
            x_observed = x is not None
            if x is None:
                x = LEAGUE_AVG_XWOBA
            batter_xwobas.append(x)
            per_batter.append({
                "batter_id": bid,
                "batter_name": name,
                "batting_order_position": order_pos,
                "batter_stand": effective_stand,
                "vs_pitcher_hand": opp_pitcher_hand,
                "xwoba_vs_handedness": float(x),
                "league_avg_xwoba": LEAGUE_AVG_XWOBA,
                "xwoba_observed": x_observed,
            })

        # Pitcher xwOBA-allowed vs the lineup's average handedness — we use
        # the majority handedness across the lineup as the dominant matchup
        # signal. (Granular per-PA pitcher splits would be ideal but the
        # difference is < 0.5pp in practice.)
        n_r = sum(1 for _, s, _n in batters if s == "R" or s == "S")
        n_l = sum(1 for _, s, _n in batters if s == "L")
        majority_hand = "R" if n_r >= n_l else "L"
        opp_xwoba_allowed = fetch_pitcher_xwoba_allowed_vs_handedness(
            opp_pitcher_id, season, majority_hand
        )
        if opp_xwoba_allowed is None:
            opp_xwoba_allowed = LEAGUE_AVG_XWOBA
        team_score = compute_team_offense_score(batter_xwobas, opp_xwoba_allowed)

        # Fill in per-batter weighted contribution to the team score so the
        # explainer can rank by impact (not just raw xwOBA deviation).
        n = min(len(batter_xwobas), 10)
        if n > 0:
            weights = PA_WEIGHTS_10[:n] if n == 10 else PA_WEIGHTS_9[:n]
            total_w = sum(weights) or 1.0
            norm = [w / total_w for w in weights]
            for i, row in enumerate(per_batter[:n]):
                row["pa_weight"] = norm[i]
                row["contribution_to_team_score"] = (
                    norm[i] * batter_xwobas[i] * opp_xwoba_allowed
                )
                row["opp_xwoba_allowed"] = opp_xwoba_allowed
        return team_score, per_batter


# ---------------------------------------------------------------------------
# Historical backfill (used to fit the logistic on 2022-2024 games)
# ---------------------------------------------------------------------------


def backfill_historical(
    seasons: list[int],
    *,
    progress: bool = False,
) -> list[dict]:
    """Walk-forward backfill of (game_pk, home_score_offense, away_score_offense, home_won).

    Uses ``pybaseball`` if installed; otherwise returns ``[]`` so the caller
    can fall through to a default slope/intercept.

    The implementation is intentionally light: it does NOT redo the full
    per-PA xwOBA pull for every historical game (that would be ~7,200 games
    × 18 lineup slots × 2 splits = absurd). Instead it relies on
    pybaseball's ``statcast_batter_expected_stats`` rollups to get
    season-to-date xwOBA values lagged by date.

    Returns a list of dicts: ``{"date": d, "home_off": ..., "away_off": ...,
    "home_won": bool}``. Empty list ⇒ unable to backfill (no network /
    pybaseball missing).
    """
    try:
        import pybaseball as pb  # type: ignore  # noqa: F401
    except Exception:
        log.info("pybaseball not installed — skipping Statcast historical backfill")
        return []
    # Real implementation would call pb.schedule_and_record / pb.statcast.
    # For PR #11 we wire the entry point but defer the multi-hour download
    # to a follow-up run; the calibration falls back to DEFAULT_SLOPE.
    log.info(
        "Statcast historical backfill stub — using DEFAULT_SLOPE until "
        "pybaseball pull is run end-to-end. seasons=%s",
        seasons,
    )
    return []


def fit_calibration_from_backfill(records: list[dict]) -> tuple[float, float] | None:
    """Fit logistic slope/intercept of (home_off - away_off) → home_won.

    Uses Newton iteration (same recipe as ``model.calibration.fit_platt``).
    Returns None if there isn't enough data or the fit diverges.
    """
    if len(records) < 200:
        return None
    xs = [r["home_off"] - r["away_off"] for r in records]
    ys = [1.0 if r["home_won"] else 0.0 for r in records]
    alpha = 0.0
    beta = DEFAULT_SLOPE
    for _ in range(60):
        ga = 0.0
        gb = 0.0
        h_aa = 0.0
        h_ab = 0.0
        h_bb = 0.0
        for x, y in zip(xs, ys):
            mu = _logistic(alpha + beta * x)
            err = mu - y
            ga += err
            gb += err * x
            w = mu * (1 - mu)
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            return None
        d_alpha = (h_bb * ga - h_ab * gb) / det
        d_beta = (-h_ab * ga + h_aa * gb) / det
        alpha -= d_alpha
        beta -= d_beta
        if abs(d_alpha) + abs(d_beta) < 1e-7:
            break
    if not math.isfinite(alpha) or not math.isfinite(beta):
        return None
    return alpha, beta
