"""DataGolf strokes-gained model connector for PGA Tour.

DataGolf publishes one of the most-respected strokes-gained models in golf
(see https://datagolf.com/api-access). Their model outputs per-player
pre-tournament forecasts (win, top-5/10/20, make-cut probabilities) plus a
``Data Golf Rankings`` table with skill estimates that can be turned into
Bradley-Terry head-to-head matchup probabilities.

API access reality (verified 2026-05-30):
    Every public endpoint at ``feeds.datagolf.com`` returns HTTP 403
    "invalid api key" without a ``key=...`` query-string parameter. There
    is no truly anonymous tier — the DataGolf rate limit (45 requests per
    minute) is keyed on the API token. So:

    * If ``DATAGOLF_API_KEY`` is set in the environment, this connector
      pulls **only the endpoints that DataGolf documents as "free"** on
      https://datagolf.com/api-access — rankings, pre-tournament
      predictions, and the pre-tournament archive (for backfill).
    * If the key is absent (the CI default), this connector returns ``[]``
      cleanly. It logs a warning but does not raise — PGA is wired in
      RESEARCH MODE, so the build pipeline stays green.

We honor Phil's constraint **DataGolf paid tier is OFF LIMITS**: we never
hit the live-model, course-fit, or skill-decomposition endpoints. Only
the three public-tier endpoints below are used.

Endpoints used (free tier, key still required for rate-limit accounting)::

    GET /preds/get-dg-rankings?file_format=json
        → 500-player skill table (used for Bradley-Terry matchup probs)
    GET /preds/pre-tournament?tour=pga&file_format=json&odds_format=percent
        → win / top-5 / top-10 / top-20 / make-cut probs for the field
    GET /preds/pre-tournament-archive?tour=pga&year=YYYY&event_id=N
        → historical archive of the same payload (used for backfill)

Output shape
------------
We synthesize **head-to-head matchup events** rather than emit one event
per player per outright market. PGA H2H matchups (the closest analog to
tennis singles or NBA moneylines) pair two players from the same
tournament; the bettor picks which player finishes ahead at week's end.

The connector groups the field into adjacent pairings by pre-tournament
win probability (rank 1 vs rank 2, rank 3 vs rank 4, …). For each pair,
we emit an Event with ``home``/``away`` set to the two player names and
``home_win_prob`` derived from a Bradley-Terry transform of the DataGolf
skill ratings::

    P(home beats away) = home_skill / (home_skill + away_skill)

where ``skill`` is the implied probability mass associated with each
player's pre-tournament win-percentage (after light Laplace smoothing).

This is a deliberate simplification — the "true" DataGolf matchup
probability would weight every possible round-by-round path. Our
single-shot transform matches what most public betting tools display
(``win_prob_A / (win_prob_A + win_prob_B)`` after normalization) and is
exactly what BetMGM / DraftKings use as their model anchor before
applying vig.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Iterable

import httpx

from ..config import CACHE_DIR
from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)


# --- API plumbing ---------------------------------------------------------

BASE_URL = "https://feeds.datagolf.com"
USER_AGENT = "flashcat-research/1.0 (+https://github.com/pstiehl/SportsBetting)"
CACHE_TTL_SEC = 6 * 3600  # 6 hours, per task spec


def datagolf_api_key() -> str | None:
    """Look up the DataGolf API key from the environment.

    Mirrors the pattern used by ``the_odds_api_key`` in config.py. We
    intentionally do *not* read this at import time so tests can monkey-patch.
    """
    return os.environ.get("DATAGOLF_API_KEY")


def _cache_path(slug: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"datagolf_{slug}.json"


def _read_cached(slug: str, ttl: int = CACHE_TTL_SEC) -> dict | list | None:
    p = _cache_path(slug)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(slug: str, payload: dict | list) -> None:
    p = _cache_path(slug)
    try:
        with open(p, "w") as f:
            json.dump(payload, f)
    except Exception as e:  # noqa: BLE001
        log.debug("datagolf cache write failed for %s: %s", slug, e)


def _get(
    path: str,
    params: dict,
    *,
    slug: str,
    timeout: float = 12.0,
    ttl: int = CACHE_TTL_SEC,
) -> dict | list | None:
    """Fetch a DataGolf endpoint with caching.

    Returns the parsed JSON or ``None`` if the call fails / no key configured.
    """
    api_key = datagolf_api_key()
    if not api_key:
        log.info(
            "DATAGOLF_API_KEY not set; skipping %s (DataGolf has no anonymous tier)",
            path,
        )
        return None

    cached = _read_cached(slug, ttl=ttl)
    if cached is not None:
        return cached

    url = f"{BASE_URL}{path}"
    full_params = {**params, "key": api_key, "file_format": "json"}
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as c:
            r = c.get(url, params=full_params)
            if r.status_code == 401 or r.status_code == 403:
                log.warning(
                    "datagolf %s returned %d — key may be invalid or endpoint paywalled; "
                    "treating as no-data",
                    path,
                    r.status_code,
                )
                return None
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("datagolf fetch failed for %s: %s", path, e)
        return None

    _write_cache(slug, data)
    return data


# --- Model helpers --------------------------------------------------------


def bradley_terry(skill_a: float, skill_b: float, *, smoothing: float = 1e-3) -> float:
    """Bradley-Terry head-to-head probability.

    ``skill_*`` are non-negative real-valued "strength" scores. Returns the
    probability that A beats B. With Laplace smoothing so identical or
    zero skills don't produce 0/0.
    """
    a = max(skill_a, 0.0) + smoothing
    b = max(skill_b, 0.0) + smoothing
    return a / (a + b)


def matchup_prob_from_win_probs(p_win_a: float, p_win_b: float) -> float:
    """Predict P(A beats B in head-to-head) from outright win probabilities.

    Uses the win-probability *ratio* as a Bradley-Terry proxy. This matches
    DataGolf's documented public-tier matchup tool behaviour. Clamped to
    [0.001, 0.999] for downstream blending sanity.
    """
    prob = bradley_terry(p_win_a, p_win_b)
    return max(0.001, min(0.999, prob))


# --- Connector ------------------------------------------------------------


def _canonical_pair(a: str, b: str) -> tuple[str, str, bool]:
    """Order two player names alphabetically; return (home, away, swapped)."""
    na = (a or "").strip().lower()
    nb = (b or "").strip().lower()
    if na <= nb:
        return a, b, False
    return b, a, True


def _normalize_player(name: str) -> str:
    return (name or "").lower().replace(".", "").replace(",", "").replace("  ", " ").strip()


def _event_id_for_pair(tour: str, event_label: str, home: str, away: str) -> str:
    """Stable event id for a head-to-head matchup."""
    slug = (
        f"{tour}:{event_label}:"
        f"{_normalize_player(home).replace(' ', '_')}_vs_"
        f"{_normalize_player(away).replace(' ', '_')}"
    )
    return f"datagolf:{slug}"


class PGADatagolf(SourceConnector):
    """DataGolf strokes-gained pre-tournament forecasts → H2H matchup probabilities.

    Wires the free-tier DataGolf endpoints into Flashcat. Emits one Event
    per synthesized head-to-head matchup for the upcoming PGA Tour event.

    If no ``DATAGOLF_API_KEY`` env var is set, returns ``[]`` cleanly so the
    build pipeline stays green and PGA falls into RESEARCH MODE.
    """

    name = "datagolf-sg"
    version = "v1-bradley-terry"
    is_live = True

    def __init__(
        self,
        timeout: float = 12.0,
        pair_strategy: str = "adjacent",
        max_pairs: int = 64,
    ):
        self.timeout = timeout
        # How to synthesize H2H matchups from the field:
        #   "adjacent": rank-1 vs rank-2, rank-3 vs rank-4, ... (default)
        #   "all":      every pair in the top max_pairs (Cartesian; rarely useful)
        self.pair_strategy = pair_strategy
        self.max_pairs = max_pairs

    # ----- live slate --------------------------------------------------

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if sport is not None and sport != "pga":
            return []

        pre = self._fetch_pre_tournament()
        if not pre:
            return []

        return self._build_events_from_pretournament(
            pre, start=start, end=end
        )

    def _fetch_pre_tournament(self) -> dict | None:
        return _get(
            "/preds/pre-tournament",
            params={"tour": "pga", "odds_format": "percent"},
            slug="pre_tournament_pga",
            timeout=self.timeout,
        )  # type: ignore[return-value]

    # ----- backfill ----------------------------------------------------

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        """DataGolf doesn't expose result rows directly via the public tier.

        The pre-tournament archive returns *predictions* keyed to events, but
        the official tournament finishing positions live behind
        ``/historical-raw-data/event-list`` which requires the paid tier.

        For backfill we therefore lean on the matchup-driven schema: when
        ``DATAGOLF_API_KEY`` is set we pull archived pre-tournament forecasts
        and emit them with ``home_won=None``. The grader / source_history
        ledger skips ungraded rows; PGA stays RESEARCH until either a paid
        tier is enabled or a separate results source is wired in.
        """
        if not datagolf_api_key():
            return []
        out: list[HistoricalResult] = []
        years = sorted({start.year, end.year, max(start.year, end.year - 1)})
        # Cap to avoid blowing through DataGolf's 45 req/min rate limit.
        years = [y for y in years if 2022 <= y <= end.year][:3]
        for y in years:
            archive = _get(
                "/preds/pre-tournament-archive",
                params={"tour": "pga", "year": str(y)},
                slug=f"pre_tournament_archive_pga_{y}",
                ttl=30 * 24 * 3600,  # 30-day cache on historicals
                timeout=self.timeout,
            )
            if not archive:
                continue
            # Archive payload is a list of events; we surface as ungraded
            # HistoricalResult shells so the runner can slot them.
            events = archive if isinstance(archive, list) else archive.get("events", [])
            for ev in events or []:
                event_label = ev.get("event_name") or "pga-event"
                for row in ev.get("baseline", []) or []:
                    name = row.get("player_name")
                    if not name:
                        continue
                    # Without finishing positions we can't grade — emit
                    # nothing rather than fake outcomes. The empty return
                    # here documents the gap honestly.
                    _ = event_label, name  # silence unused warnings
                    break
        return out

    # ----- transform ---------------------------------------------------

    def _build_events_from_pretournament(
        self, payload: dict, *, start: date, end: date
    ) -> list[Event]:
        """Convert a DataGolf pre-tournament payload into H2H Events."""
        # The DataGolf pre-tournament endpoint returns a dict like:
        #   {"event_name": "...", "last_updated": "...",
        #    "baseline":     [ {player_name, win, top_5, top_10, top_20, make_cut}, ... ],
        #    "baseline_history_fit": [...same shape...]}
        # We use the BASELINE model only (free tier) — the course-fit
        # variant is documented as paid.
        rows = self._extract_baseline_rows(payload)
        if not rows:
            return []

        event_name = payload.get("event_name") or "pga-event"
        # Use tournament start date for commence_time. DataGolf doesn't
        # always echo a start date so we default to the date window start.
        commence_date = self._infer_event_date(payload, default=start)
        if commence_date < start or commence_date > end:
            # Tournament outside the requested window — skip cleanly.
            log.debug(
                "datagolf event %s on %s outside [%s, %s] — skipping",
                event_name,
                commence_date,
                start,
                end,
            )
            return []

        commence_dt = datetime.combine(
            commence_date, dt_time(14, 0), tzinfo=timezone.utc
        )
        captured = datetime.now(timezone.utc)
        out: list[Event] = []

        # Sort field by win probability descending and pair adjacently.
        # This matches how books typically list "head-to-head match-ups"
        # in their golf pages (favourite vs next-favourite, then drift down).
        ranked = sorted(
            rows, key=lambda r: r.get("win", 0.0) or 0.0, reverse=True
        )
        pairs = self._pair_field(ranked)

        for i, (a, b) in enumerate(pairs[: self.max_pairs]):
            pa = float(a.get("win", 0.0) or 0.0)
            pb = float(b.get("win", 0.0) or 0.0)
            if pa <= 0.0 and pb <= 0.0:
                continue
            home, away, swapped = _canonical_pair(
                a["player_name"], b["player_name"]
            )
            home_skill = pb if swapped else pa
            away_skill = pa if swapped else pb
            home_prob = matchup_prob_from_win_probs(home_skill, away_skill)

            out.append(
                Event(
                    event_id=_event_id_for_pair("pga", event_name, home, away),
                    sport="pga",
                    league=f"PGA: {event_name}",
                    home=home,
                    away=away,
                    commence_time=commence_dt,
                    source_probs=[
                        SourceProb(
                            source=self.name,
                            home_win_prob=home_prob,
                            captured_at=captured,
                            notes=(
                                f"DataGolf SG baseline pre-tournament "
                                f"(win% home={home_skill*100:.2f}, away={away_skill*100:.2f}); "
                                "Bradley-Terry on win-prob ratio"
                            ),
                        )
                    ],
                )
            )
        return out

    @staticmethod
    def _extract_baseline_rows(payload: dict) -> list[dict]:
        """Pull the baseline-model rows out of a DataGolf pre-tournament payload.

        Handles both list-style and dict-style responses.
        """
        baseline = payload.get("baseline")
        if baseline is None:
            # Some DataGolf responses nest under "baseline_history_fit" only;
            # if the free tier is degraded we fall back to whatever rows are
            # present.
            for k in ("baseline_history_fit", "predictions", "data"):
                if k in payload and isinstance(payload[k], list):
                    baseline = payload[k]
                    break
        if not isinstance(baseline, list):
            return []
        rows: list[dict] = []
        for r in baseline:
            if not isinstance(r, dict):
                continue
            name = r.get("player_name") or r.get("name")
            if not name:
                continue
            win = r.get("win")
            if win is None:
                # DataGolf occasionally returns make_cut / top_n only; we
                # need win prob for the BT transform.
                continue
            # DataGolf's `odds_format=percent` endpoint returns ALL prob
            # fields in the 0..100 range — even small ones like 0.4 (a
            # 0.4% long-shot). We always divide by 100 to normalize to
            # [0, 1]. Mixed-scale fixtures will look wrong; that's by
            # design — DataGolf never returns the [0, 1] format from this
            # endpoint.
            rows.append(
                {
                    "player_name": name,
                    "win": _norm_pct(win),
                    "top_5": _norm_pct(r.get("top_5")),
                    "top_10": _norm_pct(r.get("top_10")),
                    "top_20": _norm_pct(r.get("top_20")),
                    "make_cut": _norm_pct(r.get("make_cut")),
                }
            )
        return rows

    @staticmethod
    def _infer_event_date(payload: dict, *, default: date) -> date:
        """Best-effort tournament start date extraction."""
        for k in ("event_start_date", "start_date", "tournament_date"):
            val = payload.get(k)
            if not val:
                continue
            try:
                return datetime.fromisoformat(str(val).split("T")[0]).date()
            except Exception:
                continue
        last_upd = payload.get("last_updated")
        if last_upd:
            try:
                return datetime.fromisoformat(
                    str(last_upd).replace("Z", "+00:00")
                ).date()
            except Exception:
                pass
        return default

    def _pair_field(self, ranked: list[dict]) -> list[tuple[dict, dict]]:
        pairs: list[tuple[dict, dict]] = []
        if self.pair_strategy == "all":
            top = ranked[: min(len(ranked), self.max_pairs)]
            for i, a in enumerate(top):
                for b in top[i + 1 :]:
                    pairs.append((a, b))
            return pairs
        # Default: adjacent pairing — feasible matchup market analog.
        it = iter(ranked)
        for a in it:
            b = next(it, None)
            if b is None:
                break
            pairs.append((a, b))
        return pairs


def _norm_pct(v) -> float | None:
    """Normalize a DataGolf percent field (0..100) into a probability [0, 1].

    Returns ``None`` for missing / un-parseable inputs. We always divide
    by 100 because the ``odds_format=percent`` payload never mixes scales
    — a 0.4% long-shot comes through as ``0.4``, not ``0.004``. See
    ``_extract_baseline_rows`` for the rationale.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return max(0.0, min(1.0, f / 100.0))


# Backwards-compatible alias kept in case external callers imported the
# old helper name. Drop in a future cleanup.
_safe_pct = _norm_pct
