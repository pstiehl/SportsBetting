"""PGA market consensus connector — Odds API outright winner markets.

PGA Tour betting markets diverge from team-sport moneylines:

* **Outright winner** is the only widely-traded market across US books and
  is the single market the-odds-api exposes for PGA Tour events (sport key
  ``golf_pga_championship_winner``, ``golf_masters_tournament_winner``,
  etc.). 100+ player field, ~+5000 typical favourite — high variance,
  rarely a +EV target.

* **Head-to-head match-ups** (the primary market the rest of the Flashcat
  pipeline assumes) are NOT carried by the-odds-api at any tier as of
  2026-05. They're available behind sportsbook-specific pages (DraftKings,
  BetMGM, FanDuel "matchups" tab) but those require an account-state
  cookie or graphql endpoints that aren't documented for public use.

* **Make the cut** props (binary) follow the same picture — not on the
  the-odds-api free tier.

Given the absence of a public H2H market feed, this connector serves two
purposes:

1. **Outright market source** — for each upcoming major (Masters / PGA
   Championship / US Open / The Open) we pull outright winner odds across
   all US books, de-vig them to a "market consensus" win-probability per
   player, and emit one Event per player keyed on the tournament. The
   pipeline currently doesn't blend outright bets (it's tuned for two-way
   markets), so these Events are emitted as **research-only** and exposed
   via ``source_history.db`` for future analysis. The downstream blender
   skips events that have no matching H2H opponent.

2. **Honest paywall documentation** — when the the-odds-api key is missing
   or the requested sport key is out-of-season, the connector logs a
   one-line warning and returns ``[]`` cleanly. We never fall back to
   scraped HTML, sample fixtures, or fabricated odds.

Operator note: if a sportsbook-specific H2H feed is wired in later (e.g.
a public DraftKings golf matchups graphql), it should land alongside this
connector under ``pga_market_dk_h2h.py`` rather than being shoehorned in
here. Outright odds and H2H odds have different vig structures.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Iterable

import httpx

from ..config import CACHE_DIR
from ..types import BookLine, Event, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)


THE_ODDS_BASE = "https://api.the-odds-api.com/v4"
USER_AGENT = "flashcat-research/1.0 (+https://github.com/pstiehl/SportsBetting)"

# Free-tier the-odds-api keys for PGA. These are seasonal — the API
# returns active=false outside the run-up to each major. We hit
# ``/sports?all=false`` first to filter to in-season events only.
PGA_OUTRIGHT_KEYS: tuple[str, ...] = (
    "golf_masters_tournament_winner",
    "golf_pga_championship_winner",
    "golf_the_open_championship_winner",
    "golf_us_open_winner",
)


def the_odds_api_key() -> str | None:
    # Read at call time so tests can monkey-patch ``os.environ``.
    return os.environ.get("THE_ODDS_API_KEY")


class PGAMarketConsensus(SourceConnector):
    """De-vigged outright winner consensus across US books for PGA majors.

    Pulls ``/v4/sports/{key}/odds`` for each in-season golf key and emits
    one BookLine per (book, player) for downstream introspection. We
    explicitly do NOT manufacture H2H matchup probabilities here — that
    market is paywalled (see module docstring).
    """

    name = "pga-market-consensus"
    version = "v1-outright-only"
    is_live = True

    def __init__(self, timeout: float = 12.0, max_books: int = 12):
        self.timeout = timeout
        self.max_books = max_books

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if sport is not None and sport != "pga":
            return []
        api_key = the_odds_api_key()
        if not api_key:
            log.info(
                "THE_ODDS_API_KEY not set; pga-market-consensus returns [] "
                "(no anonymous tier for the-odds-api)"
            )
            return []

        active_keys = self._active_pga_keys(api_key)
        if not active_keys:
            log.info("No active PGA outright markets in the-odds-api right now")
            return []

        out: list[Event] = []
        for k in active_keys:
            try:
                out.extend(self._fetch_outright(api_key, k))
            except Exception as e:  # noqa: BLE001
                log.warning("pga-market-consensus %s failed: %s", k, e)
        return out

    # ----- network -----------------------------------------------------

    def _active_pga_keys(self, api_key: str) -> list[str]:
        url = f"{THE_ODDS_BASE}/sports/"
        try:
            with httpx.Client(
                timeout=self.timeout, headers={"User-Agent": USER_AGENT}
            ) as c:
                r = c.get(url, params={"apiKey": api_key, "all": "false"})
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("the-odds-api /sports failed: %s", e)
            return []
        active = {s.get("key") for s in data if s.get("active")}
        return [k for k in PGA_OUTRIGHT_KEYS if k in active]

    def _fetch_outright(self, api_key: str, sport_key: str) -> list[Event]:
        url = f"{THE_ODDS_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "outrights",
            "oddsFormat": "american",
        }
        with httpx.Client(
            timeout=self.timeout, headers={"User-Agent": USER_AGENT}
        ) as c:
            r = c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        # Cache for downstream tooling.
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_DIR / f"odds_api_{sport_key}.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        return self._parse_outright(data, sport_key)

    # ----- parse -------------------------------------------------------

    @staticmethod
    def _parse_outright(data: list[dict], sport_key: str) -> list[Event]:
        """Each tournament returns one game object with N outright outcomes.

        The-odds-api shape (outrights):
            [{ "id": "...", "sport_key": "golf_masters_tournament_winner",
               "sport_title": "Masters Tournament Winner",
               "commence_time": "2026-04-09T12:00:00Z",
               "bookmakers": [
                  { "key": "draftkings", "title": "DraftKings",
                    "markets": [{ "key": "outrights",
                                  "outcomes": [{ "name": "Scottie Scheffler",
                                                  "price": 700 }, ...] }] } ]
            }]
        """
        out: list[Event] = []
        captured = datetime.now(timezone.utc)
        for tourny in data:
            try:
                commence = datetime.fromisoformat(
                    tourny["commence_time"].replace("Z", "+00:00")
                )
            except Exception:
                commence = captured
            tourny_label = tourny.get("sport_title", sport_key)
            event_id_root = f"oddsapi-pga:{tourny.get('id', sport_key)}"

            # Collapse all bookmaker outcomes into a per-player list of
            # BookLines so the de-vigging layer can later decide what to
            # do. We don't synthesize a 2-way market — outright has N>>2.
            for bk in tourny.get("bookmakers", [])[:64]:
                book = bk.get("key", "unknown")
                for market in bk.get("markets", []):
                    if market.get("key") != "outrights":
                        continue
                    for outcome in market.get("outcomes", []):
                        player = outcome.get("name") or ""
                        if not player:
                            continue
                        price = outcome.get("price")
                        try:
                            american = int(price)
                        except Exception:
                            continue
                        # One synthetic Event per (tournament, player) so
                        # source_history can ledger the implied prob.
                        # ``away`` is "field" — the implicit other side of
                        # an outright bet.
                        ev_id = f"{event_id_root}:{_slug(player)}"
                        out.append(
                            Event(
                                event_id=ev_id,
                                sport="pga",
                                league=tourny_label,
                                home=player,
                                away="Field",
                                commence_time=commence,
                                lines=[
                                    BookLine(
                                        book=book,
                                        side="home",
                                        american=american,
                                        captured_at=captured,
                                        is_opening=False,
                                    )
                                ],
                            )
                        )
        return out


def _slug(name: str) -> str:
    return (
        (name or "")
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace(" ", "-")
        .strip("-")
    )
