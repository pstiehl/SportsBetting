"""Jeff Sackmann surface-adjusted Elo for ATP / WTA.

Sackmann publishes the ``tennis_atp`` and ``tennis_wta`` repos on GitHub:
canonical CSVs of every professional singles match. We rebuild a
surface-specific Elo rating table walk-forward: each player has separate
ratings for Hard / Clay / Grass / Carpet plus an overall rating, and the
prediction for match (i, j, surface) at time t uses ONLY ratings derived
from matches before t.

Why surface matters: an ATP player's clay rating can differ from their
hard rating by 200+ Elo points. Pooling them throws away signal. Sackmann
himself recommends regressing the surface-specific rating toward the
overall when the surface-specific sample is small — we use:

    R_surface_effective = (n / (n + k)) * R_surface + (k / (n + k)) * R_overall

with k=20 (i.e. a player needs ~20 surface-specific matches before their
surface rating fully takes over).

K-factor is the standard tennis-Elo schedule:

    K = 250 / (n_career + 5) ** 0.4

so K starts ~80 for a brand-new player and decays toward ~25 for veterans.

References:
  - https://github.com/JeffSackmann/tennis_atp
  - https://github.com/JeffSackmann/tennis_wta
  - http://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/
"""

from __future__ import annotations

import csv
import io
import logging
import math
from datetime import date, datetime, time, timezone
from pathlib import Path

import httpx

from ..config import CACHE_DIR
from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

ATP_URL_TMPL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    "atp_matches_{year}.csv"
)
WTA_URL_TMPL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"
    "wta_matches_{year}.csv"
)


def _surface_canonical(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("hard", "h"):
        return "Hard"
    if s in ("clay", "c"):
        return "Clay"
    if s in ("grass", "g"):
        return "Grass"
    if s in ("carpet",):
        return "Carpet"
    return "Hard"  # default — Slams ex-Wimbledon ex-RG are hard


def _normalize_name(name: str) -> str:
    return (name or "").lower().replace(".", "").replace("-", " ").strip()


def _canonical_pair(p1: str, p2: str) -> tuple[str, str, bool]:
    a = _normalize_name(p1)
    b = _normalize_name(p2)
    if a <= b:
        return p1, p2, False
    return p2, p1, True


def _expected(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _k_factor(n_matches: int) -> float:
    return 250.0 / ((n_matches + 5) ** 0.4)


def _download_year(url: str, cache_file: str, timeout: float = 60.0) -> list[dict]:
    cache_path = CACHE_DIR / cache_file
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
            cache_path.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            log.warning("Sackmann download failed for %s: %s", url, e)
            return []
    try:
        text = cache_path.read_text(errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)
    except Exception as e:  # noqa: BLE001
        log.warning("Sackmann CSV parse failed for %s: %s", cache_file, e)
        return []


class _SackmannElo:
    """Shared implementation parameterised by tour (atp/wta)."""

    SURFACE_REGRESS_K = 20  # see module docstring
    BASE_RATING = 1500.0

    def __init__(self, tour: str, years: list[int], timeout: float = 60.0):
        self.tour = tour
        self.years = sorted(years)
        self.timeout = timeout

    def _load_rows(self) -> list[dict]:
        tmpl = ATP_URL_TMPL if self.tour == "atp" else WTA_URL_TMPL
        rows: list[dict] = []
        for y in self.years:
            url = tmpl.format(year=y)
            cache_file = f"sackmann_{self.tour}_{y}.csv"
            data = _download_year(url, cache_file, timeout=self.timeout)
            for r in data:
                # tourney_date is YYYYMMDD
                try:
                    d = datetime.strptime(r.get("tourney_date", ""), "%Y%m%d").date()
                except Exception:
                    continue
                r["_date"] = d
                r["_surface"] = _surface_canonical(r.get("surface", ""))
                rows.append(r)
        rows.sort(key=lambda x: (x["_date"], x.get("match_num") or "0"))
        return rows

    def predictions(self, start: date, end: date) -> list[tuple]:
        """Return a list of ``(event_id, sport, commence_dt, home, away, home_prob, home_won)``.

        Walk-forward: ratings are updated AFTER each match, so the prediction
        for match t uses only matches strictly before t. Surfaces are tracked
        independently; ``home_prob`` is the prob for the alphabetically-first
        player (matching tennis_history's home/away convention).
        """
        rows = self._load_rows()
        # rating tables
        overall: dict[str, float] = {}
        overall_n: dict[str, int] = {}
        surface: dict[tuple[str, str], float] = {}
        surface_n: dict[tuple[str, str], int] = {}

        out: list[tuple] = []
        sport: Sport = "atp" if self.tour == "atp" else "wta"

        def _rating(player: str, surf: str) -> tuple[float, int, float, int]:
            r_overall = overall.get(player, self.BASE_RATING)
            n_overall = overall_n.get(player, 0)
            r_surface = surface.get((player, surf), self.BASE_RATING)
            n_surface = surface_n.get((player, surf), 0)
            # Regress surface toward overall when sample is small.
            k = self.SURFACE_REGRESS_K
            blended = (n_surface * r_surface + k * r_overall) / (n_surface + k)
            return blended, n_overall, r_surface, n_surface

        for row in rows:
            d = row["_date"]
            surf = row["_surface"]
            winner = row.get("winner_name") or ""
            loser = row.get("loser_name") or ""
            if not winner or not loser:
                continue

            # Compute prediction BEFORE updating ratings.
            r_w, n_w, rs_w, ns_w = _rating(winner, surf)
            r_l, n_l, rs_l, ns_l = _rating(loser, surf)
            p_winner = _expected(r_w, r_l)

            home, away, swap = _canonical_pair(winner, loser)
            home_won = home == winner
            home_prob = p_winner if home == winner else (1.0 - p_winner)

            if start <= d <= end:
                commence = datetime.combine(d, time(12, 0), tzinfo=timezone.utc)
                # event_id matches tennis-data style for merging
                key = (
                    f"sackmann:{sport}:{d.isoformat()}_"
                    f"{_normalize_name(home).replace(' ', '_')}_"
                    f"{_normalize_name(away).replace(' ', '_')}"
                )
                out.append((key, sport, commence, home, away, home_prob, home_won, surf))

            # ---- Update ratings using the actual outcome.
            k_w = _k_factor(n_w)
            k_l = _k_factor(n_l)
            update_w = k_w * (1.0 - p_winner)
            update_l = k_l * (0.0 - (1.0 - p_winner))  # = -k_l * (1 - p_winner)

            overall[winner] = overall.get(winner, self.BASE_RATING) + update_w
            overall[loser] = overall.get(loser, self.BASE_RATING) + update_l
            overall_n[winner] = overall_n.get(winner, 0) + 1
            overall_n[loser] = overall_n.get(loser, 0) + 1
            surface[(winner, surf)] = surface.get((winner, surf), self.BASE_RATING) + update_w
            surface[(loser, surf)] = surface.get((loser, surf), self.BASE_RATING) + update_l
            surface_n[(winner, surf)] = surface_n.get((winner, surf), 0) + 1
            surface_n[(loser, surf)] = surface_n.get((loser, surf), 0) + 1

        return out


class SackmannATPElo(SourceConnector):
    """Surface-adjusted Elo for ATP, walk-forward over Sackmann's repo."""

    name = "sackmann-atp-elo"
    version = "v1-surface-regress20"
    is_live = True

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "atp":
            return []
        years = list(range(max(2020, start.year - 1), end.year + 1))
        engine = _SackmannElo("atp", years, timeout=self.timeout)
        preds = engine.predictions(start, end)
        return _emit_events(preds, "atp", self.name)

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        years = list(range(max(2020, start.year - 1), end.year + 1))
        engine = _SackmannElo("atp", years, timeout=self.timeout)
        preds = engine.predictions(start, end)
        return _emit_results(preds, "atp")


class SackmannWTAElo(SourceConnector):
    """Surface-adjusted Elo for WTA, walk-forward over Sackmann's repo."""

    name = "sackmann-wta-elo"
    version = "v1-surface-regress20"
    is_live = True

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "wta":
            return []
        years = list(range(max(2020, start.year - 1), end.year + 1))
        engine = _SackmannElo("wta", years, timeout=self.timeout)
        preds = engine.predictions(start, end)
        return _emit_events(preds, "wta", self.name)

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        years = list(range(max(2020, start.year - 1), end.year + 1))
        engine = _SackmannElo("wta", years, timeout=self.timeout)
        preds = engine.predictions(start, end)
        return _emit_results(preds, "wta")


def _emit_events(preds: list[tuple], sport: Sport, source_name: str) -> list[Event]:
    out: list[Event] = []
    for key, _sp, commence, home, away, home_prob, _hw, surf in preds:
        out.append(
            Event(
                event_id=key,
                sport=sport,
                league=sport.upper(),
                home=home,
                away=away,
                commence_time=commence,
                source_probs=[
                    SourceProb(
                        source=source_name,
                        home_win_prob=max(0.001, min(0.999, home_prob)),
                        captured_at=commence,
                        notes=f"Sackmann surface-Elo ({surf})",
                    )
                ],
            )
        )
    return out


def _emit_results(preds: list[tuple], sport: Sport) -> list[HistoricalResult]:
    out: list[HistoricalResult] = []
    for key, _sp, commence, home, away, _hp, home_won, _surf in preds:
        out.append(
            HistoricalResult(
                event_id=key,
                sport=sport,
                home=home,
                away=away,
                commence_time=commence,
                home_won=home_won,
            )
        )
    return out
