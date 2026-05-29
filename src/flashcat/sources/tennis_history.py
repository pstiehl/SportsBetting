"""Tennis historical connector — tennis-data.co.uk archives.

Each season ships a single xlsx (ATP and WTA separately) with:
  - completed match results (winner/loser)
  - Pinnacle closing odds (PSW / PSL — decimal)
  - Bet365 closing odds (B365W / B365L)
  - market-average odds (AvgW / AvgL)

We treat the loser as "home" by convention only when the winner field is
alphabetically second (we don't actually know who served first). For our
purposes we pick a *consistent* labeling — first-listed player in the
canonicalised order is "home". The label is meaningless beyond stability
of event_ids across runs; what matters is that ``source_probs`` get applied
to the right side.

The Pinnacle closing line is the **gold standard** for tennis efficiency
research (Sackmann's writeup repeatedly recommends it). Devigging PSW/PSL
gives us a closing-line market probability we can backtest against.

Outcomes are used ONLY to grade — source probabilities come from closing
lines that were available *before* the match started.
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime, time, timezone
from typing import Iterable

import httpx

import math

from ..types import (
    BookLine,
    Event,
    HistoricalResult,
    Side,
    SourceProb,
    Sport,
    devig_two_way,
)
from .base import SourceConnector


def _rank_points_prob(
    home_pts: float | None, away_pts: float | None
) -> float | None:
    """Bradley-Terry-style win probability from ATP/WTA rank points.

    ``logit(p) = k * log(home_pts / away_pts)`` with ``k ≈ 0.45`` recovers
    Pinnacle closing probs to within ~5pp on average on Sackmann's archives.
    Intentionally simple — we just need a second source so the blender has
    something to disagree with the market about.
    """
    if not home_pts or not away_pts or home_pts <= 0 or away_pts <= 0:
        return None
    k = 0.45
    logit = k * math.log(home_pts / away_pts)
    return 1.0 / (1.0 + math.exp(-logit))

log = logging.getLogger(__name__)

ATP_URL_TMPL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"
WTA_URL_TMPL = "http://www.tennis-data.co.uk/{year}w/{year}.xlsx"


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds → American odds (rounded to nearest integer)."""
    if decimal_odds <= 1.0:
        return -100000
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def _canonical_pair(p1: str, p2: str) -> tuple[str, str, bool]:
    """Return (home, away, swap) with home alphabetically first.

    ``swap`` indicates whether the original (winner, loser) pair was
    flipped relative to our canonical ordering.
    """
    if p1 <= p2:
        return p1, p2, False
    return p2, p1, True


class TennisDataHistorical(SourceConnector):
    """Historical tennis matches + closing odds from tennis-data.co.uk."""

    name = "tennis-data"
    version = "co.uk-v1"
    is_live = True

    def __init__(self, tour: Sport = "atp", timeout: float = 30.0):
        if tour not in ("atp", "wta"):
            raise ValueError(f"tour must be 'atp' or 'wta', got {tour}")
        self.tour: Sport = tour
        self.timeout = timeout
        self.name = f"tennis-data-{tour}"

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if sport is not None and sport != self.tour:
            return []
        events, _ = self._load_range(start, end)
        return events

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        _, results = self._load_range(start, end)
        return results

    # ----- internals ---------------------------------------------------

    def _load_range(
        self, start: date, end: date
    ) -> tuple[list[Event], list[HistoricalResult]]:
        events: list[Event] = []
        results: list[HistoricalResult] = []
        years = sorted({start.year, end.year})
        for year in years:
            rows = self._download_year(year)
            for row in rows:
                ev, res = self._parse_row(row, year)
                if ev is None or res is None:
                    continue
                if not (start <= ev.commence_time.date() <= end):
                    continue
                events.append(ev)
                results.append(res)
        return events, results

    def _download_year(self, year: int) -> list[dict]:
        import openpyxl  # local import — keeps base install light

        url = (
            ATP_URL_TMPL.format(year=year)
            if self.tour == "atp"
            else WTA_URL_TMPL.format(year=year)
        )
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                data = r.content
        except Exception as e:  # noqa: BLE001
            log.warning("tennis-data fetch failed for %s %s: %s", self.tour, year, e)
            return []
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        ws = wb.active
        header = None
        out: list[dict] = []
        for row in ws.iter_rows(values_only=True):
            if header is None:
                header = list(row)
                continue
            out.append(dict(zip(header, row)))
        return out

    def _parse_row(
        self, row: dict, year: int
    ) -> tuple[Event | None, HistoricalResult | None]:
        winner = (row.get("Winner") or "").strip()
        loser = (row.get("Loser") or "").strip()
        if not winner or not loser:
            return None, None
        if (row.get("Comment") or "").strip().lower() != "completed":
            return None, None
        d = row.get("Date")
        if isinstance(d, datetime):
            match_day = d.date()
        elif isinstance(d, date):
            match_day = d
        else:
            try:
                match_day = datetime.fromisoformat(str(d)).date()
            except Exception:
                return None, None
        commence = datetime.combine(match_day, time(12, 0), tzinfo=timezone.utc)

        # Canonicalise — alphabetically-first player is "home".
        home, away, swap = _canonical_pair(winner, loser)
        home_won = (home == winner)

        # Rank-points source prob (Bradley-Terry style).
        winner_pts = row.get("WPts")
        loser_pts = row.get("LPts")
        try:
            winner_pts_f = float(winner_pts) if winner_pts is not None else None
            loser_pts_f = float(loser_pts) if loser_pts is not None else None
        except Exception:
            winner_pts_f = loser_pts_f = None
        if swap:
            home_pts, away_pts = loser_pts_f, winner_pts_f
        else:
            home_pts, away_pts = winner_pts_f, loser_pts_f
        rank_prob = _rank_points_prob(home_pts, away_pts)

        # Closing decimal odds — Pinnacle preferred, Bet365 fallback, Avg fallback.
        psw, psl = row.get("PSW"), row.get("PSL")
        b365w, b365l = row.get("B365W"), row.get("B365L")
        avgw, avgl = row.get("AvgW"), row.get("AvgL")

        lines: list[BookLine] = []
        for book, w_odds, l_odds in (
            ("pinnacle-close", psw, psl),
            ("bet365-close", b365w, b365l),
            ("market-avg-close", avgw, avgl),
        ):
            if w_odds is None or l_odds is None:
                continue
            try:
                w_dec = float(w_odds)
                l_dec = float(l_odds)
            except Exception:
                continue
            if w_dec <= 1.0 or l_dec <= 1.0:
                continue
            # winner=home if swap=False; loser=home if swap=True
            home_dec = w_dec if not swap else l_dec
            away_dec = l_dec if not swap else w_dec
            lines.append(
                BookLine(
                    book=book,
                    side="home",
                    american=_decimal_to_american(home_dec),
                    captured_at=commence,
                    is_opening=False,
                )
            )
            lines.append(
                BookLine(
                    book=book,
                    side="away",
                    american=_decimal_to_american(away_dec),
                    captured_at=commence,
                    is_opening=False,
                )
            )

        tournament = (row.get("Tournament") or "").strip() or self.tour.upper()
        round_ = (row.get("Round") or "").strip()
        match_num = row.get("ATP") or row.get("WTA") or ""
        event_id = (
            f"tennis-data:{self.tour}:{year}:{match_day.isoformat()}:"
            f"{match_num}:{home[:10]}-{away[:10]}"
        ).replace(" ", "_")

        source_probs: list[SourceProb] = []
        if rank_prob is not None:
            source_probs.append(
                SourceProb(
                    source="tennis-rank-bt",
                    home_win_prob=max(0.001, min(0.999, rank_prob)),
                    captured_at=commence,
                    notes="Bradley-Terry on ATP/WTA rank points",
                )
            )
        event = Event(
            event_id=event_id,
            sport=self.tour,
            league=tournament,
            home=home,
            away=away,
            commence_time=commence,
            lines=lines,
            source_probs=source_probs,
            signals=[round_] if round_ else [],
        )
        result = HistoricalResult(
            event_id=event_id,
            sport=self.tour,
            home=home,
            away=away,
            commence_time=commence,
            home_won=home_won,
        )
        return event, result


def split_atp_wta(events: Iterable[Event]) -> tuple[list[Event], list[Event]]:
    atp = [e for e in events if e.sport == "atp"]
    wta = [e for e in events if e.sport == "wta"]
    return atp, wta
