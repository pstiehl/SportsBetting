"""Core data types — Events, Probabilities, Bets, Outcomes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Side = Literal["home", "away"]

SPORTS = ("nfl", "nba", "mlb", "nhl", "cfb", "cbb", "atp", "wta")
Sport = Literal["nfl", "nba", "mlb", "nhl", "cfb", "cbb", "atp", "wta"]


class BookLine(BaseModel):
    """A single sportsbook's moneyline on a single side at a single point in time."""

    book: str
    side: Side
    american: int  # e.g. -150, +130
    captured_at: datetime
    is_opening: bool = False

    @property
    def implied_prob(self) -> float:
        """Implied probability from American odds (no vig removal at this layer)."""
        return american_to_prob(self.american)


class SourceProb(BaseModel):
    """One source's win-probability estimate for the home side of an event."""

    source: str
    home_win_prob: float = Field(ge=0.0, le=1.0)
    captured_at: datetime
    notes: str = ""
    # Optional side-channel dict for source-specific structured data.
    # Example: mlb-statcast-lineup uses it to carry per-batter contribution
    # rows so the explainer can surface specific matchup-driving batters
    # without re-running Statcast. Default None to keep wire-compat with
    # connectors that don't need it.
    metadata: Optional[dict] = None


class Event(BaseModel):
    """A single sporting event with all collected probability + line data."""

    event_id: str
    sport: Sport
    league: str = ""
    home: str
    away: str
    commence_time: datetime

    # Source probabilities (per source name → home win prob).
    source_probs: list[SourceProb] = Field(default_factory=list)

    # Lines across books (opening + current).
    lines: list[BookLine] = Field(default_factory=list)

    # Set by model after blending.
    blended_home_prob: Optional[float] = None
    pick: Optional[Side] = None
    pick_prob: Optional[float] = None  # blended prob of the side we picked

    # Active signals (string labels).
    signals: list[str] = Field(default_factory=list)


class HistoricalResult(BaseModel):
    """Realized outcome of a past event, used in backtests."""

    event_id: str
    sport: Sport
    home: str
    away: str
    commence_time: datetime
    home_won: bool
    home_score: Optional[int] = None
    away_score: Optional[int] = None


class Bet(BaseModel):
    """A simulated $100 flat bet."""

    event_id: str
    side: Side
    stake: float = 100.0
    american_price: int  # the price we'd have gotten if we'd bet
    won: Optional[bool] = None
    profit: Optional[float] = None  # signed P&L on this bet


# --- Odds helpers ---------------------------------------------------------


def american_to_prob(american: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if american == 0:
        return 0.5
    if american > 0:
        return 100.0 / (american + 100.0)
    return (-american) / ((-american) + 100.0)


def american_to_decimal(american: int) -> float:
    """Convert American to decimal odds."""
    if american == 0:
        return 1.0
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / (-american)


def american_to_profit(american: int, stake: float = 100.0) -> float:
    """Profit on a winning bet at given American odds."""
    return stake * (american_to_decimal(american) - 1.0)


def devig_two_way(p_home: float, p_away: float) -> tuple[float, float]:
    """Multiplicative de-vig of two implied probs."""
    total = p_home + p_away
    if total <= 0:
        return 0.5, 0.5
    return p_home / total, p_away / total
