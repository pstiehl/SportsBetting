"""Source connectors. Each connector is a SourceConnector subclass."""

from .base import SourceConnector
from .bovada import Bovada
from .espn import ESPNScoreboard
from .fanduel import FanDuel
from .odds_api import TheOddsAPI
from .polymarket import Polymarket
from .nflverse import NFLverseHistorical
from .tennis_history import TennisDataHistorical
from .nba_history import FiveThirtyEightNBAHistorical
from .stubs import (
    PinnacleStub,
    DraftKingsStub,
    FanDuelStub,
    KalshiStub,
    FiveThirtyEightStub,
    MasseyStub,
    KenPomStub,
    BPIStub,
)

ALL_CONNECTORS: list[type[SourceConnector]] = [
    TheOddsAPI,
    Bovada,
    FanDuel,
    ESPNScoreboard,
    Polymarket,
    NFLverseHistorical,
    TennisDataHistorical,
    FiveThirtyEightNBAHistorical,
    PinnacleStub,
    DraftKingsStub,
    FanDuelStub,
    KalshiStub,
    FiveThirtyEightStub,
    MasseyStub,
    KenPomStub,
    BPIStub,
]

__all__ = [
    "SourceConnector",
    "ALL_CONNECTORS",
    "TheOddsAPI",
    "Bovada",
    "FanDuel",
    "ESPNScoreboard",
    "Polymarket",
    "NFLverseHistorical",
    "TennisDataHistorical",
    "FiveThirtyEightNBAHistorical",
    "PinnacleStub",
    "DraftKingsStub",
    "FanDuelStub",
    "KalshiStub",
    "FiveThirtyEightStub",
    "MasseyStub",
    "KenPomStub",
    "BPIStub",
]
