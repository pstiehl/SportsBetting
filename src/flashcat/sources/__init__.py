"""Source connectors. Each connector is a SourceConnector subclass."""

from .base import SourceConnector
from .espn import ESPNScoreboard
from .odds_api import TheOddsAPI
from .polymarket import Polymarket
from .nflverse import NFLverseHistorical
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
    ESPNScoreboard,
    Polymarket,
    NFLverseHistorical,
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
    "ESPNScoreboard",
    "Polymarket",
    "NFLverseHistorical",
    "PinnacleStub",
    "DraftKingsStub",
    "FanDuelStub",
    "KalshiStub",
    "FiveThirtyEightStub",
    "MasseyStub",
    "KenPomStub",
    "BPIStub",
]
