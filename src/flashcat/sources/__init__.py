"""Source connectors. Each connector is a SourceConnector subclass."""

from .base import SourceConnector
from .bovada import Bovada
from .espn import ESPNScoreboard
from .espn_predictor import ESPNPredictor
from .fanduel import FanDuel
from .fivethirtyeight_archives import (
    FiveThirtyEightMLBElo,
    FiveThirtyEightNBAModern,
    FiveThirtyEightNFLElo,
)
from .mlb_live import DimersMLB, DraftKingsMLB, FanGraphsMLB, PinnacleMoneyline
from .mlb_pythagorean import MLBPythagorean
from .nba_history import FiveThirtyEightNBAHistorical
from .nflverse import NFLverseHistorical
from .odds_api import TheOddsAPI
from .polymarket import Polymarket
from .sackmann_elo import SackmannATPElo, SackmannWTAElo
from .stubs import (
    BPIStub,
    DraftKingsStub,
    FanDuelStub,
    FiveThirtyEightStub,
    KalshiStub,
    KenPomStub,
    MasseyStub,
    PinnacleStub,
)
from .tennis_history import TennisDataHistorical

ALL_CONNECTORS: list[type[SourceConnector]] = [
    TheOddsAPI,
    Bovada,
    FanDuel,
    ESPNScoreboard,
    ESPNPredictor,
    Polymarket,
    NFLverseHistorical,
    TennisDataHistorical,
    SackmannATPElo,
    SackmannWTAElo,
    FiveThirtyEightNBAHistorical,
    FiveThirtyEightMLBElo,
    FiveThirtyEightNFLElo,
    FiveThirtyEightNBAModern,
    MLBPythagorean,
    FanGraphsMLB,
    DimersMLB,
    PinnacleMoneyline,
    DraftKingsMLB,
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
    "ESPNPredictor",
    "Polymarket",
    "NFLverseHistorical",
    "TennisDataHistorical",
    "SackmannATPElo",
    "SackmannWTAElo",
    "FiveThirtyEightNBAHistorical",
    "FiveThirtyEightMLBElo",
    "FiveThirtyEightNFLElo",
    "FiveThirtyEightNBAModern",
    "MLBPythagorean",
    "FanGraphsMLB",
    "DimersMLB",
    "PinnacleMoneyline",
    "DraftKingsMLB",
    "PinnacleStub",
    "DraftKingsStub",
    "FanDuelStub",
    "KalshiStub",
    "FiveThirtyEightStub",
    "MasseyStub",
    "KenPomStub",
    "BPIStub",
]
