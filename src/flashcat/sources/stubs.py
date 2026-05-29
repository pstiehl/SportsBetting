"""Stub connectors — clean interfaces, return empty in Phase 1."""

from __future__ import annotations

import logging
from datetime import date

from ..types import Event, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)


class _Stub(SourceConnector):
    is_live = False
    version = "0-stub"

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        log.debug("%s.fetch_events: not implemented in Phase 1", self.name)
        return []


class PinnacleStub(_Stub):
    name = "pinnacle"


class DraftKingsStub(_Stub):
    name = "draftkings"


class FanDuelStub(_Stub):
    name = "fanduel"


class KalshiStub(_Stub):
    name = "kalshi"


class FiveThirtyEightStub(_Stub):
    name = "fivethirtyeight"


class MasseyStub(_Stub):
    name = "massey-sagarin"


class KenPomStub(_Stub):
    name = "kenpom"


class BPIStub(_Stub):
    name = "espn-bpi"
