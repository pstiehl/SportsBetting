"""Shared connector base class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date

from ..types import Event, Sport

log = logging.getLogger(__name__)


class SourceConnector(ABC):
    """A connector pulls Events (with prob and/or line data) from a single source."""

    name: str = "abstract"
    version: str = "0"
    is_live: bool = False  # False means stub-only / not implemented in Phase 1

    @abstractmethod
    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        """Return a list of Events with whatever prob/line data this source has."""
        ...
