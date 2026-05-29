"""Phase-1 first-class signals.

- favorite-longshot bias (chalk-overpriced)
- sharp-money / reverse-line-movement + cross-book dispersion
"""

from .favlong import detect as detect_favlong
from .sharp import detect as detect_sharp

__all__ = ["detect_favlong", "detect_sharp"]
