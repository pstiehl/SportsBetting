"""Tests for the two signal detectors."""

from datetime import datetime, timezone, timedelta

from flashcat.signals.favlong import detect as detect_favlong
from flashcat.signals.sharp import detect as detect_sharp, detect_rlm, detect_dispersion
from flashcat.types import BookLine, Event, SourceProb


def _now():
    return datetime(2024, 1, 7, tzinfo=timezone.utc)


def test_chalk_overpriced_fires():
    """Market has home at -250 (~71% implied, ~70.4% devigged) but model says 0.55.
    Devigged favorite implied > blended favorite prob + 0.05 → fires."""
    now = _now()
    ev = Event(
        event_id="x", sport="nfl", home="A", away="B", commence_time=now,
        source_probs=[],
        lines=[
            BookLine(book="dk", side="home", american=-250, captured_at=now),
            BookLine(book="dk", side="away", american=210, captured_at=now),
        ],
        blended_home_prob=0.55,
    )
    assert detect_favlong(ev) == "chalk-overpriced"


def test_chalk_overpriced_no_fire_when_model_agrees():
    now = _now()
    ev = Event(
        event_id="x", sport="nfl", home="A", away="B", commence_time=now,
        source_probs=[],
        lines=[
            BookLine(book="dk", side="home", american=-200, captured_at=now),
            BookLine(book="dk", side="away", american=170, captured_at=now),
        ],
        blended_home_prob=0.68,
    )
    assert detect_favlong(ev) is None


def test_rlm_toward_away_fires(rlm_event):
    # Open home implied 60.0%, close avg < 58% → moved toward away
    sig = detect_rlm(rlm_event)
    assert sig == "reverse-line-movement-toward-away"


def test_rlm_no_data():
    now = _now()
    ev = Event(
        event_id="x", sport="nfl", home="A", away="B", commence_time=now,
        lines=[BookLine(book="dk", side="home", american=-150, captured_at=now)],
    )
    assert detect_rlm(ev) is None


def test_book_dispersion_fires():
    now = _now()
    # Underdog (away): one book +120 (45.5%), another +200 (33.3%) — spread > 4%
    ev = Event(
        event_id="x", sport="nfl", home="A", away="B", commence_time=now,
        lines=[
            BookLine(book="dk", side="home", american=-150, captured_at=now),
            BookLine(book="dk", side="away", american=120, captured_at=now),
            BookLine(book="fd", side="home", american=-220, captured_at=now),
            BookLine(book="fd", side="away", american=200, captured_at=now),
        ],
    )
    assert detect_dispersion(ev) == "book-dispersion-dog"


def test_book_dispersion_no_fire_tight():
    now = _now()
    ev = Event(
        event_id="x", sport="nfl", home="A", away="B", commence_time=now,
        lines=[
            BookLine(book="dk", side="home", american=-150, captured_at=now),
            BookLine(book="dk", side="away", american=130, captured_at=now),
            BookLine(book="fd", side="home", american=-152, captured_at=now),
            BookLine(book="fd", side="away", american=132, captured_at=now),
        ],
    )
    assert detect_dispersion(ev) is None


def test_detect_sharp_combines(rlm_event):
    out = detect_sharp(rlm_event)
    assert "reverse-line-movement-toward-away" in out
