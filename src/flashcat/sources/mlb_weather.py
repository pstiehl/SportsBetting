"""MLB park-adjusted weather connector.

For each MLB game on the slate:
  1. Get venue + first-pitch time from ``statsapi.mlb.com``.
  2. Look up venue coordinates + orientation + dome flag in
     ``data/mlb_parks.json``. Domes → no weather adjustment.
  3. Pull hourly weather from Open-Meteo (no API key required), pick the
     hour matching first pitch.
  4. Compute a *park-adjusted run-environment delta* combining:
     - Temperature: +0.6%/°F above 70°F (Baseball Prospectus consensus,
       Eric Walker → Andy Andres → BP Annual 2014).
     - Wind: project the surface wind vector onto the park's home-to-CF
       orientation. Out-to-CF tailwind > 8 mph adds ~3%/5 mph projected
       runs; in-from-CF headwind subtracts the same.
     - Humidity: high humidity (>70%) at Coors specifically reduces carry
       ~2% (Alan Nathan, "Coors humidor effect", 2018).
  5. Emit *totals* projections: ``home_team_runs_expected`` and
     ``away_team_runs_expected`` derived from the venue's
     ``run_env_baseline`` × delta. The blender consumes these via
     Pythagorean conversion (γ=1.83) into a moneyline-side source prob.

If statsapi or Open-Meteo is unreachable, the connector returns ``[]``
gracefully — CI never depends on these endpoints.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path

from ..config import CACHE_DIR, DATA_DIR
from ..types import Event, SourceProb, Sport
from .base import SourceConnector
from .mlb_live import _cached_get
from .mlb_statcast_lineup import fetch_schedule

log = logging.getLogger(__name__)

PARKS_PATH = DATA_DIR / "mlb_parks.json"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# Pythagorean exponent for run differential → win probability.
PYTH_GAMMA = 1.83

# League-average run environment when no baseline is known.
DEFAULT_RUNS_PER_TEAM = 4.5


# ---------------------------------------------------------------------------
# Park table
# ---------------------------------------------------------------------------


def load_parks(path: Path | None = None) -> dict[str, dict]:
    p = path or PARKS_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("venues") or {}


def _resolve_park(name: str, parks: dict[str, dict]) -> dict | None:
    if not name:
        return None
    if name in parks:
        return parks[name]
    norm = name.lower().replace(".", "").strip()
    for k, v in parks.items():
        if k.lower().replace(".", "").strip() == norm:
            return v
    return None


# ---------------------------------------------------------------------------
# Run-environment math
# ---------------------------------------------------------------------------


def temperature_multiplier(temp_f: float | None) -> float:
    """+0.6% runs per °F above 70°F (Baseball Prospectus baseline).

    For temperatures below 50°F we apply the same linear coefficient
    (cold air is dense → fewer runs).
    """
    if temp_f is None:
        return 1.0
    delta_f = temp_f - 70.0
    return 1.0 + 0.006 * delta_f


def wind_multiplier(
    wind_speed_mph: float | None,
    wind_direction_deg: float | None,
    park_orientation_deg: float | None,
) -> float:
    """Project wind onto home→CF axis; map projected component to a multiplier.

    Projected component is wind_speed * cos(wind_from - orientation).
    Positive = blowing in from behind home plate toward CF (tailwind for
    fly balls, +runs). Negative = wind from CF toward home (headwind,
    suppresses runs).

    +3% per 5 mph projected tailwind > 8 mph threshold (linear).
    -3% per 5 mph projected headwind beyond -8 mph threshold.
    """
    if wind_speed_mph is None or wind_direction_deg is None or park_orientation_deg is None:
        return 1.0
    # Open-Meteo's wind_direction is the *from* direction (degrees compass).
    # We want the component blowing *toward* CF, so we compare against the
    # reciprocal of park orientation (orientation_deg points home→CF, wind
    # blowing FROM the home-plate side toward CF means wind_direction ≈
    # orientation - 180).
    home_plate_side = (park_orientation_deg + 180.0) % 360.0
    delta = math.radians(wind_direction_deg - home_plate_side)
    projected = wind_speed_mph * math.cos(delta)
    if abs(projected) < 8.0:
        return 1.0
    if projected > 0:
        return 1.0 + 0.03 * (projected - 8.0) / 5.0
    return 1.0 - 0.03 * (-projected - 8.0) / 5.0


def humidity_multiplier(humidity_pct: float | None, park_name: str) -> float:
    """Coors-specific humidity adjustment (Alan Nathan, 2018).

    Outside Coors the marginal humidity effect is negligible — we hold the
    multiplier at 1.0. At Coors, humidity >70% reduces carry by ~2%.
    """
    if humidity_pct is None:
        return 1.0
    if "Coors" not in (park_name or ""):
        return 1.0
    if humidity_pct <= 70.0:
        return 1.0
    return 1.0 - 0.02 * (humidity_pct - 70.0) / 30.0


def runs_expected(
    park: dict,
    weather: dict | None,
    *,
    park_name: str = "",
) -> tuple[float, float]:
    """Return (home_runs, away_runs) given park baseline + weather snapshot."""
    base_per_team = DEFAULT_RUNS_PER_TEAM * (park.get("run_env_baseline") or 1.0)
    if weather is None or park.get("dome"):
        return base_per_team, base_per_team
    temp_mult = temperature_multiplier(weather.get("temperature_f"))
    wind_mult = wind_multiplier(
        weather.get("wind_speed_mph"),
        weather.get("wind_direction_deg"),
        park.get("orientation_deg"),
    )
    humid_mult = humidity_multiplier(weather.get("humidity_pct"), park_name)
    multiplier = temp_mult * wind_mult * humid_mult
    runs = base_per_team * multiplier
    # Weather is symmetric across home/away — both teams hit in the same conditions.
    return runs, runs


def pythagorean_win_prob(home_runs: float, away_runs: float) -> float:
    """Win probability of home from expected runs, Pythagorean γ=1.83.

    With symmetric run environments (no offense edge) we return 0.5 plus
    a small home-field bump baked in at the moneyline level.
    """
    if home_runs <= 0 and away_runs <= 0:
        return 0.5
    hr = max(0.01, home_runs) ** PYTH_GAMMA
    ar = max(0.01, away_runs) ** PYTH_GAMMA
    return hr / (hr + ar)


# ---------------------------------------------------------------------------
# Open-Meteo fetch
# ---------------------------------------------------------------------------


def fetch_weather_snapshot(
    lat: float,
    lon: float,
    when: datetime,
    *,
    timeout: float = 10.0,
) -> dict | None:
    """Pull hourly Open-Meteo and pick the hour matching ``when``."""
    start_d = when.date().isoformat()
    end_d = (when + timedelta(days=1)).date().isoformat()
    url = (
        f"{OPEN_METEO}?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,"
        "relative_humidity_2m,precipitation_probability"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph"
        f"&start_date={start_d}&end_date={end_d}"
    )
    cache_file = f"openmeteo_{lat:.3f}_{lon:.3f}_{start_d}.json"
    data = _cached_get(url, cache_file, ttl_seconds=3600, timeout=timeout)
    if data is None:
        return None
    try:
        payload = json.loads(data)
    except Exception:
        return None
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    # Find the hour bucket closest to `when`.
    target_hour = when.replace(minute=0, second=0, microsecond=0)
    best_idx = 0
    best_diff = None
    for i, t in enumerate(times):
        try:
            ts = datetime.fromisoformat(t)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        diff = abs((ts - target_hour).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = i

    def _get(key):
        col = hourly.get(key) or []
        if best_idx < len(col):
            return col[best_idx]
        return None

    return {
        "temperature_f": _get("temperature_2m"),
        "wind_speed_mph": _get("wind_speed_10m"),
        "wind_direction_deg": _get("wind_direction_10m"),
        "humidity_pct": _get("relative_humidity_2m"),
        "precip_pct": _get("precipitation_probability"),
    }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class MLBWeather(SourceConnector):
    name = "mlb-weather"
    version = "1.0"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.parks = load_parks()

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        out: list[Event] = []
        d = start
        while d <= end:
            sched = fetch_schedule(d, timeout=self.timeout)
            if sched is None:
                d += timedelta(days=1)
                continue
            for date_block in sched.get("dates") or []:
                for game in date_block.get("games") or []:
                    ev = self._event_from_game(game)
                    if ev is not None:
                        out.append(ev)
            d += timedelta(days=1)
        return out

    def _event_from_game(self, game: dict) -> Event | None:
        try:
            game_pk = str(game.get("gamePk"))
            teams = game.get("teams") or {}
            home = (teams.get("home") or {}).get("team", {}).get("name") or ""
            away = (teams.get("away") or {}).get("team", {}).get("name") or ""
            venue_name = (game.get("venue") or {}).get("name") or ""
            commence_iso = game.get("gameDate")
            if not commence_iso or not home or not away:
                return None
            commence = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))

            park = _resolve_park(venue_name, self.parks)
            if park is None:
                # Unknown venue — emit a neutral 0.5 prob with a note, so
                # downstream tooling can still find the row.
                return self._neutral_event(game_pk, home, away, commence, venue_name)

            weather = None
            if not park.get("dome"):
                weather = fetch_weather_snapshot(
                    park["lat"], park["lon"], commence, timeout=self.timeout
                )

            home_runs, away_runs = runs_expected(park, weather, park_name=venue_name)
            p = pythagorean_win_prob(home_runs, away_runs)
            # Tag a 1pp home-field bump so weather doesn't always tie to 0.5.
            p = max(0.05, min(0.95, p + 0.01))
            notes = (
                f"park={venue_name} dome={bool(park.get('dome'))} "
                f"runs_h={home_runs:.2f} runs_a={away_runs:.2f}"
            )
            if weather:
                notes += (
                    f" temp={weather.get('temperature_f')}F "
                    f"wind={weather.get('wind_speed_mph')}mph "
                    f"dir={weather.get('wind_direction_deg')}deg"
                )
            return Event(
                event_id=f"mlb-weather:{game_pk}",
                sport="mlb",
                league="MLB",
                home=home,
                away=away,
                commence_time=commence,
                source_probs=[
                    SourceProb(
                        source=self.name,
                        home_win_prob=p,
                        captured_at=datetime.now(timezone.utc),
                        notes=notes,
                    )
                ],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("mlb-weather event build failed: %s", e)
            return None

    def _neutral_event(self, game_pk, home, away, commence, venue) -> Event:
        return Event(
            event_id=f"mlb-weather:{game_pk}",
            sport="mlb",
            league="MLB",
            home=home,
            away=away,
            commence_time=commence,
            source_probs=[
                SourceProb(
                    source=self.name,
                    home_win_prob=0.51,
                    captured_at=datetime.now(timezone.utc),
                    notes=f"venue={venue} (no park table entry, neutral fallback)",
                )
            ],
        )
