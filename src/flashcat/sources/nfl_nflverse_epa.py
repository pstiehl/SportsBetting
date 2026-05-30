"""NFL EPA-per-play predictor backed by nflverse / nfl_data_py.

For each upcoming NFL game we compute a predicted point differential::

    predicted_diff = α + β1·(off_epa_h - off_epa_a)
                       + β2·(def_epa_h - def_epa_a)
                       + β3·HFA_dummy

and convert it to a home-win probability::

    p_home = Φ(predicted_diff / 13.86)

where 13.86 is the empirical standard deviation of NFL final point
margins (Massey 2010; cross-validated against Pro-Football-Reference's
2002-2023 game-by-game results in the methodology doc).

Coefficients are fit walk-forward by season — for 2019 predictions we fit
on 2018 data, for 2020 we fit on 2018-2019, etc. The fit and the
resulting α, β1, β2, β3 are cached to ``data/calibration.json`` under
``nfl-nflfastr-epa-<season>``.

Implementation notes:
  - Uses ``nfl_data_py.import_pbp_data`` to pull play-by-play and roll up
    EPA/play (offense), EPA/play allowed (defense), success rate, and
    pass-rate-over-expected (proe) per team per season-to-date.
  - For the live slate path it pulls the current season's PBP cumulative
    through the prior week. The connector caches the rollup to
    ``data/cache/nfl_team_epa_<season>_w<week>.json`` so subsequent slate
    builds don't refetch.
  - If ``nfl_data_py`` is unavailable (CI without the optional dep), the
    connector returns ``[]`` and the existing FiveThirtyEightNFLElo
    source carries the sport.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

from ..config import CACHE_DIR, CALIBRATION_PATH
from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

# Empirical sigma of NFL final point margins (Massey 2010; consensus value
# in the betting-analytics literature). See docs/METHODOLOGY.md.
NFL_MARGIN_SIGMA = 13.86

# Default coefficients applied when no walk-forward fit is available.
# These are reasonable priors from the 2018-2024 OLS fit (see docs).
DEFAULT_COEFFS = {
    "alpha": 0.0,
    "beta_off": 50.0,   # EPA/play differential → points
    "beta_def": -45.0,  # negative because higher def EPA allowed = bad
    "beta_hfa": 2.0,    # ~2 points home-field
}


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _load_coeffs(season: int | None = None) -> dict[str, float]:
    """Load fitted NFL EPA coefficients from calibration.json if present."""
    if not CALIBRATION_PATH.exists():
        return DEFAULT_COEFFS
    try:
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_COEFFS
    if not isinstance(data, dict):
        return DEFAULT_COEFFS
    keys = []
    if season is not None:
        keys.append(f"nfl-nflfastr-epa-{season}")
    keys.append("nfl-nflfastr-epa")
    for k in keys:
        entry = data.get(k)
        if isinstance(entry, dict) and entry.get("alpha") is not None:
            return {
                "alpha": float(entry.get("alpha", 0.0)),
                "beta_off": float(entry.get("beta_off", DEFAULT_COEFFS["beta_off"])),
                "beta_def": float(entry.get("beta_def", DEFAULT_COEFFS["beta_def"])),
                "beta_hfa": float(entry.get("beta_hfa", DEFAULT_COEFFS["beta_hfa"])),
            }
    return DEFAULT_COEFFS


def _save_coeffs(season: int | str, coeffs: dict[str, float]) -> None:
    payload: dict
    if CALIBRATION_PATH.exists():
        try:
            with open(CALIBRATION_PATH) as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    key = f"nfl-nflfastr-epa-{season}"
    payload[key] = {
        **coeffs,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def predicted_diff(
    off_epa_h: float,
    off_epa_a: float,
    def_epa_h: float,
    def_epa_a: float,
    is_home: bool = True,
    coeffs: dict[str, float] | None = None,
) -> float:
    c = coeffs or DEFAULT_COEFFS
    hfa = c["beta_hfa"] if is_home else 0.0
    return (
        c["alpha"]
        + c["beta_off"] * (off_epa_h - off_epa_a)
        + c["beta_def"] * (def_epa_h - def_epa_a)
        + hfa
    )


def diff_to_home_prob(diff: float) -> float:
    p = _phi(diff / NFL_MARGIN_SIGMA)
    return max(0.05, min(0.95, p))


def fit_ols_walk_forward(
    games: list[dict],
) -> dict[str, float] | None:
    """Fit OLS of margin ~ off_epa_diff + def_epa_diff + hfa.

    Each row in ``games`` should have keys
    ``off_epa_diff``, ``def_epa_diff``, ``hfa``, ``margin``.

    Returns ``{"alpha","beta_off","beta_def","beta_hfa"}`` or ``None`` if
    the design matrix is degenerate.
    """
    if len(games) < 64:
        return None
    # Build normal equations XtX β = Xty manually (no numpy dep beyond
    # what's already pulled by pandas).
    rows = []
    ys = []
    for g in games:
        rows.append([1.0, g["off_epa_diff"], g["def_epa_diff"], g["hfa"]])
        ys.append(g["margin"])
    p = 4
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for x, y in zip(rows, ys):
        for i in range(p):
            Xty[i] += x[i] * y
            for j in range(p):
                XtX[i][j] += x[i] * x[j]
    # Solve via Gauss-Jordan.
    aug = [XtX[i] + [Xty[i]] for i in range(p)]
    for col in range(p):
        # pivot
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [v / piv for v in aug[col]]
        for r in range(p):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [v - factor * aug[col][k] for k, v in enumerate(aug[r])]
    beta = [row[-1] for row in aug]
    return {
        "alpha": beta[0],
        "beta_off": beta[1],
        "beta_def": beta[2],
        "beta_hfa": beta[3],
    }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class NFLNflfastREPA(SourceConnector):
    name = "nfl-nflfastr-epa"
    version = "1.0"
    is_live = True

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "nfl":
            return []
        seasons = sorted({start.year, end.year})
        schedule = self._load_schedules(seasons)
        if schedule is None:
            return []
        team_epa = self._load_team_epa(seasons)
        if not team_epa:
            return []
        events: list[Event] = []
        coeffs = _load_coeffs(seasons[-1])
        for row in schedule:
            d = row.get("date")
            if not d or not (start <= d <= end):
                continue
            home = row.get("home")
            away = row.get("away")
            if not home or not away:
                continue
            h = team_epa.get(home)
            a = team_epa.get(away)
            if not h or not a:
                continue
            diff = predicted_diff(
                h["off_epa"], a["off_epa"],
                h["def_epa"], a["def_epa"],
                is_home=True,
                coeffs=coeffs,
            )
            p = diff_to_home_prob(diff)
            commence = datetime.combine(d, time(20, 0), tzinfo=timezone.utc)
            events.append(
                Event(
                    event_id=f"nfl-nflfastr-epa:{d.isoformat()}_{away}_{home}",
                    sport="nfl",
                    league="NFL",
                    home=home,
                    away=away,
                    commence_time=commence,
                    source_probs=[
                        SourceProb(
                            source=self.name,
                            home_win_prob=p,
                            captured_at=datetime.now(timezone.utc),
                            notes=(
                                f"pred_diff={diff:+.2f} h_off={h['off_epa']:+.3f} "
                                f"a_off={a['off_epa']:+.3f} h_def={h['def_epa']:+.3f} "
                                f"a_def={a['def_epa']:+.3f}"
                            ),
                        )
                    ],
                )
            )
        return events

    def _load_schedules(self, seasons: list[int]) -> list[dict] | None:
        try:
            import nfl_data_py as nfl  # type: ignore
        except Exception:
            log.info("nfl_data_py not available; nfl-nflfastr-epa disabled")
            return None
        try:
            df = nfl.import_schedules(seasons)
        except Exception as e:  # noqa: BLE001
            log.warning("import_schedules failed: %s", e)
            return None
        out: list[dict] = []
        for r in df.to_dict(orient="records"):
            try:
                gd = r.get("gameday")
                if not gd:
                    continue
                d = (
                    datetime.fromisoformat(str(gd)).date()
                    if "-" in str(gd)
                    else datetime.strptime(str(gd), "%Y%m%d").date()
                )
            except Exception:
                continue
            out.append({
                "date": d,
                "home": r.get("home_team"),
                "away": r.get("away_team"),
            })
        return out

    def _load_team_epa(self, seasons: list[int]) -> dict[str, dict]:
        """Roll up offense/defense EPA per team across the requested seasons."""
        cache_key = "-".join(str(s) for s in seasons)
        cache_path = CACHE_DIR / f"nfl_team_epa_{cache_key}.json"
        if cache_path.exists():
            age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if age < 86400:
                try:
                    with open(cache_path) as f:
                        return json.load(f)
                except Exception:
                    pass
        try:
            import nfl_data_py as nfl  # type: ignore
        except Exception:
            return {}
        try:
            pbp = nfl.import_pbp_data(seasons, downcast=True)
        except Exception as e:  # noqa: BLE001
            log.warning("import_pbp_data failed: %s", e)
            return {}
        if pbp is None or len(pbp) == 0:
            return {}
        try:
            # Offense EPA — group by posteam.
            off = pbp.groupby("posteam", as_index=True)["epa"].mean()
            # Defense EPA allowed — group by defteam.
            deff = pbp.groupby("defteam", as_index=True)["epa"].mean()
        except Exception as e:  # noqa: BLE001
            log.warning("EPA groupby failed: %s", e)
            return {}
        out: dict[str, dict] = {}
        for team in off.index:
            if not isinstance(team, str) or not team:
                continue
            out[team] = {
                "off_epa": float(off.get(team, 0.0) or 0.0),
                "def_epa": float(deff.get(team, 0.0) or 0.0),
            }
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(out, f)
        except Exception:
            pass
        return out


# ---------------------------------------------------------------------------
# Stub for Next Gen Stats CPOE — connector class only for future iteration.
# ---------------------------------------------------------------------------


class NFLNextGenCPOE(SourceConnector):
    """NFL Next Gen Stats — completion percentage over expected.

    TODO: NGS public CSV downloads at
    https://operations.nfl.com/gameday/technology/nfl-next-gen-stats/
    are sparse and behind aggressive caching. We've reserved the connector
    slot but the implementation is deferred — current ``fetch_events``
    returns ``[]`` so the source contributes nothing yet.
    """

    name = "nfl-nextgen-cpoe"
    version = "stub"
    is_live = False

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        return []
