# Source Inventory

Single-page registry of every prediction source the model reads, with
provenance, license/cost, refresh cadence, and the current verdict from
`paw-reports/sportsbetting/source-accountability-latest.md`.

> This document is REFERENCE. The verdict column updates every Monday via
> `scripts/weekly_source_rescore.sh`. See `docs/AGENT_LOOP.md` for the
> standing cadence.

## Live sources (read at build time)

| Source key | Sport(s) | Provenance | Refresh | Key required | Notes |
|---|---|---|---|---|---|
| `the-odds-api` | nfl, nba, mlb, atp, wta, pga, cfb | https://the-odds-api.com | per-build | `THE_ODDS_API_KEY` | Moneylines + spreads; in-season sports auto-detected via `/sports?all=false` |
| `espn-scoreboard` | nfl, nba, mlb, cfb, atp, wta, pga | https://site.api.espn.com | per-build | none | Events + ESPN `predictor.gameProjection` for team sports |
| `polymarket` | nfl, nba, mlb (active markets only) | https://gamma-api.polymarket.com | per-build | none | Crowd-implied probability |
| `nflverse` | nfl | https://github.com/nflverse/nflverse-data via `nfl_data_py` | per-build | none | Historical schedules + moneylines |
| `tennis-data` | atp, wta | http://www.tennis-data.co.uk/{year}{w}/{year}.xlsx | per-build (cached) | none | Historical results + Pinnacle/Bet365/Avg closing odds |
| `tennis-rank-bt` | atp, wta | derived from tennis-data WPts/LPts | per-build | none | Bradley-Terry probability from rank points |
| `sackmann-atp-elo` | atp | https://github.com/JeffSackmann/tennis_atp | per-build | none | Surface-aware Elo |
| `sackmann-wta-elo` | wta | https://github.com/JeffSackmann/tennis_wta | per-build | none | Surface-aware Elo |
| `fivethirtyeight-nfl-elo` | nfl | 538 archives via Wayback | per-backfill | none | Coverage ends with 2022 season |
| `fivethirtyeight-nfl-qbelo` | nfl | 538 archives via Wayback | per-backfill | none | Same archive limit |
| `fivethirtyeight-nba-elo-modern` | nba | 538 archives | per-backfill | none | Through 2014-15 |
| `fivethirtyeight-nba-raptor` | nba | 538 archives | per-backfill | none | Through 2022-23 |
| `fivethirtyeight-mlb-elo` | mlb | 538 archives | per-backfill | none | Through 2022 |
| `fivethirtyeight-mlb-rating` | mlb | 538 archives | per-backfill | none | Through 2022 |
| `mlb-pythagorean` | mlb | derived from MLB run-differential | per-build | none | Pre-game pythagorean expectation |
| `mlb-statcast-lineup` | mlb | statsapi + Statcast | per-build | none | Live forward-only; requires lineup |
| `mlb-weather` | mlb | Open-Meteo park lat/lng | per-build | none | Weather adjustment |
| `nfl-nflfastr-epa` | nfl | nflverse PBP | per-build | none | Walk-forward EPA refit |
| `cfb-cfbfastr-epa` | cfb | cfbfastR PBP via `nfl_data_py` | per-build | none | Research mode until backtest ROI > +1% |
| `cfb-espn-fpi` | cfb | ESPN FPI scoreboard | per-build | none | ESPN's CFB rating |
| `cfb-market-consensus` | cfb | OddsAPI consensus | per-build | `THE_ODDS_API_KEY` | Devigged book moneylines |
| `nba-bref-srs-pace` | nba | basketball-reference.com | per-build | none | SRS + pace adjustment |
| `pga-datagolf` | pga | https://feeds.datagolf.com | per-build | `DATAGOLF_API_KEY` | SG-based pre-tournament forecasts |
| `pga-espn-bpi` | pga | ESPN PGA scoreboard | per-build | none | In-progress score-gap logistic |
| `pga-market-consensus` | pga | OddsAPI outright | per-build | `THE_ODDS_API_KEY` | Major championship consensus |
| `market-close` (synthetic) | all | derived from book averages | per-build | none | Devigged consensus; the baseline |
| `market-consensus` | all | alias of `market-close` | per-build | none | Plumbed under second name for blender bookkeeping |

## Stub sources (Phase 2)

| Source key | Sport(s) | Why stub | Plan |
|---|---|---|---|
| `pinnacle` | all | requires VPN + scrape | Phase 2 once we have a reliable scrape harness |
| `draftkings`, `fanduel` | all | login wall on most pages | Page scrape Phase 2 |
| `kalshi` | nfl | regulatory restrictions on programmatic access | Phase 2 |
| `massey-sagarin`, `kenpom`, `espn-bpi` | cfb, nba | paywall or aggressive anti-scrape | Phase 2 |
| `fivethirtyeight` (current) | all | site shut down 2023-03 | Replaced by `fivethirtyeight-*-archives` for historical only |

## Observed-external sources (we report, we don't ingest per-event)

| Source key | Sport(s) | Provenance | Why observed-only |
|---|---|---|---|
| `predict.tennis` | atp, wta | https://predict.tennis/prediction-check/ + 2024 season review | Cloudflare-gated, no per-event API; the site self-publishes its hit rates and yields. We record those verbatim in `source_accountability.PREDICT_TENNIS_2024_SELF_REPORT`. |

## Adding a new source

1. Build a connector in `src/flashcat/sources/<source>.py` that returns
   `Event` and (where appropriate) `HistoricalResult` objects.
2. Register it in `SPORT_LOADERS` (in `flashcat/backtest/runner.py`) if it
   has historical coverage, or in `flashcat.cli._live_*_sources()` if it's
   forward-only.
3. Add a backfill script under `scripts/backfill_<sport>_*.py` if the
   source supports historical pulls.
4. Add a row to **this file** under the right table.
5. Run `python -m flashcat source-accountability` to register a verdict.
6. If the verdict is KEEP or KEEP-WITH-CAVEATS, the next `flashcat reweight`
   run will pick up the source automatically. If it's NOISE or DROP, the
   blender will exclude or down-weight it.

## How to read a verdict

| Verdict | What it means | Action |
|---|---|---|
| **KEEP** | ROI > +1% on $100/event flat | Use in blend |
| **KEEP-WITH-CAVEATS** | −3% ≤ ROI ≤ +1% (near break-even) | Use in blend; expect calibration value, not headline profit |
| **NOISE** | Brier ≈ vig territory (0.24-0.25); ROI in the vig range | Down-weight; flag for monthly retrain review |
| **DROP** | Brier ≥ 0.25 (worse than coin flip) OR ROI ≤ −10% | Exclude from blend immediately |
| **INSUFFICIENT-DATA** | < 200 graded events | Hold; revisit at next weekly rescore |

## Current verdict snapshot

See `paw-reports/sportsbetting/source-accountability-latest.md` for the
machine-readable JSON and rendered markdown.
