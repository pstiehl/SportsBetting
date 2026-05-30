# Flashcat Betting

> Probability-blended sports betting research model. Picks come from a
> weighted blend of publicly available win-probability sources; staking
> uses **1/4 Kelly fractional** by default with a **3pp edge threshold**
> over the devigged market. Coin-flips get skipped, not bet.

> **2026-05-29:** Replaced the v1 "flat-$100-on-everything" rule (which is
> structurally -vig) with the Kelly-gated rule. Added MLB + tennis live
> sources, fail-loud builds (no more silent sample fallback), and multi-sport
> historical backtest (NFL + ATP + WTA). See `PHIL_PLAN.md`.

<p align="center"><img src="docs/assets/flashcat-logo.svg" alt="Flashcat" width="180"></p>

[**Live site →**](https://pstiehl.github.io/SportsBetting/) · [Methodology (math)](docs/METHODOLOGY.md) · [Methodology (site)](https://pstiehl.github.io/SportsBetting/methodology.html) · [Backtest](https://pstiehl.github.io/SportsBetting/backtest.html) · [Sources](https://pstiehl.github.io/SportsBetting/source-scoreboard.html)

## What Flashcat does

1. **Pulls per-event win probabilities** from multiple public sources (The Odds API, ESPN, Polymarket, the devigged market consensus from book moneylines, and per-sport historical models like ATP/WTA rank-points).
2. **Blends them** with a weighted average. Weights iterate from observed accuracy.
3. **Picks the higher-blended-probability side** and stakes a fractional Kelly bet, **only when edge over the devigged market exceeds the threshold** (default 3pp). Otherwise: no bet.
4. **Backtests every source** independently, tracking **Brier score** (calibration) and **ROI** (profit ÷ wagered).
5. **Reweights sources** after every backtest via a softmax of negative Brier — better-calibrated sources get more weight.
6. **Flags two signals** Phil cares about:
   - **Chalk-overpriced** (favorite-longshot bias): market implied probability on the favorite is more than 5 percentage points above the model.
   - **Sharp money / reverse-line-movement + cross-book dispersion**: opening vs current moneylines moved against the line, or books disagree on the dog by > 4 percentage points.

## What's in this repo

```
flashcat-betting/
├── src/flashcat/
│   ├── sources/          # Source connectors (live + stubbed)
│   ├── model/            # Blending, picks, adaptive reweighting
│   ├── signals/          # Chalk-overpriced + RLM + dispersion
│   ├── backtest/         # $100-flat simulation + scoreboard
│   ├── site/             # Logo generator + Jinja templates
│   ├── build_site.py     # Renders the static site under docs/
│   └── cli.py            # `python -m flashcat <command>`
├── docs/                 # GitHub Pages source (committed)
├── data/
│   ├── samples/          # Committed fallback data so the model runs offline
│   ├── source_weights.json
│   ├── source_scoreboard.json
│   └── flashcat.db       # SQLite ledger (created on first run; gitignored)
├── tests/                # 31 pytest cases — odds math, blend, picks, signals, backtest, site smoke
├── RUN_ME_MAC.command    # Double-click on macOS
└── RUN_ME_WINDOWS.bat    # Double-click on Windows
```

## Source status

| Source | Status | What it provides |
|---|---|---|
| `the-odds-api` | **live** (key required for prod; opt-in samples for offline dev) | Moneylines across US books; in-season sports auto-detected |
| `espn-scoreboard` | **live** | Events + ESPN's `predictor.gameProjection` for team sports; per-match draws for ATP/WTA |
| `polymarket` | **live** | Crowd-implied probability from active sports markets |
| `nflverse` | **live** | Historical NFL schedules + moneylines via `nfl_data_py` |
| `tennis-data` | **live** | Historical ATP/WTA matches with Pinnacle/Bet365 closing odds (tennis-data.co.uk) |
| `tennis-rank-bt` | **live** | Bradley-Terry probability from ATP/WTA rank points |
| `fivethirtyeight-nba-elo` | **live** | 538 historical NBA Elo-based pre-game forecasts (through 2014-15) |
| `market-close` (synthetic) | **live** | Devigged consensus probability across book averages |
| `datagolf-sg` | **live** (key-gated) | DataGolf strokes-gained pre-tournament forecasts → PGA head-to-head matchup probabilities (free-tier endpoints only) |
| `pga-espn-scoreboard` | **live** | ESPN PGA leaderboard → score-gap logistic for in-progress tournament H2H matchups |
| `pga-market-consensus` | **live** (key-gated) | Odds API outright winner consensus for the four golf majors |
| `pinnacle` | stub | Phase 2 |
| `draftkings`, `fanduel` | stub | Phase 2 (page scrapes) |
| `kalshi` | stub | Phase 2 |
| `fivethirtyeight` | stub | Phase 2 (model probabilities) |
| `massey-sagarin`, `kenpom`, `espn-bpi` | stub | Phase 2 |

## Run locally

```bash
# 1. Clone + cd
git clone https://github.com/pstiehl/SportsBetting.git
cd SportsBetting

# 2. Either double-click RUN_ME_MAC.command / RUN_ME_WINDOWS.bat, OR:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Full pipeline (backtest → reweight → build site)
PYTHONPATH=src python -m flashcat all

# 4. Just today's slate (live pull + site build):
PYTHONPATH=src python -m flashcat build

# 5. Just the backtest:
PYTHONPATH=src python -m flashcat backtest --start 2023-09-01 --end 2024-02-15 --sport nfl

# 6. Just reweighting:
PYTHONPATH=src python -m flashcat reweight

# 7. Tests
PYTHONPATH=src python -m pytest tests/ -v
```

### Optional environment variables

Set in a `.env` file at the repo root (gitignored):

```
THE_ODDS_API_KEY=your_key_here  # https://the-odds-api.com/, free tier available
```

Without `THE_ODDS_API_KEY`, the OddsAPI connector returns `[]` and the pipeline
**fails loud** if no other live source returns events for any in-season sport.
That's intentional — silently shipping stale samples to the live site was the
v1 bug that put fake NBA games on `pstiehl.github.io/SportsBetting` in May 2026.
For offline local builds, opt in with `FLASHCAT_USE_SAMPLES=1` to read the
`data/samples/odds_api_sample.example.json` fallback.

## The two hunches as first-class features

### 1. "Favorite is a sucker's bet" (chalk-overpriced)

Implemented two ways:

- **Tie-breaker rule:** when blended probability is in **[0.48, 0.52]**, bet the underdog by moneyline. The market premium on the slight favorite has historically been overpriced in coin-flip games.
- **Standalone signal:** when devigged market implied probability on the favorite exceeds the model's by more than 5 percentage points, flag `chalk-overpriced`. Surface a red badge on the event card; carve out a `chalk-only` slice in the backtest.

### 2. Sharp-money / reverse-line-movement + book dispersion

- **RLM:** if home implied probability moved ≥ 2 percentage points away from opening, flag the direction of the move. Sharp money pushes lines against the public bet distribution.
- **Cross-book dispersion:** if the spread between the highest and lowest implied probability on the underdog across books exceeds 4 percentage points, flag `book-dispersion-dog`. A wide spread means soft books are exploitable.

Both signals show up on the event card as badges and get their own ROI slice on the backtest page.

## GitHub Pages

The site is published from the `docs/` directory on `main`. After the Phase-1 PR merges, **flip the Pages setting**:

- Repo Settings → Pages
- Source: **Deploy from a branch**
- Branch: **main**, Folder: **/docs**, Save

Site will be at: **https://pstiehl.github.io/SportsBetting/**

## What's out of scope (Phase 1)

- Real-money betting integration
- Live in-game model (pre-game only)
- Scraping behind login walls
- Sport-specific deep models — Flashcat is the **blender + backtest harness**; sport-specific models are Phase 2
- Account / bankroll / Kelly sizing — universal $100 flat is the entire bet sizing layer in Phase 1

## License

Research only. Not investment or gambling advice. Confirm legal jurisdiction before placing real wagers.
