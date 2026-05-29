# Meta-Model Plan — Flashcat as Accuracy-Weighted Aggregator

> Phil's clarification (2026-05-29): *"The goal of all of these is to use every predictive site on the internet that predicts sporting events from a probability standpoint and judge them based on accuracy of predicting past events. That accuracy score should feed the model and weight should be given to the source depending on its accuracy."*

Flashcat is **not** another tipster service. It is a **meta-model**: aggregate every public probability source on the internet, score each by historical accuracy (Brier + log-loss + ROI), and emit a single weighted consensus prediction with full provenance. Until the blended ROI clears 0 on out-of-sample data, the site stays in **research mode** (no live stake recommendations).

## Phase scope (one PR, multi-commit, branch `feat/multi-source-meta-model`)

### Phase A — Research-mode gate (ship first)
Hide stake recommendations site-wide whenever `flashcat-blended.roi < 0`. Show a "🟡 RESEARCH MODE" badge in the header and replace the recommended-plays panel with a research-only callout. Cards still render with sources and the blended prob — just no stake / EV / EDGE banner.

**Files:** `src/flashcat/build_site.py`, `src/flashcat/site/templates/_layout.html`, `src/flashcat/site/templates/index.html`.

**Acceptance:** with current scoreboard (ATP blended ROI = -9.6%, NBA empty, NFL skipped), the site shows RESEARCH MODE and no stake suggestions. Phil never sees a -12% backtested play recommended.

### Phase B — Source inventory (every public probability source)

#### MLB (highest priority — Phil specifically asked)
1. `mlb_538_elo.py` — 538 MLB Elo CSV (archive): `https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv`, mirrored at `https://raw.githubusercontent.com/fivethirtyeight/data/master/mlb-elo/mlb_elo.csv`. Per-game pre-game prob via Elo + rating1_pre/rating2_pre columns. Historical 2010-2022.
2. `mlb_fangraphs.py` — FanGraphs Live Win Probability scraper (current-day only, no historical archive without a paid Plus account — document the gap).
3. `mlb_espn_bpi.py` — extract `competitions[].predictor.homeTeam.gameProjection` from ESPN's MLB scoreboard. Live + historical via dated scoreboards.
4. `mlb_pythagorean.py` — controlled baseline: Pythagorean expectation from season RS/RA. Updated daily.

#### NBA
1. `nba_538_forecasts.py` — 538 NBA forecast CSV: `https://projects.fivethirtyeight.com/nba-model/nba_elo.csv` and `nba_elo_latest.csv`. Per-game Elo, CARM-Elo, RAPTOR variants. **Gold data — multiple model variants per game.**
2. `nba_espn_bpi.py` — ESPN BPI per-game projection (same predictor field).
3. `nba_inpredictable.py` — public win probabilities at `https://stats.inpredictable.com/nba/`. (Best-effort; document if blocked.)
4. Update existing `nba_history.py` (the static 538 nbaallelo CSV through 2014-15) — leave as-is but mark stale.

#### NFL
1. `nfl_538_elo.py` — 538 NFL Elo CSV: `https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv`. Weekly Elo forecasts with per-game win prob from 2002 onward.
2. `nfl_espn_fpi.py` — ESPN FPI per-game projection.
3. `nfl_nfelo.py` — nfeloapp.com derivative Elo (best-effort scrape; document if blocked).

#### Tennis
1. `tennis_jeff_sackmann_elo.py` — proper surface-adjusted Elo from `https://github.com/JeffSackmann/tennis_atp` / `tennis_wta`. Computed iteratively from `atp_matches_YYYY.csv`. Replaces the simple Bradley-Terry rank-points heuristic in `tennis_history.py` with a much better signal.

Each connector:
- Subclasses `SourceConnector` with `is_live`, `fetch_events(start, end, sport)`, and `load_results(start, end)` where applicable.
- HTTP cached under `data/cache/` to avoid re-downloads.
- Live-only sources (FanGraphs current day, ESPN today) implement `fetch_events`; historical sources implement both `fetch_events` (so backtest can replay them) and `load_results`.
- Each source's `name` becomes the key in the scoreboard.

### Phase C — Per-sport backtest harness scoring every source
Edit `src/flashcat/backtest/runner.py`:
- `SPORT_LOADERS` becomes a dict-of-lists: each sport maps to *all* connectors that can produce historical predictions for it. Backtest runs each connector and aggregates `source_probs` across them onto common Events (keyed by date+teams).
- For each (sport, source) tuple, compute Brier, log-loss, accuracy (% correct of pick), ROI at ¼ Kelly with edge threshold, calibration bins.
- The scoreboard JSON gets a `per_sport[sport].sources[source]` block with these stats, plus blended stats and a `per_sport[sport].n_events` count.
- Add MLB to the default `--sports` list.
- Walk-forward only: every connector's `fetch_events(start, end)` returns *that prediction as published before kickoff*. We never re-fit.
- Sources with <50 events graded are flagged `insufficient_data` and excluded from weighting (but still listed).

### Phase D — Source weights from accuracy (per-sport)
Edit `src/flashcat/model/reweight.py`:
- Add `ACCURACY_WEIGHT_MODE` env (`brier` | `log_loss` | `roi` | `brier_roi_hybrid`). Default `brier_roi_hybrid`.
- Hybrid: `0.5 * softmax(-Brier) + 0.5 * softmax(roi)`.
- Compute **per-sport** weights: `weights["by_sport"]["mlb"]["mlb-538-elo"] = 0.42` etc. Plus a `weights["global"]` fallback.
- Exclude sources with `n_events < 50` from the pool.
- `model/blend.py` reads per-sport weights when blending an event (look up `weights["by_sport"][event.sport]`, fall back to `weights["global"]`, fall back to uniform).
- Backwards-compat: `source_weights.json` shape becomes `{"global": {...}, "by_sport": {...}, "schema": "v2"}`. Loader handles v1 (flat dict).

### Phase E — Site additions
- `docs/source-scoreboard.html` renders per-sport tables, each with rows colored by accuracy quartile (top 25% green, middle 50% yellow, bottom 25% red).
- Event detail page shows the per-source contribution table (which sources weighed in, with weight + individual prob).
- Header status badge: 🟢 LIVE BETTING / 🟡 RESEARCH MODE (Phase A wires this up; Phase E polishes).

### Phase F — Ship discipline
- One PR `feat: meta-model — accuracy-weighted aggregation of every public probability source` to `pstiehl/SportsBetting:main`.
- Branch `feat/multi-source-meta-model`.
- Multi-commit, one per phase.
- Tests stay green; add golden-file tests for each new connector under `tests/sources/`.
- After merging, watch the GH Action, fetch the live site, verify the rendered scoreboard.

### Honesty rules
- Walk-forward only — never train weights on overlapping outcomes.
- Source-specific archive windows (538 ended 2023, ESPN BPI starts 2018 etc.) — backtest each connector only over its actual coverage.
- If blended ROI stays negative, keep the research gate ON. Report it honestly.
- Paywalled sources (FanGraphs Plus) → document and skip; never scrape behind a login.

## Implementation order
1. Phase A (research-mode gate) — ship immediately as first commit.
2. Phase B/C wired in tandem per sport: NFL 538 + ESPN FPI → NBA 538 forecasts + ESPN BPI → MLB 538 + Pythagorean + ESPN BPI → tennis Sackmann Elo.
3. Phase D — per-sport weights.
4. Phase E — site polish.
5. Run full multi-sport backtest, capture honest ROI, write PR body.

## Open risks
- 538's archive CSVs are no longer maintained — fine for backtest, useless for forward predictions. The aggregator value is in *combining* archived models with live ESPN / market consensus.
- Some sources don't publish a clean archive; for those we collect forward-only and label `live-only, no backtest` on the scoreboard.
- The Bradley-Terry tennis source is currently a market proxy — replacing it with Sackmann Elo will likely change tennis weights meaningfully.
- WTA blended is +13.4% over 305 bets in the existing backtest. That's likely overfit-by-selection (no edge gate during training). Phase D's per-sport weighting + walk-forward backtest will retest this honestly.
