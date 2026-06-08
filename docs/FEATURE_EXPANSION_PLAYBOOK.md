# Feature Expansion Playbook

> Continuous model-quality improvement, one sport per week, on rotation.

## Why this exists

Phil's directive (2026-05-31): we don't ship a frozen model. We rotate
through every sport on a weekly cycle, re-doing the feature audit →
backfill → walk-forward backtest → loss post-mortem → ship. MLB was
Week 1 — this document is the methodology so anyone (human or sub-agent)
can pick up the next sport in rotation and reproduce the same rigor
without re-deriving the steps.

The point is **honest evaluation, not green dashboards**. If a feature
set loses to vig, that's the finding — we ship it documented, then
iterate. We don't widen the edge gate to hide the result.

## Rotation order

```
Week 1: MLB   (done — see commit feat/mlb-feature-expansion)
Week 2: NBA   (done — see commit feat/nba-feature-expansion)
Week 3: CFB
Week 4: NFL
Week 5: ATP
Week 6: WTA
Week 7: PGA
```

Note on rotation order (2026-06-08): the original document had
`NFL → CFB`. The state file `~/.openclaw/workspace/data/sportsbetting_weekly_state.json`
codifies the operational order as `MLB → NBA → CFB → NFL → ATP → WTA → PGA`
(per Phil's Stiehl-thread direction 2026-06-01). CFB has the bigger
backfilled corpus, so it goes next; NFL feature work waits one cycle.

The full cycle takes ~7 weeks, then restarts. Each sport gets at minimum
one expansion pass per quarter. Sports with surprising deltas (positive
or negative) jump the queue.

## The 6-step playbook

### 1. Feature audit — what we have today

For the sport you're working on, enumerate:

* Every connector in `src/flashcat/sources/` that emits a probability for
  this sport.
* Every feature column the connector internally consumes.
* What's actually in `data/source_history.db.predictions` for this sport
  (a) by source, (b) over what date range, (c) with which optional
  columns populated (`market_close_decimal`, `closing_implied_prob`, etc).
* The current entries in `data/source_scoreboard.json::per_sport.<sport>`
  — what's the blended ROI, the per-source ROI, and which sources are
  in the active weight blend?

Drop this into `docs/<sport>_feature_audit.md` (or append to a
per-sport section in this file). The audit is the input to step 2.

### 2. Feature wish list — what should we have

Phil's MLB list was the prompt:

> "Rolling team scoring + pitching stats (L3/L5/L10/L20), starter
> pitcher splits (handedness, FIP, K/BB/HR rates, IP last 30d, rest days),
> bullpen state, park factor, weather, umpire tendencies, batter rolling
> exit velocity/barrel/whiff, lineup vs pitcher handedness."

That's the **structure of the question**, not the answer for every sport.
For each new sport, derive the analogous catalog under these categories:

| Category | MLB example | What to fill in for sport X |
|---|---|---|
| Rolling team rates | RS/RA L3/L5/L10/L20 | Off/Def efficiency L3/L5/L10 (NBA), EPA/play L4/L8 (NFL), ratings L10 matches (tennis) |
| Individual/positional splits | Starting pitcher FIP/xwOBA vs LHB/RHB | Starting QB QBR, primary scorer usage vs opp defense, recent serve hold % (tennis) |
| Environmental factors | Park run env + weather + temp | Altitude (NBA), wind/precip/dome (NFL), surface (tennis), course difficulty (PGA) |
| Lineup/availability | Probable pitcher, lineup vs RHP | DNPs/load mgmt (NBA), QB starter, injury report (NFL), withdrawal status (tennis) |
| Official/venue tendencies | Plate ump K%/BB% | Whistle profile (NBA refs), home crew bias (NFL refs), umpire pace (tennis) |
| Advanced/Statcast | xwOBA, barrel, whiff | Tracking-data on/off splits (NBA), Next Gen Stats CPOE/sep (NFL), serve+1 shotPlus (tennis) |
| Schedule fatigue | Days rest, B2B series | B2B + miles travelled (NBA), short week (NFL), tournament round + previous match minutes (tennis) |
| Market context | Closing line + line movement | Same — every sport |

The wish list is **deliberately ambitious**. Real Phase-1 implementations
will start with the cheapest-to-backfill ~10 features and add the
expensive ones in Phase 2.

### 3. Walk-forward backtest methodology

Single methodology across every sport:

* **Training window**: rolling 365 days. Configurable per-sport via
  `--train-days` if a shorter season makes 365d unrealistic.
* **Evaluation window**: 30 days. Slide forward by 30d, retrain.
* **Warmup**: 120 days before first eval window (gives rolling features
  enough sample to populate L10/L20 windows).
* **Model**: logistic regression with L2 reg (C=1.0), `StandardScaler`
  feature normalization. Single-pass walk-forward fit per fold.
* **Leakage gate**: asserted in-loop. Every training example must have
  `game_date <= split.train_end` and every eval example must have
  `split.eval_start <= game_date <= split.eval_end`. The
  `WalkForwardSplit.__post_init__` assertion fires if a misconfigured
  fold leaks.
* **Stake**: $100 flat on every model-graded game where the required
  features are populated. **No edge gate.** **No per-sport mode gate.**
  The production gates stay in place for the live site; the backtest
  bypasses them so we get an honest read on raw model quality.
* **Market proxy when real closing odds aren't available**: convert the
  best-available pre-game probability source to decimal odds with a
  standard book hold (default 4.5%). Document this clearly in output —
  it's NOT a real CLV, it's a proxy. Real CLV comes from
  `predictions.closing_implied_prob` once the live ledger captures it.

### 4. Required outputs per sport

Every backtest must produce, at minimum:

| Metric | What it tells us |
|---|---|
| `n_bets` | Sample size. < 200 = no conclusion. |
| `win_rate` | Sanity check vs. mean market implied. |
| `roi` | Headline. Negative = lose to vig. Beat -2% in 3 iterations is the bar. |
| `clv_proxy_pp` | Did the model find calibration improvements the proxy missed? Positive ≠ profitable, but it's necessary. |
| `max_drawdown` | Risk concentration. Big DD = peaky model, dangerous. |
| `sharpe` | Risk-adjusted return per-bet, annualized to 162-game seasons. |
| Loss buckets | Where do the losses come from? `pure_variance` is acceptable; `pitcher_signal_wrong` / `rolling_signal_wrong` = feature-quality problems. |
| Per-year breakdown | Catch regime shifts (juiced ball, rule changes, etc). |

Persist to `data/source_history.db.predictions` under source name
`<sport>-flashcat-v2`, and write a meta row covering the backtest
window. Also drop a JSON summary at `data/<sport>_walk_forward_backtest.json`.

### 5. Loss post-mortem

The simulator buckets every losing bet:

* `pure_variance` — pick prob in [0.45, 0.55]. Coinflip; not actionable.
* `line_moved_against` — model edge vs proxy < 1pp. Vig got us; no real signal.
* `pitcher_signal_wrong` (MLB) / sport-equivalent — feature with strongest
  signal in our direction was wrong. Calibration problem in that feature.
* `rolling_signal_wrong` — rolling rates favored our pick by > 1 run/game
  but we lost. Either rolling rates are noisy at our window size or
  there's a context feature we're missing.
* `generic` — none of the above. Tells us our bucket taxonomy needs work.

For sports other than MLB, add sport-specific loss buckets in
`src/flashcat/mlb_features/simulator.py::_classify_loss` (rename
module to `feature_eval/simulator.py` if it grows multi-sport — see
"Refactor target" below).

### 6. Ship the result honestly

Required deliverables on every weekly PR:

1. **The code** — new feature builder + tests under `tests/test_<sport>_walk_forward.py`.
2. **The data** — backtest JSON in `data/<sport>_walk_forward_backtest.json` (NOT gitignored; it's the receipt).
3. **The PR body** — the headline metrics table, the loss post-mortem
   table, and a one-sentence verdict ("Does this beat the previous
   feature set? Yes/no/coinflip.").
4. **Status doc** — append a row to the rotation tracker in
   `~/.openclaw/workspace/data/sportsbetting_rotation.md` so Ada can
   surface progress to Phil weekly.

## Where to plug new features into the existing pipeline

```
src/flashcat/
├── mlb_features/                  # Phase-1 location. Will be renamed
│   │                              # to feature_eval/ if any other sport
│   │                              # reuses the harness as-is.
│   ├── feature_builder.py         # Pure-function builders, leakage-gated.
│   ├── model.py                   # Walk-forward harness; sport-agnostic.
│   └── simulator.py               # Flat-stake sim + loss buckets.
├── sources/                       # Production connectors. Live picks read from here.
│   ├── mlb_statcast_lineup.py
│   ├── mlb_pythagorean.py
│   ├── ...
└── backtest/                      # Production backtest (per-sport gate aware).
    └── runner.py                  # Untouched by feature-expansion PRs.
```

Rule of thumb:

* **Production live picks** — modify `sources/<sport>_*.py`. These run
  every day in CI via `flashcat all`.
* **Feature-expansion R&D** — add to `mlb_features/` (or its multi-sport
  rename). These run OFF-CI, on operator demand, via the runner script
  in step "Runner" below.
* **NEVER modify** `src/flashcat/build_site.py::resolve_sport_modes()`
  or `src/flashcat/config.py::live_roi_floor()` as part of a feature-
  expansion PR. The production gate is the trust contract with the
  live site. If you want to relax it, that's a separate PR with its
  own Phil-approval.

## Runner

```bash
PYTHONPATH=src bash scripts/sport_backtest.sh mlb              # last 2y default
PYTHONPATH=src bash scripts/sport_backtest.sh mlb 2022-01-01 2023-12-31
PYTHONPATH=src bash scripts/sport_backtest.sh nba              # Phase 1 stub for now
```

The runner is idempotent (re-running on the same window overwrites
`data/<sport>_walk_forward_backtest.json` and replaces predictions in
`source_history.db` via INSERT OR REPLACE).

## Reference implementation: MLB (Week 1)

* Feature catalog: 17 features (see `FEATURE_NAMES` in
  `src/flashcat/mlb_features/feature_builder.py`).
* Backtest window: 2022-01-01 → 2023-10-01 (538 archive bound).
* Result: 1842 bets, -3.71% ROI, 58.1% win rate, +1.52pp CLV proxy.
* Verdict: model has calibration signal (positive CLV proxy) but the
  signal isn't large enough to overcome the 4.5% market hold. Phase 2
  candidate features: per-batter rolling exit velocity / barrel rate
  (requires Baseball Savant daily pulls, ~8h backfill), Statcast pitcher
  splits vs handedness (cached per `(season, pitcher_id, batter_hand)`),
  and umpire K%-zone tendencies (Retrosheet doesn't expose; needs
  Baseball Savant pitch-by-pitch).

Honest reading: the gate is correctly classifying MLB as RESEARCH mode.
The expanded feature set didn't crack +EV. Next iteration will add
Statcast handedness splits before the rotation moves to NBA.

## Anti-patterns (don't do this)

1. **Cherry-picking the window.** If 2022 was +ROI and 2023 was -ROI, you
   report BOTH and the combined. Do not retroactively narrow the window.
2. **Inflating CLV by changing the proxy.** Pick the closing-line proxy
   ONCE per sport at the start of the cycle and document it. Switching
   proxies mid-PR is hill-climbing on the metric.
3. **Adding features until ROI clears -2% then stopping.** Overfitting on
   the backtest. Walk-forward catches most of it, but if you add 30
   features and 3 of them happen to correlate with 2022-2023 outcomes,
   you're cooking the result. Cap feature additions at ~10 per PR.
4. **Disabling tests.** If a test fails, the feature broke a leakage
   gate or a numerical invariant. Fix the code, not the test.
5. **Touching the production gate.** See "Where to plug new features"
   above. Production gate changes are a separate PR.
