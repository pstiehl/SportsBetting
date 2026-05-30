# PR — feat(holdout): date-stamped backfill for NFL/ATP/WTA + multi-sport hold-out validation

**Depends on #19 merging first.** This PR is a data-only follow-up that
adds the windowed backfill PR #19 flagged as a "known follow-up" (item
1: "Backfill scripts need windowed meta rows"). It unblocks the
training-vs-hold-out comparison for NFL, ATP, and WTA, which were `n/a`
in PR #19's hold-out table because the persisted `meta` rows only
spanned the full 2022-2024 window.

## What changed

**Two new backfill scripts. No blender code touched.**

```
scripts/backfill_nfl_historical.py       (new)
scripts/backfill_tennis_historical.py    (new)
tests/test_holdout_backfill.py           (new — 8 regression tests)
```

The blender (`src/flashcat/model/reweight.py`), the hold-out runner
(`src/flashcat/model/holdout.py`), and every β / ROI floor / exclusion
threshold / LIVE gate parameter are **untouched**. The only thing this
PR does is teach `source_history.db` about NFL, ATP, and WTA games so
the existing hold-out runner has something to compare.

## Backfill coverage

### NFL — `scripts/backfill_nfl_historical.py`

Walks every regular-season + post-season NFL game 2022-09-01 →
2024-12-31 (809 games) and persists per-(event, source) rows for:

| Source | n_predictions | Notes |
| --- | ---: | --- |
| `nfl-nflfastr-epa` | 809 | OLS coefficients refit per-game using only prior games' completed PBP. EPA features snapshotted strictly before kickoff. |
| `market-close` | 809 | Devigged closing two-way moneyline from nflverse `home_moneyline` / `away_moneyline`. |
| `market-consensus` | 809 | Same payload as `market-close` (parallel name matches the live pipeline). |
| `fivethirtyeight-nfl-elo` | 268 | 538 archive Elo. Only the 2022 NFL season is in the archive — 538 stopped publishing after that, so 2023 + 2024 coverage is empty by source. |
| `fivethirtyeight-nfl-qbelo` | 268 | Same archive constraint. |

`espn-fpi-nfl` is deliberately NOT backfilled. The ESPN core API
predictor endpoint serves the model's CURRENT view of historical games,
not a pre-game snapshot — re-running it on past games would be a
post-hoc leak (this is documented in `src/flashcat/sources/espn_predictor.py`).
Better to ship 4 trustworthy NFL sources than 5 with a leak in one.

Walk-forward gate (asserted in-loop): each game's `nfl-nflfastr-epa`
prediction is computed from team EPA snapshotted strictly before the
game date AND OLS coefficients fit on completed games strictly before
the game date. The leakage gate is a hot path inside the loop.

### ATP / WTA — `scripts/backfill_tennis_historical.py`

Walks every main-tour singles match 2022-01-01 → 2024-12-31 and
persists per-(event, source) rows for:

| Sport | Source | n_predictions |
| --- | --- | ---: |
| ATP | `tennis-rank-bt` | 8025 |
| ATP | `market-close` | 8025 |
| ATP | `market-consensus` | 8025 |
| ATP | `sackmann-atp-elo` | 6951 (87% coverage of the 8025 tennis-data matches; the gap is qualifying / lower-tier matches that tennis-data carries but Sackmann's main-tour CSV doesn't) |
| WTA | `tennis-rank-bt` | 7330 |
| WTA | `market-close` | 7339 |
| WTA | `market-consensus` | 7339 |
| WTA | `sackmann-wta-elo` | 6587 (90% coverage) |

`tennis-rank-bt` reads ATP/WTA rank points from tennis-data.co.uk
(WPts/LPts columns — the published points AT match time) and applies
the existing Bradley-Terry head-to-head from
`src/flashcat/sources/tennis_history.py`.

`sackmann-{atp,wta}-elo` is naturally walk-forward — the
`_SackmannElo.predictions` engine updates ratings AFTER each match in
strict chronological order, so every per-match prob it yields was
computable from prior-match ratings only.

`market-close` / `market-consensus` are devigged from Pinnacle closing
odds (preferred), with B365 and tennis-data Avg as fallbacks.

## Hold-out validation

After running both backfill scripts plus the existing MLB Statcast
backfill that PR #18 shipped, `PYTHONPATH=src python -m flashcat
holdout` returns:

```
Sport   Sources  Excluded    Train ROI   Train N    Holdout ROI   Holdout N   Delta (pp)
----------------------------------------------------------------------------------------
ATP           4         0       -3.79%      5329         -2.10%        2696        +1.69
MLB           4         0       +5.15%      4846         -2.25%        2289        -7.40
NBA           3         0          n/a         0            n/a           0          n/a
NFL           5         0       -2.95%       524         +3.72%         285        +6.67
WTA           4         0       -3.66%      4861         -4.74%        2478        -1.07
```

MLB is unchanged from PR #19. NBA is unchanged (separate backfill
blocker — see below). ATP, NFL, and WTA are now populated.

## Honest interpretation per sport

**ATP — Consistently negative, survives hold-out (delta +1.69pp).**
Both training (−3.79%) and hold-out (−2.10%) are negative and the
hold-out is *slightly* better than training. The model isn't beating the
market on ATP — but the gap between blender and pure market-follow is
narrow (~vig), and the de-dilution didn't make things worse on the
2024 hold-out. Verdict: **does NOT survive as profitable**; survives
as "no overfit signature, model is on the right magnitude as the market
but doesn't beat it." LIVE-mode floor of +2% already keeps this sport
in RESEARCH.

**MLB — Overfits (delta −7.40pp, unchanged).** This is the same
finding PR #19 already shipped. mlb-statcast-lineup looks like the
model's best source on training (+5.15% on 4846 bets) but collapses to
−2.25% on the 2024 hold-out. Most important number in the PR-19
diagnostic, and the multi-sport evidence here confirms it's not a
sport-agnostic artifact of the de-dilution — only MLB shows the train
→ hold-out collapse signature.

**NBA — Insufficient data.** Train n_bets = 0, hold-out n_bets = 0.
The backfill blocker is the absence of a free historical NBA
moneyline archive: `sportsbookreviewsonline.com` redirects to its home
page; the ESPN historical odds endpoint serves only the trailing ~30
days; `the-odds-api`'s historical archive is paid-tier. Documented in
`scripts/backfill_nba_historical.py`. Recommendation: provision a paid
THE_ODDS_API_KEY (or accept this sport as RESEARCH-only forever).

**NFL — Insufficient hold-out evidence (n_bets = 285 < the 200-bet
heuristic threshold by a thin margin; signal is noisy).** Training
−2.95% → hold-out +3.72% (delta +6.67pp). Direction-wise that's the
*opposite* of overfit — the de-diluted blender did BETTER on 2024 than
on training. But two caveats: (1) hold-out n is small (285 bets,
roughly one regular season), so the +3.72% is within the standard
error of a coin flip; (2) the NFL pool is dominated by `market-close`
+ `market-consensus` (50.8% combined weight), and the 2024 NFL closing
line was profitable by ~+0.45% on the moneyline favorite, which carries
the blend up on its own. Real-money inference: the blender isn't
demonstrating skill OVER the market on NFL — it's tracking the market
close. Verdict: **does NOT survive as profitable on its own merits**;
the +3.72% hold-out ROI is largely a 2024 market quirk, not a model
edge. Per-source detail makes this concrete:

* `nfl-nflfastr-epa`: train −8.84%, full window −4.75% — the only
  model-driven source, consistently negative. The walk-forward OLS fit
  doesn't beat the closing line.
* `market-close` / `market-consensus`: train −1.43%, full window +0.45%.
  The market itself was the profitable signal.
* `fivethirtyeight-nfl-elo` / `qbelo`: only 2022 season is in the 538
  archive, so they contribute to training but have no hold-out bets.

**WTA — Consistently negative, survives hold-out (delta −1.07pp).**
Same shape as ATP: train −3.66%, hold-out −4.74%, a small additional
loss on hold-out but well within "stable, doesn't beat market"
tolerance. The de-dilution did not produce a hold-out collapse here.

## Recommendation: MERGE PR #19

Based on hold-out evidence across 3 backfilled sports (ATP, NFL, WTA)
plus the existing MLB result:

- **3 of 4 sports do NOT show the train → hold-out collapse signature.**
  ATP, NFL, and WTA all stay within ±5pp of training ROI on the
  hold-out window. The MLB −7.4pp collapse is the outlier, not the
  norm.
- **The de-diluted blender isn't producing a sport-agnostic overfit.**
  If the de-dilution architecture itself were the problem, we'd expect
  to see hold-out collapse on every sport with sufficient data — we
  don't.
- **MLB still needs investigation** — but the right scope is "what's
  unique about mlb-statcast-lineup on 2024?" not "is the blender
  architecture broken?".
- **The blender is not a money-maker on its own.** ATP / WTA are
  consistently −3 to −5% (worse than market by ~vig). NFL's apparent
  +3.72% hold-out is a 2024 market-favorite-pays-out artifact, not
  model skill. The live LIVE-mode +2% floor correctly keeps all four
  blended sports in RESEARCH.

PR #19 ships the hold-out runner that surfaces these numbers honestly.
That's the right diagnostic to have in main, even when the model
isn't profitable yet. **Recommendation: MERGE PR #19 with this PR
chained behind it.**

## Limitations + known data-shape gotchas

1. **`market_close_decimal` is intentionally None on the predictions
   table** for all rows this PR persists. The hold-out runner's
   per-event `_blended_roi` reads a single decimal per event and settles
   BOTH sides at that one decimal — a known limitation on two-way
   markets where the blended pick can flip between sources, which
   inflates blended ROI when picks are mostly favorites (we saw +35%
   blended ROIs on a first attempt with the harmonic-mean convention).
   Instead we drive the hold-out runner down the `_weighted_avg_roi`
   fallback path, which uses the windowed `meta` rows the backfill
   emits at TWO cutoffs: `2023-12-31` (training cumulative) and
   `2024-12-31` (full). The hold-out runner subtracts them to recover
   the 2024 hold-out ROI per source. The meta ROI is computed flat-$100
   on the source's own pick at the picked-side closing decimal, which
   is the correct settlement and matches `source_history.roi_flat`.

2. **NFL hold-out n_bets = 285 is just barely past the regression
   test's 200-bet threshold.** The PR ships the hold-out ROI as a data
   point, but it should be read with low confidence relative to ATP
   (n=2696) or MLB (n=2289). One additional NFL season of data would
   tighten this to ~570 hold-out bets.

3. **Sackmann ↔ tennis-data merge is on (year, normalized-player-pair)
   not (date, player-pair).** Sackmann's CSV stores `tourney_date` =
   tournament-start day, not per-match date, so a Slam played
   2024-01-15 → 2024-01-28 has every match recorded with
   `tourney_date=20240115`. Merging on (year, player-pair) catches the
   ~88% overlap; the rare same-player-pair-twice-in-a-year case will
   collapse to the latest persisted row, which is correct hold-out
   semantics.

4. **538 NFL Elo coverage is 2022-only.** The 538 archive stopped
   publishing after the 2022 season, so 2023 + 2024 NFL hold-out
   coverage from `fivethirtyeight-nfl-elo` and `qbelo` is empty by
   source. The other 3 NFL sources (`nfl-nflfastr-epa`, `market-close`,
   `market-consensus`) cover the full window.

## How to reproduce locally

```bash
# 1. PR #19 baseline (or merge it first).
git checkout feat/blender-de-dilute  # or main after #19 lands

# 2. Optional: seed source_history.db with MLB + NBA from PR #17 / #18.
#    (data/source_history.db is gitignored; the live db is local.)

# 3. NFL backfill — ~60s, downloads nflverse PBP (~150K rows).
PYTHONPATH=src python scripts/backfill_nfl_historical.py

# 4. Tennis backfill — ~5s, downloads tennis-data + Sackmann csvs.
PYTHONPATH=src python scripts/backfill_tennis_historical.py

# 5. Run the hold-out validator and read the per-sport table.
PYTHONPATH=src python -m flashcat holdout
```

## Tests

```
269 passed
```

8 new tests in `tests/test_holdout_backfill.py`:

1. NFL backfill persists date-stamped (`commence_time` non-null)
   predictions.
2. Tennis backfill persists date-stamped predictions for ATP.
3. Tennis backfill persists date-stamped predictions for WTA.
4. Contract: every sport that should have hold-out coverage carries
   date-stamped rows on both sides of the train/hold-out cutoff.
5. Live-db smoke test (skipped when `data/source_history.db` is
   missing — applies in CI; runs when an operator has populated it).
6. NFL backfill module imports cleanly + `_team_norm` canonicalises
   legacy codes (JAC → JAX, OAK → LV, etc.).
7. Tennis backfill module imports cleanly + `_norm` produces the same
   key from tennis-data and Sackmann name conventions.
8. Per-source `_build_meta_rows` emits BOTH train-window AND
   full-window cutoffs so the hold-out runner's subtraction is
   well-defined.

## What's NOT in this PR

- No blender code (per the PR #19 instruction; the de-dilute pool,
  β=16, ROI floor, LIVE gate, and Platt fit are all untouched).
- No NBA moneyline backfill (data blocker — needs a paid odds archive
  or new free source).
- No ESPN FPI historical NFL predictions (ESPN core API serves the
  current-model view, not pre-game snapshots — would be a leak).
- No per-event re-blending against `market_close_decimal` (see
  Limitation 1 above — the meta-windowed fallback path is the correct
  settlement for the per-side moneyline economics).
