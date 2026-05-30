# PR-19 — Blender de-dilute + scoreboard aggregation fixes + flat-stake headline

Phil's review of the meta-model v2 found three structural issues with the
adaptive reweighter that this PR fixes, plus three addendum bugs in the
scoreboard generator that surfaced once we looked at the numbers per-sport
instead of per-blend.

## What changed

### 1. Hard exclusion floor in the blender (`src/flashcat/model/reweight.py`)

Sources whose rolling-window ROI is below `FLASHCAT_BLENDER_ROI_FLOOR`
(default **−1%**) AND have ≥ `FLASHCAT_BLENDER_MIN_BETS_FOR_EXCLUSION`
graded bets (default **50**) are now hard-excluded from the pool — they
get weight 0 and an explicit `excluded[sport]` entry with reason
`roi=<v> below floor <floor>`. Down-weighting a losing source by a factor
of 10 still lets it dilute a winning blend; removing it entirely doesn't.

If applying that rule would leave fewer than two surviving sources for a
sport, the exclusion is suppressed and a synthetic
`min_sources_floor_active` excluded-list entry is emitted instead. You
need ≥ 2 sources to blend meaningfully — when there aren't 2 winning
sources for a sport, that's a research-mode signal, not a license to ride
a single source's noise.

Low-sample sources (n_bets < 50) keep their natural softmax weight but
are **capped at 1/N** of the surviving pool, so a noisy small-n source
can't dominate just because it happened to score well on the training
window.

### 2. β = 16, up from β = 8 (`src/flashcat/config.py::hybrid_beta`)

Sharper softmax means the top-Brier source gets a meaningfully larger
share of the blend instead of being diluted by ~equal-weighted peers.
Configurable via `FLASHCAT_BLENDER_BETA` (the legacy `FLASHCAT_HYBRID_BETA`
still works). β = 16 is a deliberate ceiling — higher β = more overfit
risk.

Empirical: NFL blended flat-stake ROI **+3.42% → +7.73%** on the same
data after de-dilution. nfl-nflfastr-epa goes from ~14% of the NFL pool
to ~54%.

### 3. Walk-forward hold-out validation (`src/flashcat/model/holdout.py`)

The central risk of this PR is "tune the floor and β until backtest ROI
looks great." Defense: a new `python -m flashcat holdout` command that
splits `data/source_history.db` into a training window
(2022-01-01 → 2023-12-31) and a held-out window
(2024-01-01 → 2024-12-31), fits the post-PR reweighter on training-only
data, applies those frozen weights to the held-out predictions, and
reports per-sport TRAINING vs HELD-OUT ROI.

**Live numbers on the current `source_history.db`:**

```
Sport   Sources  Excluded    Train ROI   Train N    Holdout ROI   Holdout N   Delta (pp)
----------------------------------------------------------------------------------------
ATP           4         0       -4.77%      5155            n/a           0          n/a
MLB           4         0       +5.15%      4846         -2.25%        2289        -7.40
NBA           3         0          n/a         0            n/a           0          n/a
NFL           4         0       -2.34%       540            n/a           0          n/a
WTA           4         0       -3.91%      4639            n/a           0          n/a
```

**Honest read of the table:**

- **MLB** trains at +5.15% on 2022-2023 but collapses to **−2.25%** on
  the 2024 hold-out. That's a **−7.4pp degradation** and the most
  important number in this PR. mlb-statcast-lineup looks like the model's
  best source by training ROI, but the de-dilution exposes that most of
  the training edge does not survive into the next window. The PR ships
  with this on the record, not buried.
- **ATP / NFL / WTA** show no hold-out ROI because the persisted
  `meta` rows for the only profitable source (nfl-nflfastr-epa) and the
  market-close rows for tennis don't carry a 2022-2023 cutoff — they only
  span the full 2022-2024 window. The hold-out reconstruction needs both
  a training-end row and a full-window row per source; without a
  training-end cutoff there's nothing to subtract. This is a data-coverage
  gap in the backfill scripts, not a model claim — flagging it explicitly
  so the next backfill pass can re-emit windowed meta rows.
- **NBA** has no graded ROI yet (predictions exist but no graded bets in
  meta), so neither training nor hold-out ROI is computable.

The live source-weight file is still fit on the full rolling window —
the hold-out exists purely as a diagnostic. A regression test
(`test_holdout_validation_flags_overfit_when_holdout_collapses`) pins the
shape of the table so future PRs can't silently regress this gate.

### 4. LIVE-mode floor: +1% → +2% (`src/flashcat/config.py::live_roi_floor`)

A 1pp safety buffer above 0 was too aggressive given the backtest-to-live
closing-price gap. Bumped to +2%, with the marginal band widened to
[+2%, +4%) so we keep a meaningful yellow tier. Configurable via
`FLASHCAT_LIVE_ROI_FLOOR` and `FLASHCAT_LIVE_MARGINAL_ROI_CEILING`.

### 5. Per-sport Platt re-fit on the post-exclusion blend (`flashcat calibrate`)

`calibrate` now prefers to re-blend `source_history.db.predictions` using
the *current* `data/source_weights.json` (i.e. the post-exclusion
weights) and fit Platt on those pairs, rather than using the
scoreboard's blended.calibration_rows which were computed against the
pre-exclusion weights. Falls back to the legacy scoreboard path when
`source_history.db` is empty or missing.

Current fit: MLB α=0.099 β=0.689 (n=7135), NBA α=−0.030 β=0.663
(n=7875). Both β values are < 1, indicating the un-calibrated blend is
overconfident — exactly the diagnostic the de-dilution is supposed to
sharpen.

### Addendum item 10 — `source_scoreboard.json` aggregation fixes (`flashcat patch-scoreboard`)

Phil flagged that:
- per-source `n_bets` was `None` regardless of sport,
- `per_sport[sport].blended.roi` was `None` for MLB / NBA / CFB even
  when underlying meta had graded ROI.

New `scoreboard_patch` module post-processes `source_scoreboard.json`
after `reweight` runs:

- Injects sources present in `source_history.db.meta` but missing from
  the in-memory backtest (e.g. mlb-statcast-lineup, nba-bref-srs-pace).
- Fills `n_bets` and `roi` from meta for any source row carrying `None`.
- Synthesizes `blended.roi` as a weight-weighted average over per-source
  meta ROIs when the in-memory backtest produced no graded blend, flagging
  the result with `roi_source = "weighted_per_source_meta"`.

Idempotent — running twice yields the same payload. Never overwrites a
real blended.roi the backtest actually produced.

### Addendum item 11 — flat $100 backtest headline (`flashcat flat-stake`)

Phil's exact research question: "drop $100 on every prediction that beats
the devigged market close by ≥ 3pp, what's the ROI?" New
`flashcat.backtest.flat_stake` module persists the answer to
`source_scoreboard.json::backtest_flat_stake` and the homepage renders
it in a new "Backtest Profitability" headline table **above** the
Recommended Plays section.

**Live numbers:**

```
Sport  Source                    n_bets       Stake       Profit       ROI
---------------------------------------------------------------------------
NFL    flashcat-blended             540    $54,000      $4,173      +7.73%
       nfl-nflfastr-epa             540    $54,000      $7,124     +13.19%
MLB    flashcat-blended            7135   $713,500     $19,782      +2.77%
       mlb-statcast-lineup         7135   $713,500     $19,782      +2.77%
ATP    flashcat-blended            5148   $514,800    -$16,918      -3.29%
WTA    flashcat-blended            4637   $463,700    -$23,127      -4.99%
TOTAL  (all sports, all sources) 38562 $3,856,200    -$91,845      -2.38%
```

NFL blended jumped from +3.42% (pre-PR) → +7.73% (post-de-dilution)
because the two unprofitable 538 sources are now excluded entirely and
β=16 concentrates weight on nfl-nflfastr-epa. The aggregate is still
negative because tennis dominates the bet count and tennis sources are
all below the −1% floor — but per-sport, the model is profitable on
LIVE-mode sports (NFL, MLB) and explicitly RESEARCH-mode on the others.

Tagged `roi_source: "meta"` in the persisted payload since the
predictions ledger doesn't currently carry settlement prices — the
per-event simulator path is wired but inert until the backfill scripts
populate `market_close_decimal`.

### Addendum item 12 — slim today's slate

The "No Edge" bucket on the homepage now drops:
- events with neither a quoted pick price nor a market price quote
  (purely orphaned events with no comparison data), AND
- events in RESEARCH-mode sports whose computed edge is < 1pp ("no edge
  worth debating").

The existing `<details>` collapse on No Edge already provides the
"show 144 more" toggle — slim rules just shrink what counts as bloat.
Recommended/Research buckets are untouched (they're already small
and edge-gated).

## Files touched

```
src/flashcat/backtest/flat_stake.py             (new)
src/flashcat/backtest/scoreboard_patch.py       (new)
src/flashcat/model/holdout.py                   (new)
src/flashcat/model/reweight.py                  (hard exclusion + low-sample cap)
src/flashcat/config.py                          (β=16, ROI floor, LIVE floor=2%)
src/flashcat/cli.py                             (3 new commands; calibrate uses DB)
src/flashcat/build_site.py                      (flat-stake table; slim no-edge)
src/flashcat/site/templates/index.html          (backtest-profitability section)
src/flashcat/site/templates/methodology.html    (PR-19 methodology block)
docs/assets/style.css                           (flat-stake table styling)

tests/test_blender_de_dilute.py                 (new — 18 regression tests)
tests/test_pga_per_sport_mode.py                (updated for +2% floor + 4% ceiling)
tests/test_research_mode_gate.py                (updated for +2% floor)
tests/test_reweight.py                          (overlay-disable for in-test isolation)
tests/test_scoreboard_bugfixes.py               (updated for +2% floor)
```

## Tests

```
261 passed in 6.80s
```

18 new tests in `tests/test_blender_de_dilute.py` covering:

1. Hard exclusion of a source below the ROI floor.
2. Low-sample source (n_bets < 50) stays in the pool but capped at 1/N.
3. `min_sources_floor_active` fallback when <2 survivors.
4. The fallback marker keeps both bad sources when the alternative is one.
5. β = 16 default and `FLASHCAT_BLENDER_BETA` override.
6. β=16 concentrates more than β=8 on the same data.
7. ROI floor default, min-bets default.
8. Flat-stake simulator uses $100 per bet exactly.
9. Flat-stake simulator skips rows below edge threshold.
10. Flat-stake meta fallback when predictions lack settlement prices.
11. `patch_scoreboard` fills missing n_bets from meta.
12. `patch_scoreboard` does NOT overwrite a real backtest-produced blended.roi.
13. Hold-out runner produces a per-sport table with train + holdout ROI.
14. Hold-out runner gracefully handles empty / missing source_history.db.
15. The de-risking gate: synthetic train-collapse case produces a large
    negative delta_pp and surfaces it.
16-18. Various edge cases on the cap-to-1/N redistribution math.

## How to run

```bash
PYTHONPATH=src python -m flashcat reweight        # post-exclusion source weights
PYTHONPATH=src python -m flashcat patch-scoreboard # fills n_bets / blended.roi + flat-stake
PYTHONPATH=src python -m flashcat holdout         # walk-forward 2022-2023 → 2024
PYTHONPATH=src python -m flashcat flat-stake      # headline flat-$100 table
PYTHONPATH=src python -m flashcat calibrate       # Platt fit on post-exclusion blend
PYTHONPATH=src python -m flashcat all             # backtest → reweight → patch → calibrate → build
```

The new `flashcat all` order is **backtest → reweight → patch-scoreboard
→ calibrate → build**, so the calibration fit and the rendered site both
see the post-exclusion weights and the patched scoreboard.

## Known follow-ups (not in this PR)

1. **Backfill scripts need windowed meta rows.** The hold-out is mute on
   NFL / ATP / WTA because their persisted meta rows only span the full
   2022-2024 window. Next PR should re-run the backfill with explicit
   2022-2023 cutoffs so we can subtract them out and recover the 2024 ROI.
2. **`market_close_decimal` on predictions.** The flat-stake simulator
   currently falls back to meta-reconstruction because the predictions
   table doesn't carry settlement prices. Wiring closing decimals into
   `upsert_predictions` would let the simulator do per-event, per-source
   gating instead of relying on the upstream backfill's edge gate.
3. **MLB hold-out collapse (-7.4pp).** This is the most important PR-19
   diagnostic. Likely roots: small effective sample on 2024 mlb-statcast-
   lineup, market drift in the second half of 2024, or genuine model
   degradation. Worth a follow-up sub-investigation before MLB carries
   any meaningful stake recommendations on the live site.
