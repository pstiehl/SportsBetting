# Source Accountability Report

_Generated: 2026-07-13T12:06:14.589165+00:00_

## TL;DR

- **14** (sport, source) pairs audited. **0** keep, **7** keep-with-caveats, **7** noise, **0** drop, **0** insufficient-data.
- **No source clears +1% ROI on $100/event flat.** This is exactly the structural-vig issue Phil flagged in PHIL_PLAN.md. The production +3pp Kelly gate filters most of these picks out; the source audit is intentionally meaner so we catch a source degrading before the gate stops covering for it.

**Method.** For every prediction source the model touches, score it on the full graded ledger we have. Brier + log loss come from the raw `home_prob` × outcome rows in `source_history.db`. Headline ROI, drawdown, and longest losing streak come from a $100-per-event flat-stake hypothetical: pick the higher-probability side, settle at the closing book price on that side. No edge gate, no Kelly, no skip-coin-flips — that's the production rule; this is the source audit. CLV is the source's picked-side probability minus the single-side market implied. Verdict buckets are: **KEEP** (ROI > +1%), **KEEP-WITH-CAVEATS** (−3% ≤ ROI ≤ +1%), **NOISE** (brier in vig territory), **DROP** (brier ≥ 0.25 or ROI ≤ −10%), **INSUFFICIENT-DATA** (< 200 graded events).

## Summary — every source, every sport

| Sport | Source | n | Hit | Brier ↓ | LogLoss ↓ | ROI/$100 | CLV (pp) | Max DD | Longest L-Streak | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| atp | `market-close` | 8025 | +67.89% | 0.2013 | 0.5853 | -2.93% | — | — | — | **KEEP-WITH-CAVEATS** |
| atp | `market-consensus` | 8025 | +67.89% | 0.2013 | 0.5853 | -2.93% | — | — | — | **KEEP-WITH-CAVEATS** |
| atp | `predict.tennis` | 3430 | +67.11% | — | — | -2.36% | — | — | — | **KEEP-WITH-CAVEATS** |
| atp | `tennis-rank-bt` | 8025 | +63.29% | 0.2264 | 0.6450 | -4.18% | — | — | — | **NOISE** |
| nba | `nba-bref-srs-pace` | 3643 | +63.35% | 0.2316 | 0.6655 | — | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `fivethirtyeight-nfl-elo` | 268 | +63.43% | 0.2220 | 0.6324 | -2.58% | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `fivethirtyeight-nfl-qbelo` | 268 | +63.81% | 0.2154 | 0.6153 | -4.29% | — | — | — | **NOISE** |
| nfl | `market-close` | 809 | +67.99% | 0.2090 | 0.6057 | +0.45% | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `market-consensus` | 809 | +67.99% | 0.2090 | 0.6057 | +0.45% | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `nfl-nflfastr-epa` | 809 | +61.56% | 0.2312 | 0.6560 | -4.75% | — | — | — | **NOISE** |
| wta | `market-close` | 7339 | +67.41% | 0.2041 | 0.5918 | -3.60% | — | — | — | **NOISE** |
| wta | `market-consensus` | 7339 | +67.41% | 0.2041 | 0.5918 | -3.60% | — | — | — | **NOISE** |
| wta | `predict.tennis` | 3329 | +67.95% | — | — | -4.93% | — | — | — | **NOISE** |
| wta | `tennis-rank-bt` | 7330 | +62.67% | 0.2281 | 0.6481 | -4.87% | — | — | — | **NOISE** |

## Verdict roll-up

- **KEEP**: 0
- **KEEP-WITH-CAVEATS**: 7
- **NOISE**: 7
- **DROP**: 0
- **INSUFFICIENT-DATA**: 0

## Per-source notes

### atp · `predict.tennis` — KEEP-WITH-CAVEATS
- OBSERVED-EXTERNAL: predict.tennis does not expose a per-event historical API; numbers are SELF-REPORTED.
- Self-reported on predict.tennis/prediction-check/ and predict.tennis/promo/2024-tennis-predictions-analysis-... (retrieved 2026-05-31 via web.archive.org).
- Site fields three predictor types: odds-based (their words: 'rely on the probabilities implied by betting odds'), performance-points/form-based, and a final ensemble. Reported yields use a fixed $1 stake per match. Hit rates use 'pick the side with higher implied prob' on whichever odds source the site reads at scrape time.
- Site's own overall odds-based yield (full 2024 season, both tours): -1.69%.

### wta · `predict.tennis` — NOISE
- OBSERVED-EXTERNAL: predict.tennis does not expose a per-event historical API; numbers are SELF-REPORTED.
- Self-reported on predict.tennis/prediction-check/ and predict.tennis/promo/2024-tennis-predictions-analysis-... (retrieved 2026-05-31 via web.archive.org).
- Site fields three predictor types: odds-based (their words: 'rely on the probabilities implied by betting odds'), performance-points/form-based, and a final ensemble. Reported yields use a fixed $1 stake per match. Hit rates use 'pick the side with higher implied prob' on whichever odds source the site reads at scrape time.
- Site's own overall odds-based yield (full 2024 season, both tours): -1.69%.

---

### Honesty pact

- Numbers are reported as-is. Sources losing money on the $100/event hypothetical get bucketed as **NOISE** or **DROP**, not spun.
- predict.tennis ROI numbers are SELF-REPORTED by predict.tennis on their own Prediction Check page and 2024 season review. The site publishes its own losing yields. We record those verbatim.
- Per-source ROI is hypothetical: $100 flat on every event the source weighed in on, no edge gate. Production picks pass through a +3pp edge gate and 1/4 Kelly — the source audit is intentionally meaner.
- This is the FIRST RUN of a weekly recurring process. See `docs/AGENT_LOOP.md` for the standing operational cadence.

