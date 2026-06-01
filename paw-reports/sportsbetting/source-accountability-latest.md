# Source Accountability Report

_Generated: 2026-06-01T12:55:11.831408+00:00_

## TL;DR

- **17** (sport, source) pairs audited. **1** keep, **9** keep-with-caveats, **5** noise, **2** drop, **0** insufficient-data.
- **KEEP (ROI > +1% on $100/event flat)**: mlb/mlb-statcast-lineup.
- **DROP (worse than vig or worse than a coin flip)**: atp/coin-flip, wta/coin-flip.

**Method.** For every prediction source the model touches, score it on the full graded ledger we have. Brier + log loss come from the raw `home_prob` × outcome rows in `source_history.db`. Headline ROI, drawdown, and longest losing streak come from a $100-per-event flat-stake hypothetical: pick the higher-probability side, settle at the closing book price on that side. No edge gate, no Kelly, no skip-coin-flips — that's the production rule; this is the source audit. CLV is the source's picked-side probability minus the single-side market implied. Verdict buckets are: **KEEP** (ROI > +1%), **KEEP-WITH-CAVEATS** (−3% ≤ ROI ≤ +1%), **NOISE** (brier in vig territory), **DROP** (brier ≥ 0.25 or ROI ≤ −10%), **INSUFFICIENT-DATA** (< 200 graded events).

## Summary — every source, every sport

| Sport | Source | n | Hit | Brier ↓ | LogLoss ↓ | ROI/$100 | CLV (pp) | Max DD | Longest L-Streak | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| atp | `coin-flip` | 7759 | +50.02% | 0.2500 | 0.6931 | -4.60% | -0.95 | $36,520 | 13 | **DROP** |
| atp | `market-close` | 7759 | +68.22% | 0.1998 | 0.5818 | -2.49% | -1.76 | $21,267 | 7 | **KEEP-WITH-CAVEATS** |
| atp | `predict.tennis` | 3430 | +67.11% | — | — | -2.36% | — | — | — | **KEEP-WITH-CAVEATS** |
| atp | `tennis-rank-bt` | 7747 | +63.55% | 0.2255 | 0.6430 | -3.90% | -5.88 | $31,871 | 12 | **NOISE** |
| mlb | `mlb-statcast-lineup` | 7135 | +53.83% | 0.2483 | 0.6898 | +2.77% | — | — | — | **KEEP** |
| nba | `fivethirtyeight-nba-elo-modern` | 2116 | +63.99% | 0.2271 | 0.6485 | — | — | — | — | **KEEP-WITH-CAVEATS** |
| nba | `fivethirtyeight-nba-raptor` | 2116 | +65.55% | 0.2183 | 0.6286 | — | — | — | — | **KEEP-WITH-CAVEATS** |
| nba | `nba-bref-srs-pace` | 3643 | +63.35% | 0.2316 | 0.6655 | — | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `fivethirtyeight-nfl-elo` | 268 | +63.43% | 0.2220 | 0.6324 | -2.58% | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `fivethirtyeight-nfl-qbelo` | 268 | +63.81% | 0.2154 | 0.6153 | -4.29% | — | — | — | **NOISE** |
| nfl | `market-close` | 809 | +67.99% | 0.2090 | 0.6057 | +0.45% | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `market-consensus` | 809 | +67.99% | 0.2090 | 0.6057 | +0.45% | — | — | — | **KEEP-WITH-CAVEATS** |
| nfl | `nfl-nflfastr-epa` | 809 | +61.56% | 0.2312 | 0.6560 | -4.75% | — | — | — | **NOISE** |
| wta | `coin-flip` | 7025 | +50.46% | 0.2500 | 0.6931 | -2.93% | -1.09 | $28,159 | 13 | **DROP** |
| wta | `market-close` | 7025 | +68.01% | 0.2011 | 0.5847 | -2.86% | -1.92 | $21,599 | 10 | **KEEP-WITH-CAVEATS** |
| wta | `predict.tennis` | 3329 | +67.95% | — | — | -4.93% | — | — | — | **NOISE** |
| wta | `tennis-rank-bt` | 7005 | +63.14% | 0.2268 | 0.6453 | -4.52% | -6.43 | $32,956 | 12 | **NOISE** |

## Verdict roll-up

- **KEEP**: 1
- **KEEP-WITH-CAVEATS**: 9
- **NOISE**: 5
- **DROP**: 2
- **INSUFFICIENT-DATA**: 0

## Per-source notes

### atp · `coin-flip` — DROP
- per-event from tennis-data.co.uk re-pull
- window: 2022-01-03 → 2024-11-17

### atp · `market-close` — KEEP-WITH-CAVEATS
- per-event from tennis-data.co.uk re-pull
- window: 2022-01-03 → 2024-11-17

### atp · `predict.tennis` — KEEP-WITH-CAVEATS
- OBSERVED-EXTERNAL: predict.tennis does not expose a per-event historical API; numbers are SELF-REPORTED.
- Self-reported on predict.tennis/prediction-check/ and predict.tennis/promo/2024-tennis-predictions-analysis-... (retrieved 2026-05-31 via web.archive.org).
- Site fields three predictor types: odds-based (their words: 'rely on the probabilities implied by betting odds'), performance-points/form-based, and a final ensemble. Reported yields use a fixed $1 stake per match. Hit rates use 'pick the side with higher implied prob' on whichever odds source the site reads at scrape time.
- Site's own overall odds-based yield (full 2024 season, both tours): -1.69%.

### atp · `tennis-rank-bt` — NOISE
- per-event from tennis-data.co.uk re-pull
- window: 2022-01-03 → 2024-11-17

### wta · `coin-flip` — DROP
- per-event from tennis-data.co.uk re-pull
- window: 2022-01-03 → 2024-11-09

### wta · `market-close` — KEEP-WITH-CAVEATS
- per-event from tennis-data.co.uk re-pull
- window: 2022-01-03 → 2024-11-09

### wta · `predict.tennis` — NOISE
- OBSERVED-EXTERNAL: predict.tennis does not expose a per-event historical API; numbers are SELF-REPORTED.
- Self-reported on predict.tennis/prediction-check/ and predict.tennis/promo/2024-tennis-predictions-analysis-... (retrieved 2026-05-31 via web.archive.org).
- Site fields three predictor types: odds-based (their words: 'rely on the probabilities implied by betting odds'), performance-points/form-based, and a final ensemble. Reported yields use a fixed $1 stake per match. Hit rates use 'pick the side with higher implied prob' on whichever odds source the site reads at scrape time.
- Site's own overall odds-based yield (full 2024 season, both tours): -1.69%.

### wta · `tennis-rank-bt` — NOISE
- per-event from tennis-data.co.uk re-pull
- window: 2022-01-03 → 2024-11-09

---

### Honesty pact

- Numbers are reported as-is. Sources losing money on the $100/event hypothetical get bucketed as **NOISE** or **DROP**, not spun.
- predict.tennis ROI numbers are SELF-REPORTED by predict.tennis on their own Prediction Check page and 2024 season review. The site publishes its own losing yields. We record those verbatim.
- Per-source ROI is hypothetical: $100 flat on every event the source weighed in on, no edge gate. Production picks pass through a +3pp edge gate and 1/4 Kelly — the source audit is intentionally meaner.
- This is the FIRST RUN of a weekly recurring process. See `docs/AGENT_LOOP.md` for the standing operational cadence.

