# Weekly loss post-mortem — cross-sport aggregate

Generated: 2026-07-06T12:03:30.617943Z

## Headline

- **2 sport(s) backtested** mlb, nfl
- **2,258 bets**, **905 losing bets** classified
- **Cross-sport top driver**: `pure_variance` (294 losing bets — 32.5% of all classified losses)

## Per-sport headline

| Sport | n_bets | ROI | CLV proxy | Losses | Dominant bucket | Window |
|---|---:|---:|---:|---:|---|---|
| mlb | 1,842 | -3.71% | +1.52pp | 772 | `pure_variance` | 2022-01-01 → 2023-12-31 |
| nfl | 416 | +1.45% | +7.51pp | 133 | `divisional_misjudged` | 2022-09-01 → 2024-12-31 |

## Per-sport loss buckets

### mlb

| Bucket | Count | % of losses |
|---|---:|---:|
| `pure_variance` | 270 | 35.0% |
| `pitcher_signal_wrong` | 255 | 33.0% |
| `line_moved_against` | 181 | 23.4% |
| `generic` | 45 | 5.8% |
| `rolling_signal_wrong` | 21 | 2.7% |

### nfl

| Bucket | Count | % of losses |
|---|---:|---:|
| `divisional_misjudged` | 42 | 31.6% |
| `generic` | 25 | 18.8% |
| `pure_variance` | 24 | 18.0% |
| `rolling_signal_wrong` | 16 | 12.0% |
| `bye_off_overrated` | 9 | 6.8% |
| `rest_disadvantage` | 6 | 4.5% |
| `prior_disagreement_wrong` | 6 | 4.5% |
| `line_moved_against` | 5 | 3.8% |

## Cross-sport bucket roll-up

Bucket names with the same string across sports get summed. NBA-specific buckets (`pace_signal_wrong`, `rest_disadvantage`) and MLB-specific buckets (`pitcher_signal_wrong`) are reported alongside the shared ones (`pure_variance`, `line_moved_against`, `rolling_signal_wrong`, `generic`).

| Bucket | Cross-sport count | % of all classified losses |
|---|---:|---:|
| `pure_variance` | 294 | 32.5% |
| `pitcher_signal_wrong` | 255 | 28.2% |
| `line_moved_against` | 186 | 20.6% |
| `generic` | 70 | 7.7% |
| `divisional_misjudged` | 42 | 4.6% |
| `rolling_signal_wrong` | 37 | 4.1% |
| `bye_off_overrated` | 9 | 1.0% |
| `rest_disadvantage` | 6 | 0.7% |
| `prior_disagreement_wrong` | 6 | 0.7% |

## Reading the buckets

- **`pure_variance`** — pick prob in [0.45, 0.55]. Acceptable. Coin-flippy picks lose half the time by definition.
- **`line_moved_against`** / **`market_proxy_dominant`** — model edge vs market proxy < 1pp. We're losing to the vig. Edge gate fixes this in production; feature work doesn't.
- **`pitcher_signal_wrong`** (MLB) / **`pace_signal_wrong`** (NBA) / **`rolling_signal_wrong`** (any) — model overweighted a specific feature class. This IS a feature-quality signal. If one bucket > 20% of losses for a sport, it's a candidate for Phase-2 work.
- **`rest_disadvantage`** (NBA) — we backed the team with worse rest. If this is significant, the rest features need bigger coefficients (or sign flipping).
- **`generic`** — fallback. Should be small. A large generic bucket means our taxonomy is under-fitting losses.
