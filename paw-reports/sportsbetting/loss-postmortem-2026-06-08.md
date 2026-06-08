# Weekly loss post-mortem — cross-sport aggregate

Generated: 2026-06-08T16:01:42.841317Z

## Headline

- **2 sport(s) backtested** mlb, nba
- **4,421 bets**, **1,687 losing bets** classified
- **Cross-sport top driver**: `line_moved_against` (574 losing bets — 34.0% of all classified losses)

## Per-sport headline

| Sport | n_bets | ROI | CLV proxy | Losses | Dominant bucket | Window |
|---|---:|---:|---:|---:|---|---|
| mlb | 1,842 | -3.71% | +1.52pp | 772 | `pure_variance` | 2022-01-01 → 2023-12-31 |
| nba | 2,579 | -6.75% | -0.61pp | 915 | `line_moved_against` | 2022-01-01 → 2024-04-14 |

## Per-sport loss buckets

### mlb

| Bucket | Count | % of losses |
|---|---:|---:|
| `pure_variance` | 270 | 35.0% |
| `pitcher_signal_wrong` | 255 | 33.0% |
| `line_moved_against` | 181 | 23.4% |
| `generic` | 45 | 5.8% |
| `rolling_signal_wrong` | 21 | 2.7% |

### nba

| Bucket | Count | % of losses |
|---|---:|---:|
| `line_moved_against` | 393 | 43.0% |
| `pure_variance` | 206 | 22.5% |
| `pace_signal_wrong` | 121 | 13.2% |
| `generic` | 92 | 10.1% |
| `rolling_signal_wrong` | 69 | 7.5% |
| `rest_disadvantage` | 34 | 3.7% |

## Cross-sport bucket roll-up

Bucket names with the same string across sports get summed. NBA-specific buckets (`pace_signal_wrong`, `rest_disadvantage`) and MLB-specific buckets (`pitcher_signal_wrong`) are reported alongside the shared ones (`pure_variance`, `line_moved_against`, `rolling_signal_wrong`, `generic`).

| Bucket | Cross-sport count | % of all classified losses |
|---|---:|---:|
| `line_moved_against` | 574 | 34.0% |
| `pure_variance` | 476 | 28.2% |
| `pitcher_signal_wrong` | 255 | 15.1% |
| `generic` | 137 | 8.1% |
| `pace_signal_wrong` | 121 | 7.2% |
| `rolling_signal_wrong` | 90 | 5.3% |
| `rest_disadvantage` | 34 | 2.0% |

## Reading the buckets

- **`pure_variance`** — pick prob in [0.45, 0.55]. Acceptable. Coin-flippy picks lose half the time by definition.
- **`line_moved_against`** / **`market_proxy_dominant`** — model edge vs market proxy < 1pp. We're losing to the vig. Edge gate fixes this in production; feature work doesn't.
- **`pitcher_signal_wrong`** (MLB) / **`pace_signal_wrong`** (NBA) / **`rolling_signal_wrong`** (any) — model overweighted a specific feature class. This IS a feature-quality signal. If one bucket > 20% of losses for a sport, it's a candidate for Phase-2 work.
- **`rest_disadvantage`** (NBA) — we backed the team with worse rest. If this is significant, the rest features need bigger coefficients (or sign flipping).
- **`generic`** — fallback. Should be small. A large generic bucket means our taxonomy is under-fitting losses.
