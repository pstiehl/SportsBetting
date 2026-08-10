# Weekly loss post-mortem — cross-sport aggregate

Generated: 2026-08-03T12:02:38.464444Z

## Headline

- **5 sport(s) backtested** atp, mlb, nfl, pga, wta
- **13,499 bets**, **4,953 losing bets** classified
- **Cross-sport top driver**: `pure_variance` (1,351 losing bets — 27.3% of all classified losses)

## Per-sport headline

| Sport | n_bets | ROI | CLV proxy | Losses | Dominant bucket | Window |
|---|---:|---:|---:|---:|---|---|
| atp | 5,356 | -3.94% | +1.77pp | 1,802 | `favorite_upset` | 2022-01-01 → 2024-12-31 |
| mlb | 3,075 | -3.91% | +3.24pp | 1,424 | `pure_variance` | 2022-01-01 → 2023-12-31 |
| nfl | 416 | +1.45% | +7.51pp | 133 | `divisional_misjudged` | 2022-09-01 → 2024-12-31 |
| pga | 0 | — | — | 0 | `—` | 2022-01-01 → 2024-12-31 |
| wta | 4,652 | -3.28% | +1.26pp | 1,594 | `favorite_upset` | 2022-01-01 → 2024-12-31 |

## Per-sport loss buckets

### atp

| Bucket | Count | % of losses |
|---|---:|---:|
| `favorite_upset` | 464 | 25.7% |
| `pure_variance` | 315 | 17.5% |
| `line_moved_against` | 274 | 15.2% |
| `generic` | 245 | 13.6% |
| `ranking_signal_wrong` | 219 | 12.2% |
| `fatigue_disadvantage` | 101 | 5.6% |
| `surface_form_wrong` | 101 | 5.6% |
| `h2h_signal_wrong` | 50 | 2.8% |
| `best_of_5_variance` | 33 | 1.8% |

### mlb

| Bucket | Count | % of losses |
|---|---:|---:|
| `pure_variance` | 688 | 48.3% |
| `pitcher_signal_wrong` | 352 | 24.7% |
| `line_moved_against` | 149 | 10.5% |
| `generic` | 133 | 9.3% |
| `rolling_signal_wrong` | 102 | 7.2% |

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

### wta

| Bucket | Count | % of losses |
|---|---:|---:|
| `favorite_upset` | 332 | 20.8% |
| `pure_variance` | 324 | 20.3% |
| `generic` | 294 | 18.4% |
| `line_moved_against` | 237 | 14.9% |
| `ranking_signal_wrong` | 197 | 12.4% |
| `surface_form_wrong` | 86 | 5.4% |
| `fatigue_disadvantage` | 73 | 4.6% |
| `h2h_signal_wrong` | 51 | 3.2% |

## Cross-sport bucket roll-up

Bucket names with the same string across sports get summed. NBA-specific buckets (`pace_signal_wrong`, `rest_disadvantage`) and MLB-specific buckets (`pitcher_signal_wrong`) are reported alongside the shared ones (`pure_variance`, `line_moved_against`, `rolling_signal_wrong`, `generic`).

| Bucket | Cross-sport count | % of all classified losses |
|---|---:|---:|
| `pure_variance` | 1,351 | 27.3% |
| `favorite_upset` | 796 | 16.1% |
| `generic` | 697 | 14.1% |
| `line_moved_against` | 665 | 13.4% |
| `ranking_signal_wrong` | 416 | 8.4% |
| `pitcher_signal_wrong` | 352 | 7.1% |
| `surface_form_wrong` | 187 | 3.8% |
| `fatigue_disadvantage` | 174 | 3.5% |
| `rolling_signal_wrong` | 118 | 2.4% |
| `h2h_signal_wrong` | 101 | 2.0% |
| `divisional_misjudged` | 42 | 0.8% |
| `best_of_5_variance` | 33 | 0.7% |
| `bye_off_overrated` | 9 | 0.2% |
| `rest_disadvantage` | 6 | 0.1% |
| `prior_disagreement_wrong` | 6 | 0.1% |

## Reading the buckets

- **`pure_variance`** — pick prob in [0.45, 0.55]. Acceptable. Coin-flippy picks lose half the time by definition.
- **`line_moved_against`** / **`market_proxy_dominant`** — model edge vs market proxy < 1pp. We're losing to the vig. Edge gate fixes this in production; feature work doesn't.
- **`pitcher_signal_wrong`** (MLB) / **`pace_signal_wrong`** (NBA) / **`rolling_signal_wrong`** (any) — model overweighted a specific feature class. This IS a feature-quality signal. If one bucket > 20% of losses for a sport, it's a candidate for Phase-2 work.
- **`rest_disadvantage`** (NBA) — we backed the team with worse rest. If this is significant, the rest features need bigger coefficients (or sign flipping).
- **`generic`** — fallback. Should be small. A large generic bucket means our taxonomy is under-fitting losses.
