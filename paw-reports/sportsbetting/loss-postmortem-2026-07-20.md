# Weekly loss post-mortem — cross-sport aggregate

Generated: 2026-07-20T12:02:13.193549Z

## Headline

- **4 sport(s) backtested** atp, mlb, nfl, wta
- **12,266 bets**, **4,301 losing bets** classified
- **Cross-sport top driver**: `pure_variance` (933 losing bets — 21.7% of all classified losses)

## Per-sport headline

| Sport | n_bets | ROI | CLV proxy | Losses | Dominant bucket | Window |
|---|---:|---:|---:|---:|---|---|
| atp | 5,356 | -3.94% | +1.77pp | 1,802 | `favorite_upset` | 2022-01-01 → 2024-12-31 |
| mlb | 1,842 | -3.71% | +1.52pp | 772 | `pure_variance` | 2022-01-01 → 2023-12-31 |
| nfl | 416 | +1.45% | +7.51pp | 133 | `divisional_misjudged` | 2022-09-01 → 2024-12-31 |
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
| `pure_variance` | 933 | 21.7% |
| `favorite_upset` | 796 | 18.5% |
| `line_moved_against` | 697 | 16.2% |
| `generic` | 609 | 14.2% |
| `ranking_signal_wrong` | 416 | 9.7% |
| `pitcher_signal_wrong` | 255 | 5.9% |
| `surface_form_wrong` | 187 | 4.3% |
| `fatigue_disadvantage` | 174 | 4.0% |
| `h2h_signal_wrong` | 101 | 2.3% |
| `divisional_misjudged` | 42 | 1.0% |
| `rolling_signal_wrong` | 37 | 0.9% |
| `best_of_5_variance` | 33 | 0.8% |
| `bye_off_overrated` | 9 | 0.2% |
| `rest_disadvantage` | 6 | 0.1% |
| `prior_disagreement_wrong` | 6 | 0.1% |

## Reading the buckets

- **`pure_variance`** — pick prob in [0.45, 0.55]. Acceptable. Coin-flippy picks lose half the time by definition.
- **`line_moved_against`** / **`market_proxy_dominant`** — model edge vs market proxy < 1pp. We're losing to the vig. Edge gate fixes this in production; feature work doesn't.
- **`pitcher_signal_wrong`** (MLB) / **`pace_signal_wrong`** (NBA) / **`rolling_signal_wrong`** (any) — model overweighted a specific feature class. This IS a feature-quality signal. If one bucket > 20% of losses for a sport, it's a candidate for Phase-2 work.
- **`rest_disadvantage`** (NBA) — we backed the team with worse rest. If this is significant, the rest features need bigger coefficients (or sign flipping).
- **`generic`** — fallback. Should be small. A large generic bucket means our taxonomy is under-fitting losses.
