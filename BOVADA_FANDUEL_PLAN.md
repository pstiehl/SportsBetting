# Bovada + FanDuel Live Sources — Plan

## Goal
Replace The Odds API as the *required* live moneyline source. Add Bovada
(MLB, tennis, NBA, NHL, NFL) and FanDuel (MLB, NBA, NHL, NFL) as direct
public-JSON scrapers. The Odds API stays in the connector list but no
longer being keyless means [] is no longer an empty slate — Bovada and
FanDuel fill the gap.

## Branch
`feat/bovada-fanduel-live-sources` off `main` (currently at origin/main HEAD).

## Endpoints (verified from this environment)

### Bovada
- Nav discovery: `https://www.bovada.lv/services/sports/event/v2/nav/A/description/{sport}`
  - Response shape: `{current, parents, children: [...]}`. Each child has
    `link` (e.g. `/tennis/french-open`), `description`, `numEvents`.
- Coupon (per league): `https://www.bovada.lv/services/sports/event/coupon/events/A/description{link}?marketFilterId=def&preMatchOnly=true&eventsLimit=200&lang=en`
  - Response is a list of groups. Each group has `events[]` and `path[]`.
  - `path[0]` is the leaf league (e.g. "French Open Men's Singles") → use this
    to bucket atp vs wta for tennis.
  - Each event has `description`, `startTime` (epoch ms), `sport` ("BASE",
    "TENN", "BASK", "FOOT", "HOCK"), `competitors[]`, `displayGroups[]`.
  - Moneyline market: `descriptionKey == "Head To Head"` AND
    `description == "Moneyline"`. Outcomes have `price.american` (string,
    e.g. "-139", "+116", "EVEN").
  - Sport tag mapping: BASE→mlb, BASK→nba, FOOT→nfl, HOCK→nhl. Tennis splits
    by `path[0].link` substring `men-s-singles` → atp, `women-s-singles` →
    wta. Doubles/mixed are skipped (our Sport literal doesn't model them).

### FanDuel
- Page endpoint: `https://sbapi.mi.sportsbook.fanduel.com/api/content-managed-page?page=CUSTOM&customPageId={slug}&pbHorizontal=false&_ak=FhMFpcPWXMeyZxOx&timezone=America%2FNew_York`
  - Slugs: `mlb` (works, 31 events), `nfl`, `nba`, `nhl`. Tennis slugs all
    404 or return empty — skip tennis on FanDuel.
  - Response: `attachments.events` (dict, eventId→event with name, openDate,
    competitionId). `attachments.markets` (dict, marketId→market).
  - Moneyline filter: `market.marketType == "MONEY_LINE"` AND
    `market.marketName == "Moneyline"`. (Spec said "MATCH_BETTING" but live
    response uses "MONEY_LINE" — verified.)
  - Each market has `eventId` linking back to the event.
  - Each runner has `winRunnerOdds.americanDisplayOdds.americanOdds` (int).
    (Spec said `winRunnerOdds.americanOdds` but actual path is nested.)
  - Runner.result.type is "HOME" or "AWAY" — use that to assign side.

## Devig math
For two-sided moneyline:
- Convert each american odds to implied prob (vig included)
- Sum implied probs = 1 + vig
- Devigged side prob = side_implied / sum_implied

Verification:
- Marlins -104, Mets -112 →
  - Marlins implied = 104/(104+100) = 0.50980
  - Mets implied = 112/(112+100) = 0.52830
  - Sum = 1.03810
  - Marlins devigged = 0.50980 / 1.03810 = 0.49108
  - Mets devigged = 0.52830 / 1.03810 = 0.50892
  - But spec example says "Marlins -104 / Mets -112 devigs to ~51.0% Marlins".
    Wait — that's the favorite. Marlins are -104 (slight favorite), Mets
    are -112 (heavier favorite). Mets win prob should be higher. The spec
    appears flipped or the Marlins are listed first as away. Looking at
    actual response: Marlins.result.type = "AWAY", Mets.result.type =
    "HOME". So home_win_prob = Mets devigged = 50.9% ≈ 51%. The spec text
    "~51.0% Marlins" is just shorthand for "the favorite side around 51".
    We'll target home_win_prob from the HOME runner.
- Diamondbacks +122, Mariners -144 →
  - Dbacks implied = 100/(122+100) = 0.45045
  - Mariners implied = 144/(144+100) = 0.59016
  - Sum = 1.04062
  - Mariners devigged = 0.59016 / 1.04062 = 0.56713 ≈ 57% ✓

## Sport-tag handling
- Sport Literal is strict: nfl, nba, mlb, nhl, cfb, cbb, atp, wta.
- Bovada tennis: bucket by leaf-league link substring.
- Bovada doubles/mixed events: skip (no Sport bucket).
- Add no new sport tags to types.py (the spec mentioned "tennis" generic,
  but that would touch a strict Literal across the whole codebase — defer).

## Files to create
1. `src/flashcat/sources/bovada.py` — Bovada connector
2. `src/flashcat/sources/fanduel.py` — FanDuel connector
3. `src/flashcat/sources/__init__.py` — export both
4. `src/flashcat/cli.py` — wire into build pipeline
5. `data/source_weights.json` — add bovada + fanduel keys at equal weight
6. `tests/fixtures/bovada_mlb.json` — captured live response
7. `tests/fixtures/bovada_french_open.json` — captured live response
8. `tests/fixtures/fanduel_mlb.json` — captured live response
9. `tests/test_bovada.py` — golden-file parse test + devig sanity
10. `tests/test_fanduel.py` — golden-file parse test + devig sanity

## Politeness
- Single User-Agent header set on each httpx.Client
- Per-sport rate: 1 request for nav discovery + 1 request per leaf league
- 5s timeout, 1 retry on httpx.HTTPError
- No per-event requests

## Shipping
1. Branch + commit per file group
2. `gh pr create --repo pstiehl/SportsBetting --base main --head feat/bovada-fanduel-live-sources`
3. `gh pr merge --squash --delete-branch`
4. `gh run watch --exit-status`
5. `web_fetch https://pstiehl.github.io/SportsBetting/` and verify populated cards
6. DM Phil with results

## Status
- [x] Plan written
- [ ] bovada.py
- [ ] fanduel.py
- [ ] fixtures
- [ ] tests
- [ ] wired in cli.py + __init__
- [ ] all tests green
- [ ] PR merged
- [ ] workflow green
- [ ] site verified populated
- [ ] Phil DMed
