[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gregorionainggolan8-tech/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gregorionainggolan8-tech/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
the case target, as five quantiles per trophy, for the make.fun competition
[gta6-trophy-rarity-forecast-01](https://make.fun/competitions/gta6-trophy-rarity-forecast-01).

## Model

For each trophy, the program fits a line to its logit earn rate against
log-time since release (`logit(rate) ~ a + b * ln(1 + hours)`), with
recency-weighted reads (24 h half-life). When the history is uninformative
(e.g. the page had not refreshed yet, so every rate is identical), the slope
falls back to a per-tier prior: platinum nearly flat, bronze story trophies
rising fast, gold/silver in between. The interval spread grows with the
horizon, slope uncertainty, fit residuals, unseen-refresh shocks, and a
penalty for flat/short histories. See the comments in `main.py`.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in the
input directory and writes one `<id>.out` per case. Deterministic: two runs
produce byte-identical output.
