# Pre-registration: project 20 (risk parity and the bond bull market)

Written on 2026-11-22, after the data were downloaded and before any portfolio was backtested or any era
statistic was computed. The only things I had looked at were the raw series (to check dates and gaps). Every
change after this point is listed at the bottom with its reason.

## Question

Levered risk parity beat 60/40 at the same volatility from the 1980s to 2020. Was that a structural property
(investors avoid leverage, so low-risk assets carry higher risk-adjusted returns: Asness, Frazzini & Pedersen 2012),
or was it the 40-year fall in bond yields from 15.8% to 0.5%?

## Data and eras (fixed now)

* Monthly, 1926-07 to 2026-09. US stock market excess return (Ken French), 1-month T-bill (Ken French), a
  constant-maturity 10-year Treasury rebuilt from yields (month-end ^TNX from 1962, a monthly long-term
  government yield before that), and gold from 1972 (World Bank monthly average to 2000-08, COMEX month-end after).
* Eras are cut at the extremes of the 10-year yield, not at anything to do with the strategies:
  E1 1926-07 to the pre-1950 yield trough; E2 that trough to the 1981-09 peak (bond bear);
  E3 1981-10 to the 2020-07 trough (bond bull); E4 2020-08 to 2026-09 (bond bear).

## Strategies (fixed now)

Monthly rebalancing, 36-month trailing covariance (estimated only on data before the month), 5 bp per unit
of turnover, leverage financed at T-bill + 25 bp a year.

* `60/40`: 60% stocks, 40% bonds.
* `RP` (main): inverse-volatility weights, scaled so that the ex-ante volatility equals the ex-ante volatility of
  60/40 under the same covariance estimate.
* `RP-unlevered`: inverse-volatility weights that sum to one.

## Hypotheses

* **H1 (bond bull).** The Sharpe difference RP − 60/40 is positive in E3 and not positive in E2 and E4.
* **H2 (structural).** The difference is positive in every era, bond bears included.
* H1 and H2 are judged on the four era tests (paired stationary bootstrap, mean block 12 months, Holm across four).
* **H3 (windfalls).** Removing the era's average bond-yield capital gain and the era's average equity re-rating
  (change in log price/dividend) cuts RP's E3 advantage by more than half.
* **H4 (hurdle).** The closed-form break-even ratio r*(ρ) of bond-to-stock Sharpe classifies the sign of the
  realised era difference correctly in at least 3 of 4 eras.
* **H5 (means, link to project 05).** No ex-ante rule that uses trailing mean returns (max-Sharpe on trailing means,
  or switching between RP and 60/40 on the trailing hurdle) beats RP after Holm.
* **H6 (gold).** In the 1972–2026 three-asset universe, adding gold through equal risk contribution raises RP's Sharpe
  in the two bond-bear stretches (1972-01 to 1981-09 and E4).

## Changes after registration

1. **Pre-1962 yield source (2026-11-22, after the first backtest run).** The first run used Shiller's monthly long rate
   for 1926–1961. It gave risk parity a Sharpe ratio of 3.6 in E1, which is not credible. Inspection showed that
   Shiller's series is linearly interpolated from annual yields before 1953, so the monthly bond returns built from it
   are artificially smooth. I replaced it with the Federal Reserve's monthly long-term government bond yield (FRED
   LTGOVTBD, monthly averages, 1925–2000). That moves the E1/E2 boundary, which is set by the yield trough, from
   1941-01 to 1941-07. No hypothesis, strategy or test was changed. The Shiller file was removed from the project.
2. **Additions after the first results (2026-11-22), reported as exploratory, not as tests of H1–H6:** the bonds-only and
   stocks-only windfall removals, the full-sample windfall-free gap, a 3× leverage cap, the window ending in 2010 and
   2020–26 without 2022, and the risk-share measure. They were added to answer referee-style objections. None of them
   changes a pre-registered verdict.
