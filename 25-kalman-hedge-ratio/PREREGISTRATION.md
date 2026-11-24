# Pre-registration — Project 25, the Kalman hedge ratio

Written on 23 November 2026, before any Kalman hedge loss or strategy return was computed. The data had
already been used by projects 09 (industry betas) and 11 (pairs trading), so their published results are
known; nothing in this file was chosen after looking at a Kalman result. Changes made after registration are
listed at the end, with the reason.

## Question

What does a Kalman-filter hedge ratio add, out of sample, over the static hedge ratio of project 11 and the
rolling and exponentially weighted hedge ratios of project 09? And how much of what it seems to add is
mechanical?

## Model

Observation: y_t = β_t x_t + ε_t, ε_t ~ N(0, R). State: β_t = β_{t-1} + η_t, η_t ~ N(0, q).
(For pairs, the state is (α_t, β_t) on log prices, with Q = q I, the structure popularised by Chan 2013.)
The signal-to-noise ratio r = q/R is the only parameter that matters for the hedge ratio, because the filter is
invariant to scaling q and R together. It is estimated by concentrated maximum likelihood (prediction-error
decomposition) on a 41-point log grid from 1e-7 to 1e-1, using only data before each re-estimation date
(expanding window, re-estimated every January for hedging; on the formation year for pairs).

A steady-state calculation maps any r into the half-life of an exponentially weighted (EWMA) regression with the
same average memory; this "matched EWMA" uses the same r̂ and is the main comparison.

## Universes (deliberate reuse, cross-linked)

* Hedging A: 49 Fama–French industries against the value-weighted market, daily excess returns,
  forecasts from Aug 1929 to Sep 2026 — the exact panel and loss of project 09.
* Hedging B: 21 single-country iShares ETFs against SPY, daily, evaluation Jan 2002 to Nov 2026.
* Pairs B: the 22 country ETFs of project 11, its windows (252-day formation, 126-day trading, monthly starts),
  its eligibility rules and its placebo market N1 (each asset given the same drift, whole days resampled IID,
  10 replications per window).
* Pairs A: the 49 industries of project 11, same design, as the secondary universe.

## Hedging protocols

* M (month-end): β fixed at the last day of month t and used for every day of month t+1; scored with project
  09's loss L = mean (y − β̂x)² and excess-loss ratio against its textbook 5-year monthly OLS beta.
* D (daily): β_{t|t-1} updated every day, used for day t; same loss, aggregated by industry-month.

Estimators in D: Kalman (ML r̂), matched calendar EWMA, EWMA half-life 126 days (project 09's), rolling 252-day
OLS, expanding OLS (static), and project 09's month-end Welch beta held for the month.

## Pairs rules (selection held fixed)

The same 20 pairs per window (project 11's distance ranking) are traded with three hedge ratios: static
(formation OLS on log prices), rolling 63-day OLS, and Kalman (ML r̂ on formation, state carried into trading).
Signal z = spread / its standard deviation (formation s.d.; rolling s.d.; √F_t for Kalman); open at |z| > 2,
close at the zero crossing or at the end; one-day wait; legs 1 : β at the open; 5 bp per leg for the net figure.

## Hypotheses

* **H1 (equivalence).** In protocol D, the Kalman hedge loss differs from the matched EWMA's by less than 1 %
  of the loss, on both hedging universes.
* **H2 (where they differ).** Whatever Kalman gains over the matched EWMA comes disproportionately from the
  most volatile 10 % of months (more than half of the cumulative difference).
* **H3 (daily updating).** Protocol D beats protocol M for the Kalman beta (lower loss, DM test, Holm).
* **H4 (mechanical stationarity).** The Kalman innovation spread of a selected pair rejects a unit root (ADF,
  5 %) in more than 90 % of pairs on the placebo market, so it is not evidence of a relationship.
* **H5 (mechanical convergence).** More than half of the closing of a Kalman spread on converged trades comes
  from the hedge ratio and intercept moving, not from relative prices converging.
* **H6 (no extra profit).** With selection held fixed, the Kalman rule's gross return on country ETFs is not
  higher than the static rule's, and its edge over the placebo is not larger.

Confirmatory tests: H1 and H3 on both hedging universes (4 DM tests), H6 (1 test on the edge difference).
Holm correction over these five. H2, H4 and H5 are descriptive.

## Inference

Monthly loss or return series; Newey–West t (6 lags); 95 % CIs from a stationary block bootstrap (mean block
12 months). Stress windows: 1929–32, 1987, 2000–02, 2008–09, 2020, 2022 (industries); 2008–09, 2020, 2022
(countries).

## Changes after registration

1. Grid for r widened from [1e-7, 1e-1] to [1e-5, 1e3] (41 points) after a smoke test on one ETF showed the
   likelihood choosing the upper edge. No loss had been computed.
2. Country-ETF stale-price rule tightened before any result: a series starts after the *last* calendar year
   with 10 % or more zero returns (the first version used the first clean year, which let EWK's stale 2001–03
   back in). Evaluation starts in January 2006, after at least two years of estimation data.
3. Exploratory, added after the first hedging loss table: the Kalman filter and the EWMA with their memory
   chosen by past squared hedge error ("past loss") instead of likelihood.
4. Exploratory, added while explaining result 3: simulation worlds (src/mechanism.py). Worlds A–D were written
   first; E–I were added after A–D did not reproduce the short likelihood memory.
5. Exploratory, added after the first pairs results: the Kalman rule with a 5-day wait and a one-day reversal
   benchmark rule.
6. A 2011 euro-crisis stress window was added for country ETFs.
None of these changes alters a registered analysis or enters the five confirmatory tests.
