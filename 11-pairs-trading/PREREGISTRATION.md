# Pre-registration — Project 11: Convergence or coincidence?

Written 2026-11-22, after the data were downloaded and cleaned but **before any strategy return, convergence rate
or cointegration count was computed** on real or placebo data. Changes made after that point are listed at the
bottom, with the reason.

## Question

Pairs trading is usually defended with two pieces of evidence: most trades converge, and the formation-period
statistics (a small distance, a significant cointegration test) look convincing. Both would also appear in a market
where relative prices have **no mean reversion at all**. How much of the convergence, of the statistical evidence
and of the profit of a textbook pairs strategy is genuine relative mean reversion, measured against a placebo
market with the same volatilities, fat tails and contemporaneous correlations but no serial dependence?

## Universes

* **A. 49 Fama–French industry portfolios**, daily value-weighted total returns (excess return + daily RF), Ken
  French Data Library vintage 202609, July 1926 to 30 Sep 2026. Built by project 09 (`09-beta-horse-race`) and
  copied unchanged. No survivorship bias, a century of history, but *not directly tradable*.
* **B. 22 US-listed single-country equity ETFs**: SPY and the iShares MSCI funds EWA, EWC, EWD, EWG, EWH, EWI, EWJ,
  EWK, EWL, EWM, EWN, EWO, EWP, EWQ, EWS, EWU, EWW (from March 1996), EWY, EWT, EWZ (2000) and EZA (2003). Yahoo
  adjusted closes (dividends reinvested) to 20 Nov 2026. Tradable, all priced in the same New York session, so
  there is no time-zone asynchrony between the two legs. Regional funds (EFA, EEM, IEV, EPP, ILF, EWX) are excluded
  because they hold the single-country funds and would create mechanical pairs.

## Strategy (fixed in advance)

* **Formation** 252 trading days, **trading** 126 trading days. A new portfolio starts on the first trading day of
  every month, so 6 portfolios overlap. The strategy's daily return is the average of the active portfolios'
  daily returns on *committed capital* (each selected pair gets $1 whether or not it trades, as in Gatev,
  Goetzmann and Rouwenhorst 2006, "GGR").
* An asset is eligible in a window only if it has a return on every day of formation and trading, and fewer than
  5 % zero-return days in formation (stale prices manufacture mean reversion). Industries must also hold at
  least 5 firms at the month-end before formation starts (in 1927, 24 of the 49 held fewer).
* **Distance method (primary, GGR).** Total-return indices normalised to 1 at the start of formation; pairs ranked
  by the sum of squared differences (SSD); the top 20 are traded. In trading, prices are renormalised to 1 at the
  start; a position opens when the spread is more than 2 formation standard deviations away from zero and closes
  when the spread crosses zero, or at the end of the trading period. Positions can reopen. Legs are $1 long the
  cheap asset and $1 short the rich one, held buy-and-hold for the life of the trade.
* **Cointegration method (secondary).** Engle–Granger: OLS of log price A on log price B with a constant in
  formation, ADF t-statistic with one lag and no constant on the residual, MacKinnon p-value for two variables. The
  20 most negative t-statistics are traded on the formation z-score of the residual, with the same 2/0 open/close
  rule; legs have $2 gross, split 1 : β.
* **Execution lag.** Primary: decide at the close of day t, trade at the close of day t+1 ("one-day wait", GGR's
  remedy for bid-ask bounce). Ablations: wait 0 and wait 5.
* **Costs.** Reported gross; net results charge c per leg per one-way trade (4 leg-trades per round trip). The
  break-even c is reported. For B, the pre-registered realistic cost is 5 bp per leg (half-spread plus impact for
  a small account in liquid country ETFs since 2010); 10 bp and 20 bp are shown as stress.

## Placebo markets (the counterfactuals)

For every formation + trading window (378 days) separately, using only that window's returns:

* **N1, IID-day placebo.** Resample whole days (the full cross-section of returns on one date) with replacement.
  This keeps each asset's volatility and fat tails, the contemporaneous correlation matrix and the window's average
  drift, and destroys *all* serial dependence, including any mean reversion in relative prices. Relative prices are
  random walks by construction.
* **N2, block placebo.** Stationary bootstrap of whole days with mean block length 21. Keeps dependence shorter than
  about a month (non-synchronous trading lead–lag, volatility clustering, short-term reversal inside a block) and
  destroys dependence at longer horizons.
* **Selection placebo.** On real data: all eligible pairs traded with the same rules (the expected value of a
  random pair), and GGR's pairs ranked 101–120.

Ten placebo replications per window and per placebo type, seeded by window index.

## Hypotheses

* **H1 (a historical edge exists).** Universe A, distance top-20, one-day wait, gross: the mean of (real − N1)
  monthly return on committed capital over the full sample is positive with Newey–West t > 2.
* **H2 (it has decayed).** In both universes, the real − N1 edge in 2003–2026 is smaller than in A's 1962–2002 (the
  GGR sample period) and is not significantly positive (NW t < 2) for 2003–2026.
* **H3 (convergence is mostly mechanical).** The share of opened trades that converge before the end of the
  trading period is at least 80 % as high in N1 as on real data, in both universes. Falsified if the N1 rate is
  below 0.8 × the real rate in either universe.
* **H4 (cointegration screens mostly find noise).** In universe B, the share of pairs with Engle–Granger p < 0.05 in
  formation is less than 2 × the N1 share, and the probability that a formation-significant pair is significant
  again in the next 252 days exceeds the same probability for a non-significant pair by no more than it does in N1
  (the "persistence lift" is not larger than the placebo's).
* **H5 (not tradable now).** Universe B, distance top-20, one-day wait, net of 5 bp per leg: mean monthly return in
  2010–2026 is not significantly positive (NW t < 2).
* **H6 (descriptive).** The real − N1 edge is larger in stress windows (1929-10 to 1932-06, 1987-10 to 1987-12,
  2000-03 to 2002-10, 2008-09 to 2009-03, 2020-02-20 to 2020-04-30, 2022-01 to 2022-10) than outside them.

Primary confirmatory tests: H1, the 2003–2026 part of H2 (two universes) and H5. Holm correction across these four
tests at family-wise α = 0.05. Everything else is reported with confidence intervals and labelled exploratory.

Inference: monthly series (calendar-month sums of daily returns), Newey–West with 6 lags (the overlap of the six
portfolios), and a stationary block bootstrap (mean block 12 months, 5,000 draws) for confidence intervals.

## Changes after registration

* 2026-11-22, before any result: added the 5-firm rule for industries after the data-quality table showed
  one-to-three-firm portfolios in the 1920s–30s with daily moves of +100 % or more.
* 2026-11-22, after the first run: the stale-price rule leaves fewer than 10 eligible country ETFs in every
  window before 2002 (8–24 % zero-return days per fund before 2005). A portfolio is now formed only when at
  least 10 assets are eligible, so universe B effectively starts in January 2002; the 1997–2002 period is
  dropped, and a no-filter run is reported as an ablation.
* 2026-11-22, after the first run (**a flaw in the registered placebo**): N1 and N2 resample days from the whole
  378-day window, so each asset keeps its realised drift *including the trading period's*. The placebo then
  "knows" which asset will keep outperforming, and any contrarian rule loses in it by construction (about
  −1.3 to −2 % a year). That makes real − placebo an inflated edge. Fix: before resampling, every asset's
  returns are shifted to the same drift (the window's cross-sectional mean), so relative prices are driftless
  random walks. The drift-keeping version is kept as an ablation ("ND", the perfect-momentum world). The
  hypotheses are unchanged; they are tested against the corrected placebo, and the first-run numbers are
  reported in the README's self-review section.
* 2026-11-22, after the first run: the cointegration ranking now skips pairs with a negative hedge ratio β
  (a long/short trade on them is not a hedge). The rule was missing from the code; the positive-control unit
  test exposed it.
* 2026-11-22, after the second run (**a second placebo flaw**): with the drift fix in place, the N2 block
  placebo still earned +1.6 % a year for the distance rule. A synthetic check (IID returns, no dynamics at all)
  shows the same +0.9 % per six-month window, so it is an artefact: after the drift shift each asset's relative
  path over the window sums to zero, and stitching contiguous blocks of that closed loop manufactures mean
  reversion. N2 is dropped from every comparison (the check is a unit test). Its question, how much of the
  profit comes from short-horizon reversal, is answered instead with variance ratios of the traded spreads at
  1 to 63 days, read against the same statistic in N1 (which removes the estimator's finite-sample bias).
