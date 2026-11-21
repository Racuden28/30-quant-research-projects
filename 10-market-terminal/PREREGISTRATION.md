# Pre-registration — Project 10: What on a market screen is worth looking at?

Written 2026-11-21, before any out-of-sample statistic was computed. Changes made after that point are listed at the bottom.

## Question

A retail market terminal shows the same dozen numbers for every ticker: the day's change, the week and the month,
RSI, the 50/200-day moving averages, the distance from the 52-week high, Bollinger %b, realised volatility, the
drawdown. Which of these numbers carry **out-of-sample information about the next month**, and information about *what*:
the return, the volatility or the size of the drawdown?

## Hypotheses (directional, fixed in advance)

* **H1 (returns).** No widget beats the prevailing historical mean as a forecast of the next 21-day log return,
  after Benjamini–Hochberg control at q = 0.10 across all widget × asset tests. *Falsified* if at least one
  pooled return test survives FDR with R²_OS > 0.
* **H2 (risk).** Volatility-type widgets (RV21, vol ratio, |z-score of today|) beat the prevailing mean of log
  future volatility with R²_OS > 20 % pooled. Expected, and included as a positive control.
* **H3 (the central claim).** "Trend" widgets that are read as return signals — distance from the 52-week high,
  gap to the 200-day average, drawdown, 12-1 momentum — carry significant information about future **volatility and
  drawdown** that is *incremental to RV21*, while carrying none about the return. *Falsified* if no trend widget
  has a significant (FDR q = 0.10) positive incremental R²_OS over an RV21-only volatility model.
* **H4 (redundancy).** The 14 widgets span at most four independent pieces of information (eigenvalues of the pooled
  rank-correlation matrix above 1, Kaiser rule).

## Data

* Daily total-return log returns of 7 ETFs (SPY, EEM, TLT, HYG, GLD, USO, URA), April 2006 to 20 Nov 2026, the
  same file as Project 01 (Yahoo adjusted closes). Price index reconstructed as exp(cumulative log return).
* Out-of-era replication: Shiller / datasets monthly S&P 500 composite, January 1871 to September 2026
  (`datasets/s-and-p-500`, GitHub). Price returns only (the dividend column is not maintained after 2023).

## Widgets (all computed with information up to the close of day t)

| id | widget | definition |
|---|---|---|
| chg1d | 1-day change | r_t |
| chg5d | 5-day change | sum of last 5 log returns |
| chg21d | 1-month change | sum of last 21 |
| mom12_1 | 12-1 momentum | sum r over t-251..t-21 |
| ma200 | gap to 200-day MA | log P_t − log SMA200(P) |
| cross | golden/death cross | +1 if SMA50 > SMA200 else −1 |
| hi52 | distance from 52-week high | log P_t − log max(P over 252 days) |
| dd | drawdown from all-time high | log P_t − log max(P since start) |
| rsi14 | RSI(14), Wilder smoothing | 0-100 |
| boll | Bollinger %b (20, 2σ) | (P − lower)/(upper − lower) |
| z1d | today's move in sigmas | r_t / sd(r over last 63 days) |
| streak | up/down streak | signed count of consecutive same-sign days |
| rv21 | 21-day realised vol | sqrt(252 · mean r²) over last 21 days |
| volratio | short/long vol ratio | log(RV5 / RV63) |

## Targets (from t+1 to t+h, h = 21 trading days)

* **ret** — the forward 21-day log return.
* **vol** — log of the forward 21-day realised volatility.
* **mdd** — the forward 21-day maximum drawdown (≤ 0) of the cumulative return path.

Monthly replication (Shiller): widgets chg1m, mom12_1, ma10m, hi12m, dd, rv12m, z1m; targets the next-month return,
the next-12-month return, log forward 3-month volatility and the forward 12-month drawdown. Because Shiller prices are
**monthly averages of daily closes**, consecutive monthly returns are mechanically autocorrelated (Working, 1960). Every
monthly target therefore skips one month: it starts at the end of month t+1, never at t. Training: first 360 months.

## Out-of-sample design

* Per asset, an expanding-window OLS of the target on one widget (plus a constant). The first 1,260 trading days
  (5 years) after the asset's first complete widget set are training only. Refits every 21 days, using only
  observations whose target window is fully observed at the refit date (21-day embargo, so no overlapping target
  leaks into training).
* Baseline (standalone tests): the prevailing expanding mean of the target.
* Baseline (incremental tests, vol and mdd only): an expanding OLS on log(rv21) alone (log, to match the log-vol target); the challenger adds the widget.
* Statistic: Campbell–Thompson R²_OS = 1 − SSE_model / SSE_baseline.
* Test: Clark–West (2007) MSPE-adjusted statistic, one-sided, with a Newey–West HAC variance (lag = 2h) for
  overlapping forecasts. Pooled tests average the CW adjusted loss differential across assets on each date.
* Multiple testing: Benjamini–Hochberg at q = 0.10 within each family (standalone pooled; incremental pooled;
  per-asset), reported alongside raw p-values.
* Stress windows: GFC (2008-09-01 to 2009-03-31), COVID (2020-02-15 to 2020-05-31), 2022 rate shock
  (2022-01-01 to 2022-10-31). Reported descriptively (too few independent months for inference).
* Horizon profile: the same pipeline at h = 1, 5, 21, 63 for the information half-life figure.

## What would change the conclusion

A significant pooled return forecast that survives FDR (falsifies H1); no incremental risk information in any trend
widget (falsifies H3); results confined to a single asset or a single stress window.

## Changes after pre-registration

1. *Found when running, before reading any result:* with 252 days of widget warm-up plus 1,260 training days the daily
   out-of-sample period starts in April 2012 (2013 for HYG, 2016 for URA), so the GFC window cannot be evaluated daily.
   The GFC is covered by the monthly Shiller replication instead (its OOS period starts in 1902 and includes 1929,
   1937, 1974, 1987, 2000 and 2008). No other change.
2. *Added after seeing the first pooled results:* Clark–West rejects for several widgets whose R²_OS is *negative*
   (it tests whether the widget carries information in population, net of estimation noise). A terminal user only
   sees the estimated forecast, so a Diebold–Mariano test of the raw (unadjusted) MSE difference, HAC lag 2h, one-sided,
   with the same BH families, is added as a **secondary, stricter** criterion. Both are reported; H1–H3 are judged on
   the pre-registered CW criterion and, separately, on DM. This addition is exploratory.
3. *Referee review, after results:* BH was also applied within the 14 return tests alone, as a robustness check on
   H1, because the pre-registered family pools return tests with strongly significant volatility tests. Both are reported.
