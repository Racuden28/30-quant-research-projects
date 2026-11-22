# Pre-registration: the five-factor model's holdout exam

Written on 2026-11-22, after downloading the data (Ken French library, CRSP vintage 202609) and
before computing any statistic on the holdout period. The only things looked at before writing this
were the file headers, the first and last rows (to check dates) and the row counts.

## Samples

* **In-sample (IS):** 1963-07 to 2013-12, the sample of Fama and French (2015).
* **Holdout (HO):** 2014-01 to 2026-09 (153 months). These returns did not exist when the model was
  proposed and chosen.
* **Real-time (RT):** walk-forward from 1983-12 (20 years of history) to 2026-09, expanding window,
  re-estimated every December.

## Models

CAPM {Mkt}; FF3 {Mkt, SMB, HML}; C4 {FF3 + UMD}; FF5 {Mkt, SMB, HML, RMW, CMA}; FF5−HML
{Mkt, SMB, RMW, CMA}; FF6 {FF5 + UMD}; FF6−HML {Mkt, SMB, RMW, CMA, UMD}.

## Primary hypotheses (tested on the holdout; Holm correction across H1 to H4)

* **H1, HML stays redundant.** In the holdout, the intercept of HML regressed on Mkt, SMB, RMW and
  CMA is not different from zero (two-sided, Newey–West 6 lags), and not different from its
  in-sample value (interaction test in a pooled regression).
* **H2, dropping HML is free out of sample.** The tangency portfolio of FF5−HML, with weights fixed
  from in-sample data, has a holdout Sharpe ratio no lower than the FF5 tangency portfolio
  (stationary bootstrap of the difference, mean block 6 months, 10,000 draws).
* **H3, FF5 still prices better than FF3.** On the 75 sorted portfolios (25 size–B/M, 25 size–OP,
  25 size–Inv), the holdout average absolute alpha A|a| of FF5 is lower than that of FF3, using
  out-of-sample alphas (betas estimated on the prior 60 months only). Bootstrap over months.
* **H4, choosing the model in real time adds nothing.** A strategy that, every December, picks the
  model with the highest bias-adjusted in-sample squared Sharpe ratio and holds its tangency portfolio
  for the next year does not earn a higher Sharpe ratio than always holding FF6.

## Secondary (descriptive, no Holm)

Factor premia IS vs HO with CIs; GRS tests on each test-asset family IS and HO; out-of-sample
Sharpe haircut by model size; the real-time history of HML's spanning t-statistic; stress windows
(2007-10 to 2009-03, 2020-02 to 2020-04, 2022-01 to 2022-10); the 10 momentum deciles and 49
industries as extra test assets; data-vintage drift (2017 vintage in `linearmodels` vs 2026).

## Decision rules

H1, H2 and H4 are claims that nothing changed, so each is tested by its null and "holds" if the
null is not rejected at 5% after Holm (H2 and H4 one-sided: the null is rejected only if FF5−HML,
or the real-time selector, is significantly *worse*). H3 is a claim of superiority, so it "holds" only
if FF5's A|a| is significantly lower than FF3's (one-sided) after Holm. Because the holdout is short, each
result is reported with the minimum effect detectable at 80% power, so "not rejected" is never read as
"confirmed" when the test had no power.
