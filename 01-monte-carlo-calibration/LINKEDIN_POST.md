# LinkedIn post (project 01)

**Attach:** `figures/fig8_calm_vs_stress_all_models.png` (main image). Optional second image:
`figures/fig7_factorial_dynamics_vs_tails.png`. Put the GitHub link in the first comment, not in the
post, because LinkedIn shows posts with external links to fewer people.

---

Most Monte Carlo projects stop at a fan of simulated paths. I wanted to know something else: do the probabilities actually come true?

Project 1 of 30 in my quant research series: "Which Monte Carlo Tells the Truth?"

I ran 5 widely used simulators (GBM, Merton jumps, GJR-GARCH-t, Markov switching, block bootstrap) through 112,626 strictly out-of-sample forecasts, plus three controls (Gaussian GARCH, a 1-year-window GBM and RiskMetrics EWMA). That covers every week from 2008 to 2026 across 7 asset classes: US and EM equities, Treasuries, high yield, gold, oil and uranium.

What I found:

📉 Right on average, wrong in every regime. GBM's 90% interval contains 92% of outcomes overall. But in calm weeks it contains 97%, and in the most stressed fifth of weeks only 78% (95% CI 73–83%). Same result on every one of the 7 assets.

🧪 It is the volatility dynamics, not the fat tails. Adding jumps to a static model changes stressed-week coverage by about 1 point. Adding GARCH dynamics, even with Gaussian shocks, moves it from 78% to 91%. But not just any dynamics: RiskMetrics EWMA, which has no mean reversion, overshoots in the other direction (85% in calm weeks).

🔥 In the 2008 crisis, the static models saw moves outside their 98% band 10 to 11 times more often than promised. GJR-GARCH-t: 1.0x.

🧩 Merton's "jumps" are not rare. In 88% of 1,567 re-estimations the fitted intensity hits its cap. The model uses jumps to approximate volatility clustering.

⚠️ Nobody was ready for COVID. Even the best model breached 7.6x too often. A shock that fast needs forward-looking information, such as option-implied volatility.

The lesson for anyone using Monte Carlo for risk: check calibration by regime, not on average. A single coverage number can hide a 20-point error in exactly the weeks that matter.

Full paper, code and data (every number reproducible with one command) in the first comment.

#QuantitativeFinance #RiskManagement #MonteCarlo #Python #Research #FinancialEngineering

---

**First comment:**
Paper and code: https://github.com/<your-github-username>/30-quant-research-projects/tree/main/01-monte-carlo-calibration
Article version: https://<your-github-username>.github.io/30-quant-research-projects/01-monte-carlo-calibration/
