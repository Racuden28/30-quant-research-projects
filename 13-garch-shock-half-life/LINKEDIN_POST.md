# LinkedIn post (project 13)

**Image:** `figures/fig1_shock_decay.png` (alternative: `figures/fig3_decade_stability.png`)

---

Fit a GARCH(1,1) to daily S&P 500 returns and it tells you that a volatility shock takes about 140 trading days to fade by half. Roughly seven months.

I wanted to know whether that's true, so I measured it on 10 equity indices, with the S&P 500 going back to 1928.

My first answer was dramatic: shocks are half gone in 17 days, against 81 implied by GARCH. Then I ran a placebo. I simulated worlds where GARCH is exactly right and applied the same measurement. It reported about half the true half-life. My yardstick was biased. It is a check that comparisons of GARCH persistence with the data rarely make explicit.

So I rebuilt the test against those simulated worlds. What survives:

⏱️ In the data, half the shock is gone after 16 days. In GARCH worlds, the same measurement gives 37 days (p = 0.02).

🐢 But after a year, 12% of the shock is still visible, against 0.5% in GARCH worlds (p = 0.02). Shocks fade faster *and* linger longer than one exponential allows.

📅 Fit GARCH decade by decade and the S&P 500's half-life is about 20 days. Over the full century it is 140. The long persistence comes from shifts in the volatility *level* between eras, not from how individual shocks decay. Parameter constancy is rejected in 9 of 10 markets.

📉 On 130 real shock days, out of sample, the variance over the next 6 to 12 months came in a third below GARCH's forecast. A two-speed (component) GARCH was right on average.

Takeaway: beyond about a month, a GARCH forecast is a forecast of the volatility regime, anchored wherever your estimation sample happens to sit. For longer horizons, use two speeds or a rolling anchor. And calibrate your persistence tests before you trust them.

Methods: walk-forward with yearly refits, 6 models, 5 horizons, local projections, filtered-historical-simulation null worlds, a parametric-bootstrap LR test, month-cluster bootstrap and Holm corrections. The limitations section says what didn't work.

Paper, code and every number as CSV: https://racuden28.github.io/30-quant-research-projects/13-garch-shock-half-life/

Project 13 of 30 in my self-directed quant research series.

#QuantitativeFinance #Volatility #GARCH #RiskManagement #Econometrics
