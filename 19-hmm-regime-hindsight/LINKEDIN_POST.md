# LinkedIn post (project 19)

**Image:** `figures/fig1_hindsight_ladder.png` (alternative: `figures/fig9_belief_vs_reality.png`)

---

A hidden Markov model finds "bull" and "bear" regimes in the S&P 500, and the classic backtest looks great: stay in the market only in the calm regime and the Sharpe ratio is 1.01 since 1934. Buy & hold gets 0.32.

I took that backtest apart, one source of look-ahead at a time, on 10 equity markets.

1️⃣ Most tutorials trade on *smoothed* regime probabilities. Those use the whole sample, including every return after the day you trade. Switching to the *filtered* probability, which only uses the past, costs a median 0.61 of Sharpe. 10 of 10 markets, all significant.

2️⃣ Re-estimating the parameters in real time, which is the fix everyone reaches for, costs almost nothing (0.01). It fixes the wrong problem.

3️⃣ In real time and after costs, the HMM rule trails buy & hold in 7 of 10 markets, and it does exactly as well as a placebo that only looks at volatility.

Why? Hindsight leaves each sell-off about 2 trading days before the filter can, and those days cost 41–103 bp each.

The part I didn't expect: the "bear" regime's large negative mean is a hindsight artefact. On the days the real-time model stepped out because it expected losses, the index went *up* in 10 of 10 markets. If the fitted HMM were the true model, real-time timing would beat buy & hold by about half a Sharpe point. Real data falls below the model's own 5th percentile in every market.

What's left is useful, but it's not alpha: the real-time regime probability forecasts next-month volatility better than EWMA (9 of 10 markets after Holm). It's a risk model, not a return model.

Methods: an HMM written from scratch and tested against brute-force enumeration, walk-forward refits every month, a look-ahead test that changes the future, 5 bp costs, HAC and block-bootstrap inference with Holm across markets, a simulated "model world", stress windows from 1987 to 2022, and a limitations section.

Full paper, code and every number as CSV: https://racuden28.github.io/30-quant-research-projects/19-hmm-regime-hindsight/

Project 19 of 30 in my self-directed quant research series.

#QuantitativeFinance #RegimeSwitching #HiddenMarkovModel #Backtesting #Python
