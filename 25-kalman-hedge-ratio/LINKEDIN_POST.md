# LinkedIn post (draft, not published; Carlos reviews it first)

**Image:** figures/fig1_kalman_is_an_ewma.png

---

The Kalman filter is the go-to "advanced" hedge ratio. I tested what it adds out of sample, and most of it is
an EWMA in disguise.

Project 25 of my 30 quant research projects reuses two earlier ones on purpose: a beta horse race on 49 US
industries from 1929 to 2026 and a pairs-trading placebo market on 22 country ETFs.

What I found:

1️⃣ With the same memory, a Kalman beta and an exponentially weighted regression hedge equally well (difference
−0.08 % of the hedge loss, equivalence within ±1 % significant). Across every noise setting, from a 3-day to a
6,000-day memory, the two loss curves overlap.

2️⃣ The filter's own likelihood picks a memory of 9–15 days. The out-of-sample optimum is 35–55. Tuned that way,
the Kalman beta is 2.3 % worse than a plain 126-day EWMA. A simulation shows why: when volatility clusters
and tails are fat, the random-walk model reads big errors on volatile days as beta uncertainty and forgets too
fast.

3️⃣ Tune the same filter by past hedging loss instead and it becomes the best hedge ratio in the study.

4️⃣ In pairs trading, the Kalman spread converges on 99 % of trades and looks stationary for 100 % of pairs, on
real prices and on a placebo market where nothing should converge. Over 90 % of each "convergence" is the
filter moving its own intercept. What it earns is one-day reversal, and 5 bp per leg wipes it out.

Takeaway: a Kalman filter is a good way to write a time-varying hedge ratio and a poor way to tune one. And a
sophisticated estimator mostly changes how convincing the diagnostics look, not what the hedge is worth.

Paper, code and data (all reproducible): [link to project 25]

#QuantitativeFinance #KalmanFilter #Hedging #PairsTrading #Backtesting #Python
