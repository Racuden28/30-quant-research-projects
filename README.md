# 30 Quant Research Projects

A self-directed research programme in quantitative finance: thirty projects, from Monte Carlo and
portfolio construction to stochastic volatility, statistical arbitrage, market making and deep hedging.

Every project starts from a **research question**, not a tutorial. Each one is evaluated strictly out
of sample, compared against honest baselines, and published with its code, data pipeline, every number
in the write-up as CSV, and a section on what does *not* work.

**Website:** https://racuden28.github.io/30-quant-research-projects/ (served from `docs/`)

## Projects

| # | Project | Research question | Status |
|---|---|---|---|
| 01 | [Which Monte Carlo Tells the Truth?](01-monte-carlo-calibration/) | Do the probabilities from five standard return simulators come true out of sample, in calm and in crisis? | ✅ Published |
| 02 | [What Your Volatility Dashboard Gets Wrong](02-volatility-dashboard-audit/) | Which hidden choice in a volatility dashboard (estimator, window, √t scaling) matters out of sample? | ✅ Published |
| 03 | [Not a Volatility Filter: What the 200-Day Moving Average Actually Knows](03-moving-average-volatility/) | Is the 200-day rule just a volatility filter, or does the trend know something today's volatility does not? | ✅ Published |
| 04 | [The Smile History Would Have Charged](04-black-scholes-breakeven-smile/) | What Black–Scholes volatility, strike by strike, would have made a delta-hedged option seller break even, and where does the smile come from? | ✅ Published |
| 05 | [Where Does the Efficient Frontier Break?](05-efficient-frontier/) | When mean-variance fails out of sample, is it the expected returns or the covariances? | ✅ Published |
| 06 | [Is It Cheaper to Be Wrong?](06-var-backtesting-incentives/) | Does the Basel VaR traffic light reward banks for understating risk when tails are fat? | ✅ Published |
| 07 | [The Kelly Fraction You Cannot Know](07-kelly-estimation-risk/) | How many years of data does it take to know the Kelly bet, and what survives out of sample when you don't? | ✅ Published |
| 08 | [Do Correlations Really Go to One in a Crash?](08-correlation-crash-illusion/) | Once the bias from how crash days are picked is removed, do correlations really rise in crises, and is it crash-specific? | ✅ Published |
| 09 | [The Beta You Hedge With](09-beta-horse-race/) | Which of nine beta estimators hedges best out of sample, and does any of them rescue the CAPM security market line? | ✅ Published |
| 10 | [What on a Market Screen Is Worth Looking At?](10-market-terminal/) | Which of fourteen terminal widgets forecast returns, volatility or drawdowns out of sample, beyond realised volatility? | ✅ Published |
| 11 | Pairs trading | | ⏳ Planned |
| 12 | Momentum with transaction costs | | ⏳ Planned |
| 13 | GARCH volatility forecasting | | ⏳ Planned |
| 14 | Implied volatility surface | | ⏳ Planned |
| 15 | Binomial tree pricing | | ⏳ Planned |
| 16 | [The Floor Under Delta Hedging](16-delta-hedging-floor/) | How much of an option's hedging risk can rebalancing more often remove, and what removes the rest? | ✅ Published |
| 17 | [The Five-Factor Model's Holdout Exam](17-five-factor-holdout/) | Does the five-factor model's 2013 verdict survive 13 years of data that did not exist when it was chosen? | ✅ Published |
| 18 | Yield-curve PCA | | ⏳ Planned |
| 19 | Regime detection with HMMs | | ⏳ Planned |
| 20 | Risk parity | | ⏳ Planned |
| 21 | Event-driven backtest engine | | ⏳ Planned |
| 22 | Purged cross-validation | | ⏳ Planned |
| 23 | Deflated Sharpe ratio | | ⏳ Planned |
| 24 | PCA statistical arbitrage | | ⏳ Planned |
| 25 | Kalman-filter hedge ratio | | ⏳ Planned |
| 26 | Heston calibration | | ⏳ Planned |
| 27 | Merton structural credit model | | ⏳ Planned |
| 28 | Market-making agent | | ⏳ Planned |
| 29 | Optimal execution | | ⏳ Planned |
| 30 | Deep hedging | | ⏳ Planned |

## Standards every project follows

1. **A falsifiable question** stated up front, and a headline answer with a number attached.
2. **Out-of-sample only.** Rolling or walk-forward estimation, no look-ahead, fixed seeds.
3. **Honest baselines.** The textbook method is always in the comparison.
4. **Proper statistics.** Formal tests with stated assumptions, and effect sizes rather than p-values alone.
5. **Limitations section.** What would change the conclusion, and what was not tested.
6. **One command to reproduce** every figure and table.

## Repository layout

```
30-quant-research-projects/
├── README.md                  this index
├── docs/                      GitHub Pages site (one article per project)
├── 01-monte-carlo-calibration/
│   ├── README.md              the paper
│   ├── src/  data/  results/  figures/
│   └── requirements.txt  run_all.sh
└── ...
```

## License

Code: MIT. Text and figures: CC BY 4.0. Market data belongs to its providers and is used for research only.
