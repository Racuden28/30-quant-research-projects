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
| 02 | Returns & volatility dashboard | | ⏳ Planned |
| 03 | Moving-average backtest | | ⏳ Planned |
| 04 | Black–Scholes pricer | | ⏳ Planned |
| 05 | Efficient frontier | | ⏳ Planned |
| 06 | Value-at-Risk calculator | | ⏳ Planned |
| 07 | [The Kelly Fraction You Cannot Know](07-kelly-estimation-risk/) | How many years of data does it take to know the Kelly bet, and what survives out of sample when you don't? | ✅ Published |
| 08 | Correlation structure | | ⏳ Planned |
| 09 | Beta & CAPM regression | | ⏳ Planned |
| 10 | [What on a Market Screen Is Worth Looking At?](10-market-terminal/) | Which of fourteen terminal widgets forecast returns, volatility or drawdowns out of sample, beyond realised volatility? | ✅ Published |
| 11 | Pairs trading | | ⏳ Planned |
| 12 | Momentum with transaction costs | | ⏳ Planned |
| 13 | GARCH volatility forecasting | | ⏳ Planned |
| 14 | Implied volatility surface | | ⏳ Planned |
| 15 | Binomial tree pricing | | ⏳ Planned |
| 16 | Delta-hedging simulation | | ⏳ Planned |
| 17 | Fama–French factors | | ⏳ Planned |
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
