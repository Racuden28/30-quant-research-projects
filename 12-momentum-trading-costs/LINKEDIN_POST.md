# LinkedIn post: project 12 (draft, not published)

Image: figures/fig1_turnover_frontier.png

---

Every paper on momentum and trading costs reports one number: the break-even cost. It is the cost per dollar traded at which the strategy stops making money.

That number is an average. It tells you whether to run momentum, but not how fast to trade it. For that you need the marginal break-even: what the last, most frequent trades actually earn.

I measured both on 98 years of US industry momentum (49 industries, 1928–2026), with 21 ways of trading the same signal and exact turnover accounting.

What I found:

📉 The textbook strategy breaks even at 71 bps per dollar traded. Its last trades break even at only 24 bps. Trading at half speed cuts turnover by 38% and gives up just 13% of the premium.

⏳ The reason is that the signal ages slowly. A ranking that is 1–3 months old kept 71% of a fresh one's premium. The decay curve alone predicts the return of every slower design.

🔁 After 2000 the gap closed. Every trade now earns about the same per dollar, so slowing down no longer helps.

🌪️ Momentum does trade most when trading is expensive: in 1932, 2008 and 2009 its monthly costs ran 3–7 times normal. But over a century, a constant-cost backtest understates the true cost by only 8%.

📊 Volatility scaling was the one change that won at every cost level: about +0.18 net Sharpe. It trades more on average, but less exactly when trading is expensive.

🌍 The version you can actually trade, 42 single-country ETFs from 1998 to 2026, has no momentum premium at all, before or after costs. Also, the popular formulas that estimate spreads from daily prices put SPY's spread at about 100 times its real one-cent quote.

Takeaway: report two break-evens, not one. The average tells you whether a strategy is worth running. The marginal tells you whether you are trading it at the right speed.

Full paper, code and data (one command reproduces every number): [link]

Project 12 of 30 in my self-directed quant research series.

#QuantitativeFinance #Momentum #TradingCosts #Backtesting #AssetPricing #Python
