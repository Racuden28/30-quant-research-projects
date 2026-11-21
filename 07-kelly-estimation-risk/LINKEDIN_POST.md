# LinkedIn post (project 07)

**Attach:** `figures/fig1_knowability.png` (main image). Optional second image:
`figures/fig4_realised_growth.png`. Put the GitHub link in the first comment, not in the post.

---

Every Kelly criterion tutorial plugs in a historical mean and calls it the optimal bet. I wanted to know what happens when you can't know the mean, which is always.

Project 7 of 30 in my quant research series: "The Kelly Fraction You Cannot Know"

I combined closed-form theory, a Monte Carlo Kelly simulator built on real return paths, and an 18-year out-of-sample backtest of 15 sizing rules on 7 asset classes: US and EM equities, Treasuries, high yield, gold, oil and uranium.

What I found:

🎯 With 20 years of daily data, the sign of the Kelly bet is statistically clear for only 2 of 7 assets. Knowing the size of the bet to within ±0.5 (2.0x versus 2.5x leverage) would take more than 80 years of data for every asset, and 215 for the S&P 500.

📉 Full Kelly with an estimated mean is expected to lose to cash unless Sharpe² × years > 1. Out of sample it did: plug-in Kelly grew at −0.8% a year, T-bills +1.5%.

📈 Refusing to estimate the mean worked better. Assume the same Sharpe of 0.3 for every asset and size only by forecast volatility: +5.0% a year, better than plug-in Kelly on all 7 assets, and 54% of what perfect hindsight would have earned.

⚖️ The mean is worth 7 points a year between the best and worst input. Volatility timing is worth about 1. The problem is that the mean estimate is mostly noise, and Kelly turns noise into leverage.

🧪 Honest caveat: no pre-registered test survives multiple-testing correction. Eighteen years cannot rank leveraged strategies with confidence, which is the same lesson one level up.

💥 Kelly's drawdown law (full Kelly halves your wealth with probability 50%) holds on real data. Its tail does not: at half Kelly, crude oil lost 90% of wealth 9x more often than the law says.

The part of Kelly you can know is the denominator.

Full paper, code and data (every number reproducible with one command) in the first comment.

#QuantitativeFinance #KellyCriterion #PositionSizing #RiskManagement #Python #Research

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/07-kelly-estimation-risk
Article version: https://racuden28.github.io/30-quant-research-projects/07-kelly-estimation-risk/
