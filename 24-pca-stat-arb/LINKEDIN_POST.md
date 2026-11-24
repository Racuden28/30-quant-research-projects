# LinkedIn post (draft, not posted; Carlos reviews first)

**Project 24/30: The Filter That Lets Noise Through**

PCA statistical arbitrage (Avellaneda & Lee, 2010) has a part that looks very scientific. You fit each stock's
residual as an Ornstein-Uhlenbeck process and only trade the ones that "mean-revert within 30 days".

I tested that step against a placebo market: the same stocks and the same daily covariances, but every day's
returns multiplied by a random sign, so nothing can be predicted.

What I found, on 26 years of S&P 500 stocks and on every Binance coin including the 217 that were delisted:

🔹 The filter passes 99% of real residuals and 98% of pure noise. Regression residuals sum to zero, so their
cumulative path always looks mean-reverting.
🔹 The famous s-score is just a linearly weighted 60-day residual sum (correlation 0.997). A plain 5-day
z-score does as well.
🔹 In stocks the edge is real: gross Sharpe 1.31 vs 0.00 ± 0.15 for 20 placebo markets. It breaks even at about
5 bp of trading cost, and below zero once you wait one day to trade and pay 5 bp.
🔹 In crypto, the same machine does worse than every placebo. Coin residuals do reverse, but the book is short
the 0.4% of days when a coin jumps by more than 30%, and those days decide everything.

A side lesson: my first two placebo markets were biased themselves (Sharpe +0.8 and −3.2 on pure noise). Both
are documented and unit-tested in the repo.

Full paper, code and data: [link to GitHub Pages]

#QuantitativeFinance #StatisticalArbitrage #PCA #Backtesting #Crypto #Python

*Image: figures/fig1_filter_and_edge.png*
