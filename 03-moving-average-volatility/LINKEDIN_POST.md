# LinkedIn post (project 03)

**Image:** `figures/fig2_decomposition.png` (alternative: `figures/fig3_return_vs_risk_forecast.png`)

---

Everyone "knows" why the 200-day moving average works: it gets you out of the market when volatility is high.

I tested that explanation on 10 equity markets, the S&P 500 back to 1933. It's wrong.

I built a placebo that copies the rule's exposure in every volatility state and knows nothing else. That pure volatility-timing piece has a negative effect on the Sharpe ratio in 10 of 10 markets. Each market alone is noisy, but 10 out of 10 with the same sign is hard to get by chance. Stepping out when markets are turbulent also means missing the violent rebounds.

So where does the edge come from? I split the rest into two parts:

📉 Risk: while the rule is invested, volatility is 1.4–2.3 points a year lower than the placebo's. Significant in 10 of 10 markets after multiple-testing correction. A market below its trend is a market whose risk is about to rise, even at the same volatility today.

📈 Returns: about +2 points a year in 8 of 10 markets, but only the S&P 500 is significant after the correction.

And the uncomfortable part: on the S&P 500, all of the rule's Sharpe advantage over buy & hold was earned before it was published in 1992. Since then it has had the same Sharpe as buy & hold, with half the drawdown.

My takeaway: the moving average is a risk forecaster, not a return forecaster, and not a volatility filter. It belongs in a risk overlay, not in an alpha model.

Methods: pre-registered design, no look-ahead, 5 bp costs, HAC and block-bootstrap inference, Holm and Romano–Wolf corrections, 9 rules × 10 markets, a stress test from 1937 to 2022, and a limitations section.

Full paper, code and every number as CSV: https://racuden28.github.io/30-quant-research-projects/03-moving-average-volatility/

Project 3 of 30 in my self-directed quant research series.

#QuantitativeFinance #TrendFollowing #Backtesting #RiskManagement #Python
