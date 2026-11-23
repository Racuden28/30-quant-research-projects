# LinkedIn post (project 11)

**Attach:** `figures/fig1_convergence_is_not_evidence.png` (main image). Optional second image:
`figures/fig3_where_reversion_lives.png`. Put the GitHub link in the first comment, not in the post, because
LinkedIn shows posts with external links to fewer people.

**Do not publish until Carlos has reviewed it.**

---

"Most of the trades converged." That sentence is in almost every pairs-trading backtest. I wanted to know how much it actually proves.

Project 11 of 30 in my quant research series: "Convergence Is Not Evidence".

I built a placebo market: the same assets, the same volatilities, fat tails and correlations, but with every bit of relative mean reversion removed. Then I ran the textbook pairs strategy (Gatev, Goetzmann & Rouwenhorst) on both, with the design pre-registered, on two universes: 49 US industries over a century (1927–2026) and 22 country ETFs (2002–2026).

What I found:

🎲 Convergence proves nothing. 36% of trades converge on real industry data. In the placebo market, where nothing can revert: 36%. Country ETFs: 36% vs 32%.

🔍 Most "cointegrated" pairs are luck. 6.5% of industry pairs pass the Engle–Granger test at 5%; the placebo gives 5.2%. So roughly four in five discoveries are what chance alone produces.

📉 Industry pairs lost money for a century: −1.0% a year gross (t = −3.0). Their spreads don't come back; they keep drifting apart. That's industry momentum, and the strategy is short it.

📈 Country ETF pairs do revert, at every horizon up to three months: +1.3% a year gross (t = 2.7). After a realistic 5 bp per leg since 2010: +0.4% a year, statistically indistinguishable from zero.

🧪 The most useful finding was about my own method. My first placebo let the fake market "know" each asset's future drift, and it would have confirmed exactly the edge I expected (t = 3.6). It was an artefact. The fix, a second bug in the block bootstrap and the tests that now catch both, are in the paper.

The habit I'm keeping: before trusting any stat-arb backtest, run it on a placebo market built from the same data, and make sure the placebo can't see the future.

Full paper, code and data (every number reproducible with one command) in the first comment.

#quant #pairstrading #statisticalarbitrage #backtesting #research

---

**First comment:** Paper and code: https://racuden28.github.io/30-quant-research-projects/11-pairs-trading/ ·
https://github.com/Racuden28/30-quant-research-projects/tree/main/11-pairs-trading
