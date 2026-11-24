# LinkedIn post (project 26)

**Attach:** `figures/fig1_two_volvols.png` (main image). Optional second image: `figures/fig7_chain_snapshot.png`.
Put the GitHub link in the first comment, not in the post.

---

Calibrating the Heston model usually needs a full option chain, and that data costs money.

But CBOE gives away six numbers every day that summarise the chain: the VIX term structure (9 days, 30 days, 3 and 6 months), SKEW and VVIX. Under Heston each one has an exact formula. They pin the five parameters in a neat triangle: the term structure gives the variance level and mean reversion, VVIX gives the vol of vol, and SKEW gives the spot-vol correlation.

Six numbers and five parameters leaves one spare equation, so every day carries a free consistency test.

Project 26 of 30 in my quant research series: "Heston From Six Public Numbers."

What I found on 4,681 days (2008–2026):

📉 The test fails on 90% of days. SKEW needs 2.2× the vol of vol that VVIX allows, even with a correlation of −1.

🧩 At least 55% of the S&P 500's priced 30-day skewness is out of reach of diffusive stochastic volatility, and that share has grown from about 25% in 2008 to about 70% since 2023.

🎯 VVIX and history agree. The vol of vol implied by VVIX is 0.58. A robust estimate from realised variance moves is also 0.58. SKEW is the one that doesn't fit.

🔁 Heston's own reactions are the wrong size: it says VVIX should rise 0.5 points per VIX point. The data say 2.

✅ What works is the term structure. Out of sample, moving only the variance level predicts next-day VIX3M and VIX6M 33% and 64% better than simple rules (Holm p ≤ 0.001).

🧪 Live check on today's S&P 500 chain (10,454 quotes): my replication of the CBOE formula gives VIX 27.43 against 27.42 published. The best Heston fit still misses by 2.4 vol points (33× the bid-ask), and to get there it needs a vol of vol that implies VVIX 134 against 119 observed.

Takeaway: Heston's vol of vol is real, but SKEW prices something else, namely jumps.

Exact closed forms, 14 tests, block-bootstrap CIs, Holm corrections and a full referee self-review are in the repo.

#QuantFinance #Options #Volatility #Heston #Research
