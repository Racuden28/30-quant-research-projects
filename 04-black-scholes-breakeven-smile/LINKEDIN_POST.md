# LinkedIn post (project 04)

**Attach:** `figures/fig2_breakeven_smiles.png` (main image). Optional second image:
`figures/fig6_oos_bias.png`. Put the GitHub link in the first comment, not in the post, because LinkedIn
shows posts with external links to fewer people.

---

Every Black-Scholes project prices an option with historical volatility and stops there. I asked a different question: which volatility *should* you have used?

Project 4 of 30 in my quant research series: "The Smile History Would Have Charged."

Black-Scholes is also an accounting identity. A delta-hedged option seller earns ½·Γ·S²·(σ²dt − R²) every day. So for any path you can solve for the volatility at which the seller breaks even, strike by strike. I did that for every trading day from 2007 to 2026 on 7 asset classes: US and EM equities, high yield, oil, Treasuries, gold and uranium. That is 33,719 one-month option paths per strike.

What I found:

📍 At the money, the textbook recipe works. Trailing 63-day vol is unbiased on all 7 assets.

📉 In the wings it fails badly. A put 2.5 standard deviations below spot needed 1.7× trailing vol on the S&P 500 to break even, and 1.8× on high-yield credit. The steepest realised skew in the sample is credit's, not equities'.

⚖️ Treasuries, gold and uranium show a symmetric smile with no skew. Every one of the 7 assets has a statistically significant smile, so flat Black-Scholes is rejected everywhere.

🧪 The skew comes from the leverage effect, not from jumps. In simulation, large negative jumps produce a symmetric smile, while GARCH with leverage and no jumps reproduces the equity skew.

💰 VIX was above the volatility a hedger actually needed on 83% of days, by 3.6 points on average. The gap flipped negative in 2008 and in 2020.

🔮 The historical smile is forecastable. Re-estimated each month using only options that had already expired, it removed 74–93% of Black-Scholes' pricing bias out of sample (2012–2026).

Under the hood: the textbook N(d1)/N(d2) formula returns zero correct digits for 18% of deep out-of-the-money puts. The pricer here keeps at least 11.8 digits everywhere, checked against a 60-digit reference.

Full paper, code and data (one command to reproduce) in the first comment.

#QuantitativeFinance #Options #BlackScholes #Volatility #Python #Research

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/04-black-scholes-breakeven-smile
Article version: https://racuden28.github.io/30-quant-research-projects/04-black-scholes-breakeven-smile/
