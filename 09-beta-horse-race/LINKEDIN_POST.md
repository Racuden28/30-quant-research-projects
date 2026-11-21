# LinkedIn post (project 09)

**Attach:** `figures/fig1_horizon_flip.png` (main image). Optional second image: `figures/fig4_sml.png`.
Put the GitHub link in the first comment, not in the post, because LinkedIn shows posts with external
links to fewer people.

---

Every finance course teaches the same beta: 5 years of monthly returns, one OLS regression. I wanted to know how good that number really is, and whether a better one fixes the CAPM.

Project 9 of 30 in my quant research series: "The Beta You Hedge With".

I ran 9 beta estimators (textbook OLS, daily OLS, Blume, Vasicek, EWMA, Dimson, Frazzini–Pedersen, Welch's slope-winsorized beta and a combination) on the 49 Fama–French industries, every month from 1929 to 2026: 54,245 out-of-sample forecasts each. Each one is scored by what a practitioner pays for a bad beta: the risk left over after hedging the market.

What I found:

📉 Daily data cut the textbook's beta error by about 40% for next month's hedge (Welch: 0.58 of the textbook's error, 95% CI 0.53–0.64). It holds in every era and every crash since 1929.

🔄 Then it disappears. Hedge over the next 12 months instead, and the same daily betas are 6–10% worse than the textbook. Only lag-corrected betas (Dimson) keep an edge.

🔍 The reason is stale prices. Industries of small firms react to the market with a delay, so daily regressions understate their beta by 0.12–0.15 before 2000 (Spearman with firm size: −0.83). Since 2000 the gap has shrunk by two thirds.

🌪️ Beta error is priced by volatility. The hedge penalty is (β̂ − β)² × market variance, so the 10% most volatile months deliver about half of the total gain.

📏 A better beta does not rescue the CAPM. Whatever estimator you sort on, the security market line rises at only 0.17–0.34 of the market premium.

The practical rule: estimate beta at the horizon you will hedge at, and shrink it.

Full paper, code and data (every number reproducible with one command) in the first comment.

#QuantitativeFinance #AssetPricing #CAPM #RiskManagement #Python #Research

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/09-beta-horse-race
Article version: https://racuden28.github.io/30-quant-research-projects/09-beta-horse-race/
