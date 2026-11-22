# LinkedIn post: project 18

**Image:** `figures/fig1_two_uses_of_pca.png` (second slide: `figures/fig4_geometry_not_rotation.png`)

---

"Three factors explain 99% of the yield curve."

It's the first thing every fixed-income course teaches, and the next thing it teaches is the PCA hedge:
pick three bonds and size them so your book has zero exposure to level, slope and curvature.

I tested that hedge out of sample on 50 years of daily US Treasury data (1976–2026), with 1,000 random
curve books re-hedged monthly. The benchmark was the simple thing desks do with no model: split each
bond's DV01 between its two neighbouring hedge points.

What I found:

→ The textbook PCA hedge was the worst hedge I tested. With 2Y/5Y/10Y it left a median of 20× the risk of
the no-model baseline, and it won in only 11% of months.
→ The cause isn't unstable factors. It's geometry. Seen at 2Y/5Y/10Y, "curvature" looks too much like
"slope", so the hedge asks for huge offsetting positions. Hedge size explains the failures (corr 0.96);
factor rotation doesn't (−0.21).
→ The same PCA used as a risk model (minimum-variance hedge) beats the baseline by 24–32%. Honest caveat:
two thirds of that comes from the 1Y point. Between hedge points the gain is 7–12%.
→ For hedging, the best number of factors is all of them. Stopping at three costs 3–10%.
→ The "99%" mostly comes from yield levels and fitted curves. For a curve trade, three factors miss 16% of
daily risk, and the Fed's smoothed curve hides three quarters of that.
→ In SVB (March 2023) the model-free hedge beat every estimated one.

PCA explains the curve. Hedging it is a different job.

Full paper, code and every number: [link to the project page]

Project 18 of 30 in my self-directed quant research series.

#FixedIncome #QuantitativeFinance #RiskManagement #InterestRates #PCA #Python
