# LinkedIn post: project 08

**Image:** `figures/fig1_three_heatmaps.png` (or `figures/fig2_gap_decomposition.png` as a second slide)

---

"In a crash, all correlations go to one."

I tested it on 33 years of daily data, and the crash-day heatmap shows the opposite. Across 19 US large
caps, average correlation is 0.32 on all days and 0.20 on the worst 5% of days.

Neither number is telling you about the market. Picking days by one tail of the market mechanically
removes correlation, even in a world where nothing changes. So I built that world: I kept every stock's
real volatility path, held the latent correlation fixed, and re-ran the same heatmap 1,000 times.

What's left once you subtract it:

→ Crash days do carry extra correlation (+0.07), but rally days carry exactly as much. Over 1990–2026 I
find no downside asymmetry.
→ The real effect is a volatility *regime* you can see the evening before: +0.17 correlation, with no
selection bias.
→ For a stock–bond portfolio the crash-day heatmap gets the sign wrong. It says Treasuries hedge less in
sell-offs. In high-volatility regimes they hedge more. 2022 was the exception.
→ For next month's allocation, slow and shrunk heatmaps (DCC, Ledoit–Wolf) beat the standard 1-year
heatmap out of sample. "Fast" crisis heatmaps lose 6–13%.

Correlations do rise in crises (+0.51 in COVID). A crash-day heatmap is just the wrong way to see it.

Full paper, code and every number: [link to the project page]

Project 8 of 30 in my self-directed quant research series.

#QuantitativeFinance #RiskManagement #Correlation #PortfolioConstruction #Python #Research
