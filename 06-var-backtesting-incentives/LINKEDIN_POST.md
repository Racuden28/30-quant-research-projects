# LinkedIn post (project 06)

**Attach:** `figures/fig1_break_even.png` (main image). Optional second image:
`figures/fig4_score_vs_capital.png`. Put the GitHub link in the first comment, not in the post,
because LinkedIn shows posts with external links to fewer people.

---

Every VaR project counts breaches. I wanted to know something else: does the rule that turns breaches into capital make honesty the cheapest strategy?

Project 6 of 30 in my quant research series: "Is It Cheaper to Be Wrong?"

Since 1996, a bank's market-risk capital is a multiple of the VaR it reports itself. The multiple goes up (the Basel "traffic light") only after 5 or more breaches in a year. So what happens if a bank reports 80% of its true VaR?

What I found:

📐 The 1996 penalty table is the Gaussian break-even point, almost to the second decimal. If losses are normal, shading your VaR saves almost nothing: 0.40, 0.65, 0.88, 0.99 is what break-even requires, and 0.40, 0.65, 0.85, 1.00 is what the table charges.

🔥 With fat tails, it pays. With Student-t (ν = 3) losses, a bank can report 75% of its true VaR, keep the chance of a red-zone year under 5%, and save 17.6% of its capital. A GARCH Monte Carlo gives 17.1%. FRTB inherits the same incentive.

📊 On real data the room is smaller but always points the same way. Six standard models (HS, RiskMetrics, FHS, GARCH-t, GJR skew-t, GARCH + EVT), 7 ETFs and a multi-asset book, 216,552 out-of-sample forecasts from 2008 to 2026. In all 48 series-model pairs, the capital-minimising scale is below what a proper scoring rule calls honest (median gap 12%).

⚖️ Same forecasts, opposite incentives. The FZ0 joint VaR/ES score is U-shaped around honest reporting. The capital charge keeps falling until red quarters appear.

🚦 But there is no free lunch ex ante. A bank that tries to shade in real time saves 7% and spends 28% of quarters in the red zone. The traffic light's real teeth are crises and the red zone, not the plus factor.

The lesson: a backtest that only counts exceptions is not incentive compatible once tails are fat. A capital add-on tied to a strictly consistent score would be.

Full paper, code and data (every number reproducible with one command) in the first comment.

#QuantitativeFinance #RiskManagement #ValueAtRisk #Basel #FRTB #Python
