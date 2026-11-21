# LinkedIn post (project 02)

**Attach:** `figures/fig1_which_choice_matters.png` (main image). Optional second image:
`figures/fig3_conditional_bias.png`. Put the GitHub link in the first comment, not in the post, because
LinkedIn shows posts with external links to fewer people.

---

Every volatility dashboard shows the same number: 21 days of closing prices, standard deviation, times √252. I wanted to know how good that number is as a forecast, and which of its hidden choices matters most.

Project 2 of 30 in my quant research series: "What Your Volatility Dashboard Gets Wrong"

I built 120 versions of the dashboard (5 estimators × 12 windows × 2 horizon scalings) plus HAR and GJR-GARCH benchmarks. I then forecast next month's volatility for SPY and the nine SPDR sector ETFs at every trading day from 2005 to 2026. That is about 8 million out-of-sample forecasts per horizon.

What I found:

📏 The estimator matters most, and the textbook picks the worst one. Using the day's high, low and open (Parkinson, Garman–Klass) instead of the close cuts forecast error by 13%. Tuning the window only gains 6%.

🎯 More choice did worse. Picking the best of all 120 versions each year using past data gained less than just switching the estimator. With 22 years of data, a big menu is mostly noise-fitting.

📉 The textbook number is wrong in a predictable direction. When it says 9% vol, the next month delivers 14%. When it says 37%, the next month delivers 34%. A GJR-GARCH is within a few percent in every state, and 25% better overall.

⏱️ √t overstates risk. In these sectors, only about 76% of the √t-scaled one-month return variance actually arrived. Correcting it helps under one loss function and not under another. There is no free lunch.

🧹 Free data are dirty. Before 2010, the sector ETFs' open was the day's high or low on 25% of days (SPY: 2.5%). Range estimators still won in 77% of ETF-years.

⚠️ The most common shortcut, Parkinson without the overnight gap, is 44% worse than the textbook. It misses roughly a third of the variance.

The interactive dashboard shows the textbook number next to the forecasts that beat it, for every sector.

Full paper, code, data and dashboard (everything reproducible with one command) are linked in the first comment.

#QuantitativeFinance #Volatility #RiskManagement #Python #Research #DataScience

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/02-volatility-dashboard-audit
Article version: https://racuden28.github.io/30-quant-research-projects/02-volatility-dashboard-audit/
