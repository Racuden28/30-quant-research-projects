# LinkedIn post (project 17)

**Attach:** `figures/fig1_holdout_report_card.png` (main image). Optional second image:
`figures/fig2_realtime_redundancy.png`. Put the GitHub link in the first comment, not in the post,
because LinkedIn shows posts with external links to fewer people.

---

In 2013 Fama and French chose their five-factor model using 50 years of data, and declared the value factor redundant. Every test of that model since has re-used the same 50 years.

But there are now almost 13 years of returns that did not exist when the model was chosen. So I gave it a holdout exam, with the hypotheses written down before I looked.

Project 17 of 30 in my quant research series: "The Five-Factor Model's Holdout Exam".

What I found (Ken French data, Jan 2014 to Sep 2026):

✅ Value stayed redundant. HML's alpha over the other factors was +0.04% a month (t = 0.16).

⚠️ But the exam could not have failed it. The holdout can only detect an alpha of 0.71% a month, almost twice value's entire historical premium. "Not rejected" here means "not testable yet".

📉 The bigger story: everything else became redundant too. Size and profitability lost alphas that were 2.6 and 4.8 standard errors from zero. Only the market kept its premium.

✂️ The best five-factor mix, frozen at its 2013 weights, kept about 30% of its Sharpe ratio: 0.35 against 1.12 promised in-sample, even after the usual bias correction (p = 0.007). An equal-weight mix lost the same, so the premia shrank; the optimizer was not the problem.

🕰️ In real time, value was NOT redundant until 1999: its t-stat was 1.7 to 2.3 every year from 1985 to 1998. Choosing the model each year by the best in-sample criterion always picked the biggest model with momentum, and added nothing.

🛡️ Factors were still useful crash insurance: in 2022 the six-factor mix made +18% while the market fell 19%.

Lesson: treat in-sample factor Sharpe ratios as upper bounds, and always report what your test could have detected.

Full paper, code, pre-registration and data in the first comment.

#QuantitativeFinance #AssetPricing #FactorInvesting #FamaFrench #Python #Research

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/17-five-factor-holdout
Article version: https://racuden28.github.io/30-quant-research-projects/17-five-factor-holdout/
