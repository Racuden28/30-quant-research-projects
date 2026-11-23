# LinkedIn post (project 15)

**Attach:** `figures/fig3_rq_century.png` (main image). Optional second image:
`figures/fig5_boundary_misspec.png`. Put the GitHub link in the first comment, not in the post, because LinkedIn
shows posts with external links to fewer people.

---

Every binomial-tree tutorial prices one American put, shows that it converges, and stops. I asked what the tree is actually for: when is an American option worth more than a European one, and what does the exercise decision need to know?

Project 15 of 30 in my quant research series: "When Is an American Option Worth More Than a European One?"

What I found:

🌲 Accurate to what? I raced 13 lattices on 1,300 random American options. Every tree is 14–57× less accurate, in relative terms, on the early-exercise premium than on the price, and the premium is the part that makes the option American. Leisen–Reimer with Richardson extrapolation wins, but only with 1/n weights, not the textbook 1/n². Reaching a 1% premium error is then about 500× faster than with CRR.

📜 98 years of the American premium on the S&P 500, with real T-bill rates, dividend yields and volatility since 1928. Which side should you exercise early? Calls in 1930–63, puts in 1964–2008, calls again in 2009–22, puts since 2023. It is all carry: interest on the strike against dividends forgone.

📏 Peaks: 7.9% of a 3-month call's price in 1942, 12.3% of a 3-month put's in 1981. On 73% of days a European formula would be off by more than 1% on one side or the other.

💥 In the 2008 and 2020 crashes the put premium was zero. Not because of fear: rates were zero, so exercising early earned nothing.

⚠️ The exercise rule needs to know volatility. Drawn at 80% of the true vol, the boundary forfeits 11–24% of the premium. At half the vol you exercise so early that the American option is worth less than its European twin.

🌊 With GARCH volatility, a boundary frozen at the start captures only 5–83% of the Longstaff–Schwartz premium. The leverage effect makes it exercise too early. The same tree boundary re-read daily at the current volatility captures 91–128%.

Also in the repo: numba lattices, exact policy evaluation on the tree, a risk-neutral GJR-GARCH, and 36 tests.

Full paper, code and data (one command to reproduce) in the first comment.

#QuantitativeFinance #Options #BinomialTree #AmericanOptions #NumericalMethods #Python #Research

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/15-american-early-exercise
Article version: https://racuden28.github.io/30-quant-research-projects/15-american-early-exercise/
