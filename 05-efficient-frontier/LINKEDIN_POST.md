# LinkedIn post (project 05)

**Attach:** `figures/fig1_information_factorial.png` (main image). Optional second image:
`figures/fig2_value_of_covariance_information.png`. Put the GitHub link in the first comment, not in the post.

---

Everyone knows the efficient frontier fails out of sample. I wanted to know which input is to blame: the expected returns or the covariances?

Project 5 of 30 in my quant research series: "Where Does the Efficient Frontier Break?"

I gave a Markowitz optimiser perfect foresight on one input at a time. At every rebalance it got 5 kinds of expected returns (including the true future average) crossed with 3 kinds of covariance (including the true future covariance). I ran it walk-forward on 12 and 49 US industries (1949–2026) and 19 S&P 500 stocks, varied the data window to stress the high-dimensional case, and compared everything against simulated stationary markets.

What I found:

🎯 The frontier breaks in the means. Perfect expected returns add +0.55 to +0.75 to the Sharpe ratio. Perfect covariances add between −0.02 and +0.09. Means explain 86–105% of the value of perfect information.

🔗 Covariance information only pays once the means are right. With 49 industries, a perfect covariance adds +0.27 Sharpe when the means are also perfect, and −0.08 with ordinary sample means. A sharper risk model lets the optimiser bet harder on whatever the means say, right or wrong.

📐 What matters is the number of assets, not the data-to-asset ratio. Shortening the window never made the covariance matter for 12 or 19 assets. Going to 49 assets did.

🧮 Ledoit–Wolf shrinkage barely matters for long-only portfolios. The no-short constraint already does most of its job.

📉 On 49 industries, a stationary world says Markowitz should beat 1/N by 0.10 Sharpe. In reality it loses by 0.05. Expected returns move, and a 10-year window can't keep up.

⚖️ After costs and multiple-testing correction, nothing beat a simple 1/N portfolio. Minimum variance came closest and lost 7–15 points less than 1/N in 1987, 2008 and 2022.

The lesson: in long-only portfolio construction, the expected-return view is the investment decision. If you can't defend one, use none.

Full paper, code and data (every number reproducible with one command) in the first comment.

#QuantitativeFinance #PortfolioOptimization #Markowitz #Python #Research
