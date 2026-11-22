# LinkedIn post (project 16)

**Attach:** `figures/fig1_rebalancing_floor.png` (main image). Optional second image:
`figures/fig4_which_delta.png`. Put the GitHub link in the first comment, not in the post, because LinkedIn
shows posts with external links to fewer people.

---

Every options textbook says the same thing about delta hedging: hedge more often and the error shrinks like 1/√N. Hedge daily instead of weekly and the variance should fall by 76%.

I tested that on 98 years of the S&P 500 and 7 ETFs from 2006 to 2026. That is 24,757 one-month options on the index alone, each sold at a volatility known on the day and hedged every 1 to 21 days.

Project 16 of 30 in my quant research series: "The Floor Under Delta Hedging."

What I found:

📉 Daily instead of weekly hedging removed 38% of the variance on the S&P 500, not 76%. Across equities, credit, oil, Treasuries, gold and uranium the range was 22% to 63%.

🧱 About 80% of the risk is a floor that more frequent hedging cannot remove. It comes from the volatility you did not know when you sold the option, and from jumps.

🧪 A GARCH world with fat tails, fitted to the S&P 500, reproduces the real numbers almost exactly (0.79 vs 0.81). Inside it, knowing the volatility the month actually delivered would halve the floor. A better forecast barely moves it.

💥 In Sep–Dec 2008, hedging daily instead of weekly bought nothing (0%). For EM equities it added risk.

🎯 Changing which delta you use beats trading more often. A minimum-variance delta that adds the leverage effect, estimated only from past data, cut the terminal hedging-error variance by 23% (Holm p = 0.003).

⚖️ At equal cost, rebalancing when delta drifts by 0.16 beats rebalancing every week by 1.2 vol points of risk. My own hypothesis that costs would make weekly hedging optimal was rejected: frequency buys little, but it also costs little.

Everything is out of sample, with block-bootstrap confidence intervals, multiple-testing corrections and a validated hedging engine that my later deep-hedging project builds on.

Full paper, code and data (one command to reproduce) in the first comment.

#QuantitativeFinance #Options #DeltaHedging #Volatility #RiskManagement #Python

---

**First comment:**
Paper and code: https://github.com/Racuden28/30-quant-research-projects/tree/main/16-delta-hedging-floor
Article version: https://racuden28.github.io/30-quant-research-projects/16-delta-hedging-floor/
