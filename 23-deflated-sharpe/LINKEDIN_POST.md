# LinkedIn post (draft, not published)

**"DSR > 0.95" is not a 5% test.**

The Deflated Sharpe Ratio is the standard correction for "I tried N strategies and kept the best one". Most of us read DSR > 0.95 as "significant after multiple testing". For project 23 of my 30 quant research projects, I checked whether it behaves like that.

Setup: a zoo of 1,316 trading rules on 10 equity indices (back to 1928) and 49 US industries (back to 1926). In 300 placebo worlds, every day's return is sign-flipped, which keeps the real volatility, fat tails and correlation but makes every rule's true Sharpe ratio exactly zero.

What I found:
→ Across 29,700 null searches, the textbook DSR never crossed 0.95. Its false-positive rate is about 0%, not 5%.
→ When I planted a rule with a true Sharpe ratio of 1.0, the DSR found it only 15–18% of the time.
→ Two causes. It measures distance to the *expected* best null Sharpe ratio in units of *one* strategy's standard error. And its noise estimate (the variance of the trials' Sharpe ratios) also counts every real strategy in the zoo, and its mirror image, as noise.
→ A one-line fix does better: p = 1 − PSRᴺ (a Šidák bound) with the raw trial count. Its false-positive rate is 2–3%, and it has twice the power.
→ On real data from 1940 to 2020, the textbook DSR flagged 0 of 99 in-sample winners. The 28 that the Šidák test flagged earned a 1.10 Sharpe ratio over the next five years, against 0.27 for the rest.

Then I ran the same test on my own portfolio. Industry momentum (#12) and the real-time factor selector (#17) survive. The S&P 500 200-day rule's alpha (#3), country-ETF pairs (#11) and risk parity's alpha over 60/40 (#20) do not.

Paper, code and data: [link to GitHub Pages]

#QuantitativeFinance #Backtesting #MultipleTesting #SharpeRatio #Research

*(Image: figures/fig1_dsr_is_not_a_p_value.png)*
