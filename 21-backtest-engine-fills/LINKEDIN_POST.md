# LinkedIn post (draft, not posted)

**Image:** figures/fig1_fill_convention_by_era.png

---

Most backtests hide an assumption in one line of code: `position.shift(1) * returns`.

It says the trade fills at the same close the signal was computed from. I wanted to know how much three of my own earlier backtests depended on it.

So for project 21 of my 30 quant research projects I wrote one backtest engine that keeps the *decision* apart from the *fill*. I checked that it reproduces my earlier results (the 200-day moving-average study matches to four decimals in 10 of 10 indices) and then moved only the fill.

What I found:

📉 On price indices before 1990, filling one day later costs a 200-day trend rule about 19 bp per trade, nearly four times the cost I had assumed. Old index prices were stale, so the day after a crossing kept moving in the same direction, and no one could trade that.

📈 On the ETFs you can actually buy, from 1993 to 2005, the same delay *earned* about 33 bp per trade, in all 10 ETFs. Young ETFs traded thinly around their NAV, so the day after a crossing reversed.

🔁 Over the same dates, the index and its ETF disagree by 15 bp per trade in 9 of 10 pairs.

⚖️ Since 2006, both effects are close to zero.

🧭 One number predicts the sign: lag-1 autocorrelation × the size of the trade-day move.

What didn't change: a regime-switching (HMM) rule still trails buy & hold under every fill, because volatility, not direction, triggers its trades. Industry momentum loses only 1.9 bp per unit of turnover against a 78 bp break-even.

The engine also ships a causality audit. It truncates the data at random dates and checks that no past decision moves. It passed all three real strategies and caught both bugs I planted (a centred moving average and a smoothed HMM).

The lesson I'm taking away: the longer the history in a backtest, the more its fill convention matters, and the sign depends on whether you're testing an index you can't trade or an instrument you can.

Paper, code and data: [link to the project page]

#QuantitativeFinance #Backtesting #MarketMicrostructure #Python #Research
