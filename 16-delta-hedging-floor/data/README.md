# Data

* `gspc_daily.csv`: S&P 500 price index (^GSPC) daily close, 1927-12-30 to 2026-11-17, from Yahoo Finance. It is the same series as project 03 (`../../03-moving-average-volatility/data/index_closes.csv`, column GSPC). Before 1952 the exchange also traded on Saturdays; every row is treated as one trading day.
* `vix_daily.csv`: CBOE VIX daily OHLC, 1990-01-02 to 2026-09-22, from https://github.com/datasets/finance-vix (data/vix-daily.csv), which mirrors CBOE's published history. Same file as project 04.
* `etf_daily_log_returns.csv`: daily total-return log returns (adjusted close) for SPY, EEM, TLT, GLD, USO, URA and HYG, 2006-04 to 2026-11-20, from Yahoo Finance, rounded to 4 decimals. Same file as projects 01 and 04.

No option prices are used. Options are priced with Black-Scholes at a quoted volatility (VIX or an EWMA forecast), so the study measures hedging error, not option-market mispricing. Market data belongs to its providers and is used here for research only.
