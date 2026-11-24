# Data

All files in `raw/` are daily closes downloaded on 2026-11-23 from Yahoo Finance's chart API
(`/v8/finance/chart/<symbol>`, interval 1d) through a browser, because Yahoo, CBOE and FRED are blocked from
the cloud machine this project was built on. Columns: date, open, high, low, close.

| file | symbol | what it is | first day | used |
|---|---|---|---|---|
| VIX9D.csv | ^VIX9D | CBOE 9-day implied variance index | 2011-01-03 | yes |
| VIX.csv | ^VIX | CBOE 30-day | 1990 | yes |
| VIX3M.csv | ^VIX3M | CBOE 93-day (formerly VXV) | 2006-07-17 | yes |
| VIX6M.csv | ^VIX6M | CBOE 183-day | 2008-01-02 | yes (sets the start date) |
| SKEW.csv | ^SKEW | CBOE SKEW, 30-day risk-neutral skewness | 1990 | yes |
| VVIX.csv | ^VVIX | CBOE VIX of VIX, 30-day | 2007-01-03 | yes |
| GSPC.csv | ^GSPC | S&P 500 price index | 1970 | physical-measure checks |
| VIX1D.csv, IRX/TNX/FVX.csv, VIX1Y.csv | | downloaded, not used (VIX1Y has one row on Yahoo) | | no |
| chain_SPX_20261123.csv.gz | ^SPX | every listed S&P 500 option, 53 expiries, bid/ask/last/OI, snapshot 2026-11-23 19:46 UTC | | experiment 6 |
| chain_SPY_20261123.csv.gz | SPY | same for SPY, 30 expiries, 20:01 UTC | | not used in the paper (American exercise) |

The 2026-11-23 index values are the last prints at about 20:00 UTC, one hour before the close; the
SKEW value for that day was not yet published when the data were pulled, so experiment 6 uses the SKEW
replicated from the chain itself (and the replication of VIX/VIX3M/VIX6M matches the published values to
within 0.1 vol points).
