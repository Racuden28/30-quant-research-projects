"""Every design choice of project 20 in one place (fixed in PREREGISTRATION.md unless marked 'robustness')."""
import pandas as pd

SEED = 20
WINDOW = 36            # months of trailing data for the covariance
TC = 0.0005            # 5 bp per unit of turnover
SPREAD = 0.0025        # financing spread over T-bills on leverage above 1, per year
N_BOOT = 5000
BLOCK = 12             # mean block length (months) of the stationary bootstrap
BENCH_W = (0.6, 0.4)   # 60/40

# Eras are cut at the extremes of the 10-year yield (see data/monthly_panel.csv, column y10):
# trough 1.98% in 1941-07 (Fed long-term government yield, FRED LTGOVTBD), peak 15.84% in 1981-09 and trough 0.54% in 2020-07 (month-end ^TNX).
# E1 starts in 1926-07 but the first backtest month is 1929-07 (36-month warm-up), hence its label.
ERAS = {
    "E1 1929-1941 (yields fall)": ("1926-07", "1941-07"),
    "E2 1941-1981 (bond bear)": ("1941-08", "1981-09"),
    "E3 1981-2020 (bond bull)": ("1981-10", "2020-07"),
    "E4 2020-2026 (bond bear)": ("2020-08", "2026-09"),
}
ERA_SHORT = {k: k.split(" (")[0] for k in ERAS}
FULL = ("1926-07", "2026-09")

STRESS = {
    "1929-32 crash": ("1929-09", "1932-06"),
    "1937-38 recession": ("1937-03", "1938-03"),
    "1973-74 stagflation": ("1973-01", "1974-09"),
    "1979-81 Volcker shock": ("1979-10", "1981-09"),
    "1987 crash": ("1987-09", "1987-11"),
    "2000-02 dot-com": ("2000-09", "2002-09"),
    "2008 GFC": ("2007-11", "2009-02"),
    "2020 Covid": ("2020-01", "2020-03"),
    "2022 inflation": ("2022-01", "2022-10"),
}


def period(s: str) -> pd.Period:
    return pd.Period(s, "M")
