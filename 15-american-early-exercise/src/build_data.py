"""Build the daily S&P 500 input panel for Part II: spot, short rate, dividend yield, trailing volatility.

Sources (all in data/raw, all reachable from a cloud container or copied from earlier projects):
    gspc_irx_daily.csv          ^GSPC daily close 1927-12-30 .. 2026-11-17 and ^IRX (13-week T-bill) 1960 ..
                                (Yahoo, via project 03)
    F-F_Research_Data_Factors   Ken French monthly RF (1-month T-bill), 1926-07 .. 2026-09 (via project 09)
    sp500_monthly_shiller.csv   Shiller monthly S&P 500 price and 12-month dividends, to 2023-06 (via project 10)
    spy_total_return_log.csv    SPY daily total-return log returns 2006-04 .. 2026-11 (via project 01)

Rate. r is the continuously compounded 1-month T-bill rate from Ken French's RF, 12*ln(1+RF), stepped at
month starts; after 2026-09 the 13-week T-bill (^IRX) is used, converted from bank-discount yield.
Dividend yield. q = ln(1 + D12/P) with Shiller's trailing 12-month dividend D12 and the monthly average price
P, through 2023-06. From 2023-07 on, q is implied from SPY: trailing-252-day SPY total-return log return minus
the ^GSPC price log return over the same window, plus SPY's 0.0945% expense ratio. The overlap 2007-2023
validates this splice (results/dividend_splice_check.csv).
Volatility. sigma = trailing 63-trading-day close-to-close volatility of ^GSPC, annualised with sqrt(252).
All inputs are known at the close of the pricing date (no look-ahead): monthly values are applied from the
first trading day of the following month.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
SPY_ER = 0.000945
RES.mkdir(exist_ok=True)


def load_rf():
    z = zipfile.ZipFile(RAW / "F-F_Research_Data_Factors_CSV.zip")
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    rows = []
    for line in txt:
        parts = [p.strip() for p in line.split(",")]
        if "Annual" in line:
            break
        if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 6:
            rows.append((pd.Timestamp(parts[0][:4] + "-" + parts[0][4:] + "-01"), float(parts[4])))
    rf = pd.Series(dict(rows)).sort_index()
    return 12.0 * np.log1p(rf / 100.0)   # continuous annual rate, value for that calendar month


def main():
    px = pd.read_csv(RAW / "gspc_irx_daily.csv", parse_dates=["date"], index_col="date")
    S = px.GSPC.dropna()
    irx = px.IRX.dropna()
    irx_cont = -np.log(1.0 - irx / 100.0 * 91.0 / 360.0) / (91.0 / 365.0)

    # ---- rate ----
    rf = load_rf()   # RF of month m is the yield on a bill bought at the start of month m: known at start of m
    # map each day to its own month's RF (the bill rate quoted at the start of that month)
    month = S.index.to_period("M").to_timestamp()
    r = pd.Series(rf.reindex(month).to_numpy(), index=S.index)
    last_rf_month = rf.index[-1]
    after = S.index >= last_rf_month + pd.offsets.MonthBegin(1)
    r[after] = irx_cont.reindex(S.index).ffill()[after]
    rate_src = np.where(after, "IRX", "FF_RF")

    # ---- dividend yield ----
    sh = pd.read_csv(RAW / "sp500_monthly_shiller.csv", parse_dates=["Date"])
    sh = sh[sh.Dividend > 0].set_index("Date")
    q_sh = np.log1p(sh.Dividend / sh.SP500)                    # D12 / P for month m
    # Shiller's month-m value uses the month-m average price: usable from the first day of month m+1
    q_daily = pd.Series(q_sh.reindex(month - pd.offsets.MonthBegin(1)).to_numpy(), index=S.index)
    spy = pd.read_csv(RAW / "spy_total_return_log.csv", parse_dates=["date"], index_col="date").SPY
    spy_cum = spy.cumsum()
    lS = np.log(S)
    common = spy_cum.index.intersection(lS.index)
    q_spy = (spy_cum.reindex(common).diff(252) - lS.reindex(common).diff(252)) + SPY_ER
    q_spy = q_spy.reindex(S.index).ffill(limit=5)
    # splice check
    chk = pd.DataFrame(dict(q_shiller=q_daily, q_spy=q_spy)).dropna()
    chk = chk.resample("ME").last()
    chk.to_csv(RES / "dividend_splice_check.csv")
    print("splice check (monthly, overlap): corr %.3f, mean diff %.4f, MAE %.4f, n=%d" % (
        chk.corr().iloc[0, 1], (chk.q_spy - chk.q_shiller).mean(), (chk.q_spy - chk.q_shiller).abs().mean(), len(chk)))
    q = q_daily.copy()
    after_q = q.isna() & (S.index > pd.Timestamp("2023-06-30"))
    q[after_q] = q_spy[after_q]
    q_src = np.where(after_q, "SPY_implied", "Shiller_D12")

    # ---- volatility ----
    ret = np.log(S).diff()
    vol = ret.rolling(63).std() * np.sqrt(252)

    out = pd.DataFrame(dict(S=S, r=r, q=q, sigma=vol, rate_src=rate_src, q_src=q_src)).dropna(subset=["r", "q", "sigma"])
    out.to_csv(ROOT / "data" / "sp500_option_inputs.csv", float_format="%.6g")
    print(out.iloc[[0, -1]])
    print(out.groupby(out.index.year // 10 * 10)[["r", "q", "sigma"]].mean().round(4))


if __name__ == "__main__":
    main()
