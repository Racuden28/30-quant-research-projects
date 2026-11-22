"""Build the monthly panel for project 20 from the raw files in data/raw/.

Output: data/monthly_panel.csv with, for each month-end from 1926-07 to 2026-09,
  rf           1-month T-bill return (Ken French)
  eq_x         US stock market excess return (Ken French Mkt-RF)
  bond_x       10-year constant-maturity Treasury excess return, rebuilt from yields (main series)
  bond_carry   the part of bond_x that comes from coupon minus T-bill
  bond_price   the part of bond_x that comes from the change in yield (the 'windfall' channel)
  bond_x_avg   the same bond, but rebuilt only from monthly-AVERAGE yields (data robustness)
  gold_x       gold excess return (from 1972-01)
  y10          10-year yield used at the end of the month (decimal)
  y10_src      'month-end' (^TNX) or 'monthly-average' (FRED LTGOVTBD)
  dy           S&P 500 dividend yield (decimal, multpl / Shiller)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MATURITY = 10.0


def par_bond_price(y, c, T):
    """Price per 1 of face of a bond with annual coupon rate c (paid semi-annually), T years left, yield y."""
    y = np.asarray(y, float)
    c = np.asarray(c, float)
    n = 2.0 * T
    disc = (1.0 + y / 2.0) ** (-n)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(y) > 1e-10, (1.0 - disc) / (y / 2.0), n)
    return (c / 2.0) * annuity + disc


def bond_returns_from_yields(y: pd.Series, maturity: float = MATURITY) -> pd.DataFrame:
    """Monthly total return of a constant-maturity par bond.

    At the end of month t-1 buy a new par bond (coupon = y[t-1], maturity `maturity`). At the end of month t
    value it at y[t] with maturity - 1/12 left, collect one month of coupon accrual, and roll into a new par bond.
    Returns 'income' (y[t-1]/12) and 'price' (price change) separately; total = income + price.
    """
    y0, y1 = y.shift(1), y
    price = par_bond_price(y1, y0, maturity - 1.0 / 12.0) - 1.0
    out = pd.DataFrame({"income": y0 / 12.0, "price": price}, index=y.index)
    out["total"] = out["income"] + out["price"]
    return out.iloc[1:]


def _month(idx) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(idx), freq="M")


def load_ff() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "F-F_Research_Data_Factors_CSV.zip") as z:
        txt = z.read(z.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = []
    for l in lines[start + 1:]:
        p = [s.strip() for s in l.split(",")]
        if len(p) < 5 or not p[0].isdigit() or len(p[0]) != 6:
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["ym", "mkt_rf", "smb", "hml", "rf"])
    df.index = pd.PeriodIndex([f"{s[:4]}-{s[4:]}" for s in df.ym], freq="M")
    return df[["mkt_rf", "rf"]].astype(float) / 100.0


def load_yields() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (month-end ^TNX, Fed long-term government yield monthly average, GS10 monthly average), decimals."""
    tnx = pd.read_csv(RAW / "yahoo_TNX_monthend.csv")
    tnx.index = _month(tnx.date)
    tnx = tnx["close"] / 100.0
    tnx = tnx[tnx.index <= pd.Period("2026-10", "M")]  # the last row is a partial month

    # Federal Reserve long-term government bond yield (bonds of 10+ years), monthly averages 1925-2000 (FRED
    # LTGOVTBD). Shiller's long rate is NOT used: before 1953 it is linearly interpolated from annual data.
    lt = pd.read_csv(RAW / "fred_LTGOVTBD_monthly_avg.csv")
    lt.index = _month(lt.Date)
    shl = lt["LTGOVTBD"] / 100.0

    gs = pd.read_csv(RAW / "gs10_monthly_avg.csv")
    gs.index = _month(gs.Date)
    gs = gs["Rate"] / 100.0
    return tnx, shl, gs


def load_gold() -> pd.Series:
    """Gold price: World Bank monthly average to 2000-08, COMEX front-month month-end afterwards.
    The splice month 2000-09 compares the COMEX month-end of 2000-09 with the COMEX month-end of 2000-08."""
    wb = pd.read_csv(RAW / "gold_worldbank_monthly.csv")
    wb.index = pd.PeriodIndex(wb.Date, freq="M")
    wb = wb["Price"]
    gc = pd.read_csv(RAW / "yahoo_GCF_monthend.csv")
    gc.index = _month(gc.date)
    gc = gc["adjclose"]
    gc = gc[gc.index <= pd.Period("2026-10", "M")]
    r_wb = np.log(wb).diff()
    r_gc = np.log(gc).diff()
    r = pd.concat([r_wb[r_wb.index <= pd.Period("2000-08", "M")], r_gc[r_gc.index >= pd.Period("2000-09", "M")]])
    r = np.expm1(r)
    return r[r.index >= pd.Period("1972-01", "M")]


def load_dividend_yield() -> pd.Series:
    d = pd.read_csv(RAW / "multpl_sp500_dividend_yield_monthly.csv")
    d.index = pd.PeriodIndex(d.month, freq="M")
    return d["div_yield_pct"] / 100.0


def build() -> pd.DataFrame:
    ff = load_ff()
    tnx, shl, gs = load_yields()

    # main yield series: month-end ^TNX from 1962-01, Fed long-term government bond yield (monthly average) before
    y_main = pd.concat([shl[shl.index < tnx.index[0]], tnx]).sort_index()
    src = pd.Series(np.where(y_main.index < tnx.index[0], "monthly-average", "month-end"), index=y_main.index)
    # averaged-only series: Fed long-term yield to 1953-03, GS10 averages afterwards
    y_avg = pd.concat([shl[shl.index < gs.index[0]], gs]).sort_index()

    b = bond_returns_from_yields(y_main)
    ba = bond_returns_from_yields(y_avg)

    idx = ff.index
    df = pd.DataFrame(index=idx)
    df["rf"] = ff["rf"]
    df["eq_x"] = ff["mkt_rf"]
    df["bond_x"] = b["total"].reindex(idx) - df["rf"]
    df["bond_carry"] = b["income"].reindex(idx) - df["rf"]
    df["bond_price"] = b["price"].reindex(idx)
    df["bond_x_avg"] = ba["total"].reindex(idx) - df["rf"]
    df["gold_x"] = load_gold().reindex(idx) - df["rf"]
    df["y10"] = y_main.reindex(idx)
    df["y10_src"] = src.reindex(idx)
    df["dy"] = load_dividend_yield().reindex(idx)
    df.index.name = "month"
    assert df[["rf", "eq_x", "bond_x"]].notna().all().all(), "gap in the core panel"
    return df


def validation() -> pd.DataFrame:
    """Compare the rebuilt 10-year bond with two real Treasury mutual funds (total return, Yahoo adjusted close)."""
    tnx, _, gs = load_yields()
    b = bond_returns_from_yields(tnx)["total"]
    ba = bond_returns_from_yields(gs)["total"]
    rows = []
    for name, dur in [("VFITX", "Vanguard Intermediate-Term Treasury (duration ~5)"),
                      ("VUSTX", "Vanguard Long-Term Treasury (duration ~15)")]:
        f = pd.read_csv(RAW / f"yahoo_{name}_monthend.csv")
        f.index = _month(f.date)
        f = f["adjclose"]
        f = f[f.index <= pd.Period("2026-10", "M")].pct_change().dropna()
        for lab, s in [("10y rebuilt from month-end yields", b), ("10y rebuilt from monthly-average yields", ba)]:
            j = pd.concat([f, s], axis=1, join="inner").dropna()
            j.columns = ["fund", "rebuilt"]
            beta = np.cov(j.rebuilt, j.fund)[0, 1] / j.rebuilt.var()
            rows.append(dict(fund=name, fund_desc=dur, series=lab, start=str(j.index[0]), end=str(j.index[-1]),
                             n=len(j), corr=j.corr().iloc[0, 1], beta_fund_on_rebuilt=beta,
                             ann_ret_fund=(1 + j.fund).prod() ** (12 / len(j)) - 1,
                             ann_ret_rebuilt=(1 + j.rebuilt).prod() ** (12 / len(j)) - 1,
                             vol_fund=j.fund.std() * np.sqrt(12), vol_rebuilt=j.rebuilt.std() * np.sqrt(12)))
    return pd.DataFrame(rows)


def smoothing_check() -> pd.DataFrame:
    """Over 1962-2026, how different are bond returns rebuilt from monthly-average yields (what we must use before
    1962) from those rebuilt from month-end yields? Reports vol, AR(1), correlation with stocks, and Sharpe."""
    ff = load_ff()
    tnx, _, gs = load_yields()
    out = []
    for lab, y in [("month-end (^TNX)", tnx), ("monthly-average (GS10)", gs)]:
        r = bond_returns_from_yields(y)["total"]
        j = pd.concat([r, ff], axis=1, join="inner").dropna()
        j = j[(j.index >= pd.Period("1962-02", "M")) & (j.index <= pd.Period("2026-09", "M"))]
        x = j.iloc[:, 0] - j.rf
        out.append(dict(series=lab, n=len(j), ann_excess=x.mean() * 12, ann_vol=x.std() * np.sqrt(12),
                        sharpe=x.mean() / x.std() * np.sqrt(12), ar1=x.autocorr(1),
                        corr_with_stocks=np.corrcoef(x, j.mkt_rf)[0, 1]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = build()
    df.to_csv(ROOT / "data" / "monthly_panel.csv", float_format="%.8g")
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    validation().to_csv(res / "data_bond_validation.csv", index=False, float_format="%.4f")
    smoothing_check().to_csv(res / "data_smoothing_check.csv", index=False, float_format="%.4f")
    print(df.describe().T)
    print(validation().to_string())
    print(smoothing_check().to_string())
