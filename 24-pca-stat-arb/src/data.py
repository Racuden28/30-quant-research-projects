"""Load the two return panels used in the study.

Universe A ("crypto"): every Binance spot pair quoted in USDT, including the 217 pairs
that were later delisted (status BREAK), daily 00:00 UTC closes, 2017-08-17 .. 2026-11-22.
Universe B ("equity"): the 503 S&P 500 members of November 2026, adjusted closes
2000-01-03 .. 2026-11-20 (survivor-picked; see README, Limitations).

Both loaders return (ret, liq, meta): simple returns (dates x assets, NaN when not trading),
a liquidity panel used only to rank assets (trailing dollar volume), and a per-asset table.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# Bases that are not crypto risk: fiat-backed or algorithmic stablecoins, fiat, gold tokens.
STABLE = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USD1", "RLUSD", "U", "UST", "USTC",
    "EUR", "EURI", "AEUR", "GBP", "AUD", "PAXG", "XAUT", "SUSD", "USDS", "USDSB", "USDE", "XUSD",
    "BFUSD", "PYUSD", "USDSOLD",
}
LEVERAGED_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")
LEVERAGED_KEEP = {"JUP", "SYRUP"}          # real coins whose ticker happens to end in UP
TOKENISED_STOCK_FIRST_DAY = "2026-05-01"    # Binance tokenised equities ('...B') listed from 2026-06


def crypto_exclusion(sym: str, first_date: str) -> str | None:
    base = sym[:-4]
    if base in STABLE:
        return "stablecoin/fiat/gold"
    if base not in LEVERAGED_KEEP and base.endswith(LEVERAGED_SUFFIX):
        return "leveraged token"
    if base.endswith("B") and first_date >= TOKENISED_STOCK_FIRST_DAY and sym != "BUSDUSDT":
        return "tokenised stock"
    return None


def load_crypto(survivors_only: bool = False):
    d = pd.read_csv(os.path.join(DATA, "binance_usdt_daily.csv.gz"))
    first = d.groupby("sym").date.min()
    status = d.groupby("sym").status.first()
    excl = {s: crypto_exclusion(s, first[s]) for s in first.index}
    keep = [s for s, e in excl.items() if e is None]
    if survivors_only:
        keep = [s for s in keep if status[s] == "TRADING"]
    d = d[d.sym.isin(keep)]
    px = d.pivot(index="date", columns="sym", values="close").sort_index()
    qv = d.pivot(index="date", columns="sym", values="qv").sort_index()
    px.index = pd.to_datetime(px.index); qv.index = pd.to_datetime(qv.index)
    px = px.where(px > 0)
    ret = px.pct_change(fill_method=None)
    # A pair that is suspended and later resumes would otherwise show one huge return; the
    # gap return is set to NaN (the asset is out of the PCA window anyway).
    gap = px.isna().shift(1, fill_value=False) & px.notna()
    ret = ret.mask(gap)
    meta = pd.DataFrame({"status": status, "first": first, "exclusion": pd.Series(excl)})
    return ret, qv, meta


def load_equity():
    px = pd.read_csv(os.path.join(DATA, "sp500_adjclose.csv.gz"), index_col=0, parse_dates=True)
    dv = pd.read_csv(os.path.join(DATA, "sp500_dollar_volume_musd.csv.gz"), index_col=0, parse_dates=True)
    spy = px.pop("SPY"); dv = dv.drop(columns="SPY")
    px = px.where(px > 0)
    ret = px.pct_change(fill_method=None)
    meta = pd.read_csv(os.path.join(DATA, "sp500_constituents_2026-11.csv")).set_index("sym")
    meta["added"] = pd.to_datetime(meta["added"].astype(str).str[:10], errors="coerce")
    return ret, dv, meta, spy.pct_change(fill_method=None)
