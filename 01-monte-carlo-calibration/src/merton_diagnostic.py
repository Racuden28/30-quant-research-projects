"""
Diagnostic: are Merton "jumps" rare events, or volatility clustering in disguise?

Re-estimates the Merton model on rolling 5-year windows every 80 trading days and reports how often
the MLE jump intensity sits at its cap (LAM_MAX), together with the implied share of daily variance
attributed to jumps.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from models import LAM_MAX, Merton

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "etf_daily_log_returns.csv", index_col=0, parse_dates=True)
rows = []
for a in df:
    r = df[a].dropna().values
    x0 = None
    for t in range(500, len(r), 80):
        p = Merton.fit(r[max(0, t - 1260):t], x0=x0)
        x0 = p["x"]
        jump_var = p["lam"] * (p["sJ"] ** 2 + p["muJ"] ** 2)
        rows.append({"asset": a, "t": t, "lam_per_year": p["lam"] * 252, "at_cap": p["lam"] > 0.98 * LAM_MAX,
                     "jump_var_share": jump_var / (jump_var + p["sigma"] ** 2)})
out = pd.DataFrame(rows)
summary = out.groupby("asset").agg(fits=("t", "size"), share_at_cap=("at_cap", "mean"),
                                   median_jumps_per_year=("lam_per_year", "median"),
                                   median_jump_var_share=("jump_var_share", "median"))
summary.to_csv(ROOT / "results" / "merton_diagnostic.csv", float_format="%.3f")
print(summary.round(3).to_string())
print("overall share at cap:", round(out.at_cap.mean(), 3))
