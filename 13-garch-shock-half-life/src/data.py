"""Load the 10-index daily panel (shared with project 03) as percent log returns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "index_closes.csv"

INDICES = ["GSPC", "IXIC", "GSPTSE", "FTSE", "GDAXI", "FCHI", "N225", "HSI", "AXJO", "BVSP"]
LABEL = {"GSPC": "S&P 500", "IXIC": "Nasdaq Composite", "GSPTSE": "S&P/TSX", "FTSE": "FTSE 100",
         "GDAXI": "DAX", "FCHI": "CAC 40", "N225": "Nikkei 225", "HSI": "Hang Seng",
         "AXJO": "S&P/ASX 200", "BVSP": "Ibovespa"}
# Ibovespa before 1995 is the hyperinflation era (same cut as project 03).
START = {"BVSP": "1995-01-01"}


def load_returns(path: Path = DATA) -> dict[str, pd.Series]:
    """Daily close-to-close log returns in percent, one Series per index.

    Price indices (no dividends), which does not matter for variance. Days with a missing close
    are dropped, so a return can span a holiday. That is the standard daily-close convention.
    """
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    out = {}
    for c in INDICES:
        s = px[c].dropna()
        s = s[s > 0]
        if c in START:
            s = s[s.index >= START[c]]
        r = 100 * np.log(s).diff().dropna()
        out[c] = r.astype(float)
    return out


if __name__ == "__main__":
    for k, r in load_returns().items():
        print(f"{k:7s} {r.index[0].date()} {r.index[-1].date()} n={len(r):6d} "
              f"ann.vol={r.std() * np.sqrt(252):5.1f}%")
