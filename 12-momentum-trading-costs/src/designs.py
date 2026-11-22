"""The design grid. Every design is run on every signal and both universes; nothing is dropped."""
from __future__ import annotations

from engine import Design

VOL_TARGET = 0.15     # annualised, Barroso & Santa-Clara (2015) use 12%; scale-free for Sharpe

GRID: list[Design] = [Design("base", "Textbook (full monthly rebalance)")]
GRID += [Design(f"partial{int(t * 100)}", f"Partial rebalance θ = {t:g}", theta=t)
         for t in (0.75, 0.5, 0.33, 0.25, 0.15)]
GRID += [Design(f"band{int(b * 100)}", f"Band: exit beyond {int(b * 100)}%", q_exit=b)
         for b in (0.35, 0.4, 0.5, 0.6)]
GRID += [Design(f"hold{k}", f"Hold {k} months (overlapping)", hold=k) for k in (2, 3, 6, 12)]
GRID += [Design(f"smooth{k}", f"Rank smoothed over {k} months", smooth=k) for k in (2, 3, 6)]
GRID += [Design("vol", "Vol-scaled", vol_target=VOL_TARGET),
         Design("vol_band50", "Vol-scaled + band 50%", vol_target=VOL_TARGET, q_exit=0.5),
         Design("vol_partial33", "Vol-scaled + partial θ = 0.33", vol_target=VOL_TARGET, theta=0.33)]
LONG_ONLY = Design("long_only", "Long-only top leg", long_only=True)

BY_KEY = {d.key: d for d in GRID + [LONG_ONLY]}
FAMILY = {d.key: ("base" if d.key == "base" else "vol" if d.key.startswith("vol") else
                  "partial" if d.key.startswith("partial") else "band" if d.key.startswith("band") else
                  "hold" if d.key.startswith("hold") else "smooth") for d in GRID}
SLOW = [d.key for d in GRID if FAMILY[d.key] in ("partial", "band", "hold", "smooth")]

SIGNALS = {"12-1": (12, 1), "6-1": (6, 1), "1-0": (1, 0)}
PRIMARY_SIGNAL = "12-1"

ERAS = {"A": {"1928-2026": ("1928-01", "2026-09"), "1928-1962": ("1928-01", "1962-12"),
              "1963-1999": ("1963-01", "1999-12"), "2000-2026": ("2000-01", "2026-09")},
        "B": {"1998-2026": ("1998-01", "2026-10"), "1998-2008": ("1998-01", "2008-12"),
              "2009-2026": ("2009-01", "2026-10")}}

STRESS = {"1932 rebound": ("1932-06", "1932-09"), "1933 rebound": ("1933-04", "1933-06"),
          "1987 crash": ("1987-10", "1987-12"), "2000-01 tech unwind": ("2000-03", "2001-03"),
          "2008 crisis": ("2008-09", "2008-12"), "2009 momentum crash": ("2009-03", "2009-05"),
          "2020 Covid crash": ("2020-02", "2020-04"), "Nov 2020 vaccine rotation": ("2020-11", "2020-11"),
          "2022 rate shock": ("2022-01", "2022-12")}
