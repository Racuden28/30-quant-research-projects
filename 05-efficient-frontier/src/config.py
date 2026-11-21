"""Pre-registered design for project 05. Everything that could be tuned is fixed here, before results."""
from __future__ import annotations

SEED = 20261121
COST_BPS = 10.0            # one-way transaction cost, bps of traded notional
RESAMPLE_DRAWS = 100       # Michaud draws per rebalance
BOOT_REPS = 2000           # paired stationary-bootstrap replications
N_SIM_PATHS = 200          # stationary counterfactual paths

UNIVERSES = {
    # A: 12 industries, monthly excess returns. 10-year window, monthly rebalance, 3-year clairvoyant.
    "A": dict(name="12 US industries, monthly, 1949-2026", ppy=12, window=120, step=1, horizon=36,
              block=12, robust_windows=[60, 240], robust_horizons=[60, 120]),
    # B: 19 large-cap stocks, daily returns. 3-year window, rebalance every 21 days, 3-year clairvoyant.
    "B": dict(name="19 S&P 500 stocks, daily, 1990-2022", ppy=252, window=756, step=21, horizon=756,
              block=63, robust_windows=[252, 1260], robust_horizons=[252, 1260]),
    # C: 49 industries, monthly excess returns (the high-N arm). 10-year window; 10-year clairvoyance so
    # that the clairvoyant covariance of 49 assets is estimated from more observations than assets.
    "C": dict(name="49 US industries, monthly, 1969-2026", ppy=12, window=120, step=1, horizon=120,
              block=12, robust_windows=[60, 240], robust_horizons=[60]),
}

# N/T sweep (referee point: is "covariance does not matter" an artefact of small N/T?)
# Same code, only the estimation window W changes; N/T = assets / observations in the window.
SWEEP = {
    "A": [24, 36, 60, 120, 240],          # N=12:  N/T 0.50 .. 0.05
    "C": [60, 72, 96, 120, 180, 240],     # N=49:  N/T 0.82 .. 0.20
    "B": [42, 63, 126, 252, 756],         # N=19:  N/T 0.45 .. 0.03
}

MEAN_LEVELS = ["None", "Sample", "Bayes-Stein", "Full-sample", "Clairvoyant"]
COV_LEVELS = ["Sample", "Ledoit-Wolf", "Clairvoyant"]

# The horse race: strategies an investor could actually run (no look-ahead).
STRATEGIES = ["1/N", "Inverse vol", "Min-var (sample)", "Min-var (LW)", "Max-Sharpe (sample)",
              "Max-Sharpe (BS + LW)", "Max-Sharpe (resampled)"]

STRESS = {
    "A": {"1973-74 oil shock": ("1973-01", "1974-09"), "1987 crash": ("1987-09", "1987-11"),
          "2000-02 dot-com": ("2000-09", "2002-09"), "2008 GFC": ("2007-10", "2009-03"),
          "2020 COVID": ("2020-02", "2020-03"), "2022 inflation": ("2022-01", "2022-12")},
    "C": {"1987 crash": ("1987-09", "1987-11"),
          "2000-02 dot-com": ("2000-09", "2002-09"), "2008 GFC": ("2007-10", "2009-03"),
          "2020 COVID": ("2020-02", "2020-03"), "2022 inflation": ("2022-01", "2022-12")},
    "B": {"2008 GFC": ("2008-09-01", "2009-03-09"), "2020 COVID": ("2020-02-19", "2020-04-30"),
          "2022 inflation": ("2022-01-03", "2022-12-28")},
}
