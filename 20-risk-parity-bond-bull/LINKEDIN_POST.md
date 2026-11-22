# LinkedIn post (draft, not published; waiting for Carlos's review)

**Was risk parity just a 40-year bond bull market?**

Risk parity (equal risk in stocks and bonds, levered up to 60/40's volatility) beat 60/40 from 1982 to 2020. Then 2022 happened.

I tested the two explanations on 97 years of monthly US data (1929–2026), with the bond rebuilt from yields and
validated against real Treasury funds:

📉 The sign of risk parity's edge followed bond yields in all four eras: +0.61 Sharpe when yields fell (1929–41),
−0.18 when they rose (1941–81), +0.13 when they fell (1981–2020), −0.60 when they rose (2020–26).

🧮 A closed-form hurdle: risk parity wins only if bonds' Sharpe ratio exceeds r* × stocks', with r* ≈ 0.6 at zero
correlation and ≈ 0.84 at +0.5. It called the winner in 4 of 4 eras.

🎁 Strip out the gain from falling yields and the 1981–2020 edge goes from +0.13 to −0.02.

🔍 Over the whole century, a small premium (≈ +0.15 to +0.17) does survive once the yield cycle is removed, but
50 bp of extra financing cost wipes it out.

🪤 A data trap worth knowing: Shiller's long-term yields before 1953 are interpolated from annual data. Using them
gave risk parity a Sharpe ratio of 3.6 in the 1930s.

Risk parity is not "no view". It is a view that bonds clear a hurdle, and that view depends on the yield cycle and the
stock–bond correlation.

Paper, code and every number: https://racuden28.github.io/30-quant-research-projects/ (project 20)

#QuantitativeFinance #RiskParity #AssetAllocation #FixedIncome #Python

*Image to attach: figures/fig1_edge_follows_yields.png*
