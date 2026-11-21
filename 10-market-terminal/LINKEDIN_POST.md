# LinkedIn post — Project 10

Every market terminal shows the same 14 numbers: today's change, RSI, Bollinger bands, the 200-day average, distance from the 52-week high, the drawdown, realised vol…

Most people read them as hints about where the price goes next. I tested whether they are.

Project 10 of my 30 Quant Research Projects is a pre-registered, out-of-sample audit of those 14 numbers. It covers 7 ETFs across asset classes from 2012 to 2026 and 122 years of S&P 500 data, with three targets: next month's return, volatility and drawdown.

What I found:

📉 None of the 14 forecasts the return. On the strict test, nothing beats the historical average after multiple-testing control, in either dataset. The trend numbers (200-day MA gap, golden cross, momentum) do worse than the average.

📈 Almost all of them forecast risk. The numbers we read as return signals carry the most volatility information beyond realised volatility itself. Bollinger %b adds +3.2% out-of-sample R² for next-month volatility, +6.4% at a one-week horizon. That is the leverage effect, seen through a terminal.

🔄 The same number means opposite things in different markets. A high Bollinger %b cuts next-month vol by 25% for US equities and raises it by 17% for gold.

🧩 The 14 numbers contain about four independent ideas, and combining them all does worse than the best one alone.

Then I built the terminal the audit implies. Every number carries a badge for what it has actually predicted. The return forecast reads "No signal", and the numbers that never predicted anything are greyed out.

Paper, code, every number as CSV, and the terminal: [link]

#QuantitativeFinance #Volatility #TechnicalAnalysis #Python #Research
