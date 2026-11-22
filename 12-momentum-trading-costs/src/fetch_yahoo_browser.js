// Browser version of fetch_data.py (universe B). Open any page on https://query1.finance.yahoo.com,
// paste this into the developer console, and it downloads country_etf_daily.csv.gz.
// This is how the committed file was produced on 2026-11-22 (Yahoo blocks cloud machines).
(async () => {
  const T = ["EWA","EWC","EWD","EWG","EWH","EWI","EWJ","EWK","EWL","EWM","EWN","EWO","EWP","EWQ","EWS",
    "EWU","EWW","EWZ","EWT","EWY","EZA","FXI","ECH","EIS","TUR","THD","EPU","EIDO","EPHE","ENZL","EPOL",
    "EIRL","ENOR","EDEN","EFNL","INDA","GREK","ARGT","KSA","QAT","UAE","EWUS","SPY","VNM"];
  const g = x => x == null ? "" : (+x).toPrecision(7).replace(/\.?0+$/, "");
  const L = ["ticker,date,open,high,low,close,adjclose,volume"];
  for (const t of T) {
    const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${t}?period1=0&period2=1795000000&interval=1d&events=div,split`);
    const R = (await r.json()).chart.result[0], q = R.indicators.quote[0], a = R.indicators.adjclose[0].adjclose;
    R.timestamp.forEach((ts, i) => L.push([t, new Date(ts * 1000).toISOString().slice(0, 10), g(q.open[i]),
      g(q.high[i]), g(q.low[i]), g(q.close[i]), g(a[i]), q.volume[i] == null ? "" : q.volume[i]].join(",")));
  }
  const gz = new Blob([L.join("\n")]).stream().pipeThrough(new CompressionStream("gzip"));
  const url = URL.createObjectURL(await new Response(gz).blob());
  Object.assign(document.createElement("a"), { href: url, download: "country_etf_daily.csv.gz" }).click();
})();
