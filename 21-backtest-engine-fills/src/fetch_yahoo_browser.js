// Builds data/yahoo_ohlc.csv.gz. Yahoo Finance is blocked from cloud machines, so this runs in a browser
// tab opened on https://query1.finance.yahoo.com/ (same origin, no CORS). Paste into the console; it
// downloads yahoo_ohlc.csv.gz. The sample is cut at 2026-11-20 by src/data.py (END).
(async () => {
  const syms = ['^GSPC','^IXIC','^GSPTSE','^FTSE','^GDAXI','^FCHI','^N225','^HSI','^AXJO','^BVSP','QQQ','^IRX'];
  const f = v => (v == null || !isFinite(+v)) ? '' : (+(+v).toPrecision(8)).toString();
  const out = ['ticker,date,open,high,low,close,adjclose,volume'];
  for (const s of syms) {
    const u = `/v8/finance/chart/${encodeURIComponent(s)}?period1=-1500000000&period2=1800000000&interval=1d&events=div%2Csplit&includeAdjustedClose=true`;
    const c = (await (await fetch(u)).json()).chart.result[0];
    const q = c.indicators.quote[0], a = (c.indicators.adjclose || [{}])[0].adjclose || [], off = c.meta.gmtoffset || 0;
    c.timestamp.forEach((t, i) => {
      if (q.close[i] == null) return;
      const d = new Date((t + off) * 1000).toISOString().slice(0, 10);
      out.push([s, d, f(q.open[i]), f(q.high[i]), f(q.low[i]), f(q.close[i]), f(a[i]), q.volume[i] ?? ''].join(','));
    });
  }
  const gz = new Blob([out.join('\n')]).stream().pipeThrough(new CompressionStream('gzip'));
  const url = URL.createObjectURL(await new Response(gz).blob());
  Object.assign(document.createElement('a'), {href: url, download: 'yahoo_ohlc.csv.gz'}).click();
})();
