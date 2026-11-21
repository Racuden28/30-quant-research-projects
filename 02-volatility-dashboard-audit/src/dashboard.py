"""Build the interactive returns & volatility dashboard (dashboard/index.html, self-contained except
for Plotly from a CDN). Run after forecast.py and evaluate.py.

    python src/dashboard.py

The dashboard is the practical output of the paper: for each ETF it shows the textbook number
(21-day close-to-close vol, sqrt-of-time) next to the specification that won out of sample, the
realized vol that followed, a volatility cone, return distribution statistics and the variance-ratio
term structure, so a reader can see exactly where the textbook number goes wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import ASSETS, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "dashboard"
OUT.mkdir(exist_ok=True)
NAMES = {"SPY": "S&P 500", "XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials",
         "XLK": "Technology", "XLP": "Consumer Staples", "XLU": "Utilities", "XLV": "Health Care",
         "XLY": "Consumer Discretionary"}
SHOW = {"C2C|R21|sqrt": "Textbook: 21d close-to-close", "GK|E10|sqrt": "Range (GK), EWMA 10d",
        "HAR-GK": "HAR on range", "GJR-GARCH-t": "GJR-GARCH-t"}


def main():
    _, data, _ = load()
    z = np.load(RES / "forecasts_h22.npz", allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    models = list(z["models"])
    vr = pd.read_csv(RES / "variance_ratios.csv")
    payload = {"assets": [], "names": NAMES, "series": SHOW}
    for a in ASSETS:
        df = data[a]
        F = pd.DataFrame(z[f"F_{a}"][:, [models.index(m) for m in SHOW]], index=dates, columns=list(SHOW))
        rv = pd.Series(z[f"RV_{a}"], index=dates)
        ann = lambda x: np.sqrt(x * 252 / 22) * 100  # noqa: E731
        wk = df.index[::5]
        price = np.exp(df.r.cumsum())
        block = {
            "ticker": a,
            "dates": [d.strftime("%Y-%m-%d") for d in wk],
            "price": np.round(price.loc[wk].to_numpy() * 100, 2).tolist(),
            "realized": [None if np.isnan(v) else round(float(v), 2) for v in ann(rv.loc[wk]).to_numpy()],
            "fc": {m: [None if np.isnan(v) else round(float(v), 2) for v in ann(F[m].loc[wk]).to_numpy()]
                   for m in SHOW},
        }
        # volatility cone: percentiles of annualized close-to-close vol over rolling windows
        cone = {}
        for n in [5, 10, 21, 63, 126, 252]:
            v = (df.r.rolling(n).std() * np.sqrt(252) * 100).dropna()
            cone[n] = [round(float(np.percentile(v, p)), 2) for p in [5, 25, 50, 75, 95]] + [round(float(v.iloc[-1]), 2)]
        block["cone"] = cone
        r = df.r
        block["stats"] = {
            "ann_return": round(float(r.mean() * 252 * 100), 2),
            "ann_vol": round(float(r.std() * np.sqrt(252) * 100), 2),
            "skew": round(float(r.skew()), 2), "kurt": round(float(r.kurt()), 2),
            "worst_day": round(float(r.min() * 100), 2), "best_day": round(float(r.max() * 100), 2),
            "max_dd": round(float((price / price.cummax() - 1).min() * 100), 1),
            "overnight_share": round(float((df.o ** 2).sum() / (r ** 2).sum() * 100), 1),
        }
        v = vr[(vr.asset == a) & (vr.period == "1999-2026")]
        block["vr"] = {int(q): round(float(x), 3) for q, x in zip(v.q, v.vr)}
        payload["assets"].append(block)
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("dashboard written:", OUT / "index.html", f"{len(html) / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sector Volatility Dashboard</title>
<meta name="description" content="Returns and volatility dashboard for SPY and nine SPDR sector ETFs: the textbook 21-day vol next to the forecasts that won out of sample.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;600&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#f6f7f9;--surface:#fff;--fg:#12151a;--fg2:#4a5160;--muted:#7a8291;--rule:#dde1e7;--accent:#1c5cab;
--s1:#8a8984;--s2:#2a78d6;--s3:#eb6834;--s4:#1baf7a;--real:#12151a;
--display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1216;--surface:#161a20;--fg:#e6e8ec;--fg2:#b4bac5;--muted:#8a92a0;--rule:#2a303a;--accent:#6da7ec;--real:#e6e8ec;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1216;--surface:#161a20;--fg:#e6e8ec;--fg2:#b4bac5;--muted:#8a92a0;--rule:#2a303a;--accent:#6da7ec;--real:#e6e8ec;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 var(--body);padding:0 16px}
.wrap{max-width:72rem;margin:0 auto;padding:2rem 0 3rem}
h1{font:700 clamp(1.7rem,4vw,2.4rem)/1.1 var(--display);margin:.2rem 0 .4rem}
.eyebrow{font:500 .72rem var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{color:var(--fg2);max-width:48rem}
.bar{display:flex;flex-wrap:wrap;gap:.4rem;margin:1.4rem 0}
.bar button{font:500 .8rem var(--mono);border:1px solid var(--rule);background:var(--surface);color:var(--fg2);padding:.35rem .6rem;border-radius:4px;cursor:pointer}
.bar button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.6rem;margin:1rem 0}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.7rem .8rem}
.tile .k{font:500 .7rem var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font:600 1.35rem var(--display);font-variant-numeric:tabular-nums}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.8rem;margin:1rem 0}
.card h2{font:600 1.05rem var(--display);margin:.2rem 0 .1rem}.card p{margin:.1rem 0 .4rem;color:var(--fg2);font-size:.88rem}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,26rem),1fr));gap:1rem}
.plot{width:100%;height:360px}
footer{color:var(--muted);font-size:.82rem;margin-top:2rem;border-top:1px solid var(--rule);padding-top:.8rem}
a{color:var(--accent)}
</style></head><body><div class="wrap">
<p class="eyebrow">Project 02 · 30 Quant Research Projects</p>
<h1>Sector Volatility Dashboard</h1>
<p class="lede">The dashed grey line is the number most dashboards show: 21-day close-to-close volatility, annualized with √252.
The coloured lines are the forecasts that beat it out of sample in the accompanying paper. Black is the volatility that actually
happened over the next 22 trading days. Data are weekly samples from 1999 to November 2026.</p>
<div class="bar" id="assets" role="group" aria-label="Choose an ETF"></div>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Forecast vs what happened next</h2><p>Annualized volatility of the next 22 trading days (%), forecast at each date. Click the legend to add HAR and GJR-GARCH. Drag to zoom; double-click to reset.</p><div class="plot" id="p_fc" style="height:420px"></div></div>
<div class="grid2">
<div class="card"><h2>Volatility cone</h2><p>Percentiles of realized close-to-close vol by window length since 1999, with today's value.</p><div class="plot" id="p_cone"></div></div>
<div class="card"><h2>Does √t hold?</h2><p>Variance ratio VR(q): variance of q-day returns over q × daily variance. Below 1 means √t overstates long-horizon risk.</p><div class="plot" id="p_vr"></div></div>
</div>
<div class="card"><h2>Growth of 100</h2><p>Total return index (log scale).</p><div class="plot" id="p_px" style="height:300px"></div></div>
<footer>Source: Yahoo Finance daily OHLC, adjusted closes. Code and paper: project 02 of 30 Quant Research Projects. Not investment advice.</footer>
</div>
<script>
const D=__DATA__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colors=()=>[css('--s1'),css('--s2'),css('--s3'),css('--s4')];
function layout(extra){return Object.assign({margin:{l:48,r:12,t:10,b:36},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
 font:{family:'IBM Plex Sans, system-ui',size:12,color:css('--fg2')},xaxis:{gridcolor:css('--rule'),zeroline:false},
 yaxis:{gridcolor:css('--rule'),zeroline:false},legend:{orientation:'h',y:-0.18},hovermode:'x unified'},extra||{})}
const cfg={displaylogo:false,responsive:true,modeBarButtonsToRemove:['lasso2d','select2d']};
let cur=0;
function tiles(a){const s=a.stats;const last=a.fc[Object.keys(D.series)[0]].filter(v=>v!=null).slice(-1)[0];
 const best=a.fc['HAR-GK'].filter(v=>v!=null).slice(-1)[0];
 const T=[['Textbook vol now',last.toFixed(1)+'%'],['HAR-range forecast',best.toFixed(1)+'%'],['Ann. return',s.ann_return+'%'],['Ann. vol',s.ann_vol+'%'],
 ['Skew',s.skew],['Excess kurtosis',s.kurt],['Worst day',s.worst_day+'%'],['Max drawdown',s.max_dd+'%'],['Overnight share of variance',s.overnight_share+'%']];
 document.getElementById('tiles').innerHTML=T.map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('')}
function draw(){const a=D.assets[cur];tiles(a);const C=colors();
 const tr=Object.keys(D.series).map((m,i)=>({x:a.dates,y:a.fc[m],name:D.series[m],mode:'lines',visible:i>=2?'legendonly':true,line:{width:i?1.6:2,color:C[i],dash:i?'solid':'dash'}}));
 tr.push({x:a.dates,y:a.realized,name:'Realized next 22 days',mode:'lines',line:{width:1.4,color:css('--real')}});
 Plotly.react('p_fc',tr,layout({yaxis:{gridcolor:css('--rule'),ticksuffix:'%',type:'log'}}),cfg);
 const ns=Object.keys(a.cone);const P=i=>ns.map(n=>a.cone[n][i]);
 Plotly.react('p_cone',[
  {x:ns,y:P(4),name:'95th',mode:'lines',line:{color:C[1],dash:'dot',width:1}},
  {x:ns,y:P(3),name:'75th',mode:'lines',line:{color:C[1],width:1}},
  {x:ns,y:P(2),name:'median',mode:'lines',line:{color:C[1],width:2.2}},
  {x:ns,y:P(1),name:'25th',mode:'lines',line:{color:C[1],width:1}},
  {x:ns,y:P(0),name:'5th',mode:'lines',line:{color:C[1],dash:'dot',width:1}},
  {x:ns,y:P(5),name:'now',mode:'markers+lines',marker:{size:8,color:C[2]},line:{color:C[2],width:1.5}}],
  layout({xaxis:{title:'window (trading days)',type:'category',gridcolor:css('--rule')},yaxis:{ticksuffix:'%',gridcolor:css('--rule')},hovermode:'x'}),cfg);
 const qs=Object.keys(a.vr);
 Plotly.react('p_vr',[{x:qs.map(q=>q+'d'),y:qs.map(q=>a.vr[q]),type:'bar',marker:{color:C[1]},hovertemplate:'VR=%{y:.2f}<extra></extra>'}],
  layout({yaxis:{range:[0.5,1.3],gridcolor:css('--rule')},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:css('--fg'),dash:'dash',width:1}}],hovermode:'x'}),cfg);
 Plotly.react('p_px',[{x:a.dates,y:a.price,mode:'lines',line:{color:C[1],width:1.6},name:a.ticker}],layout({yaxis:{type:'log',gridcolor:css('--rule')},showlegend:false}),cfg);}
const bar=document.getElementById('assets');
D.assets.forEach((a,i)=>{const b=document.createElement('button');b.textContent=a.ticker+' · '+D.names[a.ticker];
 b.setAttribute('aria-pressed',i===0);b.onclick=()=>{cur=i;[...bar.children].forEach((c,j)=>c.setAttribute('aria-pressed',j===i));draw()};bar.appendChild(b)});
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
</script></body></html>
"""

if __name__ == "__main__":
    main()
