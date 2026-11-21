"""
Build the GitHub Pages site in docs/ from the project READMEs.

    python docs/build.py

* docs/index.html                      programme landing page (all 30 projects)
* docs/<project>/index.html            one article per published project, rendered from its README.md
* docs/<project>/figures/*.png         copied from the project folder

Requires: pip install markdown
"""
from __future__ import annotations

import re
import shutil
from html import escape
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/Racuden28/30-quant-research-projects"  # edit once

PROJECTS = [
    # (number, folder or None, title, one-line question)
    (1, "01-monte-carlo-calibration", "Which Monte Carlo Tells the Truth?",
     "Five return simulators, 112,626 out-of-sample forecasts, seven asset classes: whose probabilities come true in calm and in crisis?"),
    (2, None, "Returns & volatility dashboard", ""), (3, None, "Moving-average backtest", ""),
    (4, None, "Black–Scholes pricer", ""), (5, None, "Efficient frontier", ""),
    (6, None, "Value-at-Risk calculator", ""), (7, "07-kelly-estimation-risk", "The Kelly Fraction You Cannot Know",
     "With 20 years of data, can you even know the sign of the Kelly bet, and what should you size by instead?"),
    (8, None, "Correlation structure", ""), (9, None, "Beta & CAPM regression", ""),
    (10, "10-market-terminal", "What on a Market Screen Is Worth Looking At?",
     "Fourteen terminal widgets, seven ETFs and 122 years of S&P 500: which numbers on a market screen forecast anything out of sample?"), (11, None, "Pairs trading", ""),
    (12, None, "Momentum with transaction costs", ""), (13, None, "GARCH volatility forecasting", ""),
    (14, None, "Implied volatility surface", ""), (15, None, "Binomial tree pricing", ""),
    (16, None, "Delta-hedging simulation", ""), (17, None, "Fama–French factors", ""),
    (18, None, "Yield-curve PCA", ""), (19, None, "Regime detection with HMMs", ""),
    (20, None, "Risk parity", ""), (21, None, "Event-driven backtest engine", ""),
    (22, None, "Purged cross-validation", ""), (23, None, "Deflated Sharpe ratio", ""),
    (24, None, "PCA statistical arbitrage", ""), (25, None, "Kalman-filter hedge ratio", ""),
    (26, None, "Heston calibration", ""), (27, None, "Merton structural credit model", ""),
    (28, None, "Market-making agent", ""), (29, None, "Optimal execution", ""),
    (30, None, "Deep hedging", ""),
]
# extra folders inside a project that the site serves as-is (e.g. an interactive page)
STATIC_DIRS = ("terminal",)
TIERS = {range(1, 11): "Foundations", range(11, 21): "Intermediate", range(21, 31): "Advanced"}

CSS = """
:root{
  /* Layout: single reading column (~68ch) with full-bleed figures; landing = tiered index of 30 rows */
  --bg:#f6f7f9; --surface:#ffffff; --fg:#12151a; --fg2:#4a5160; --muted:#7a8291; --rule:#dde1e7;
  --accent:#1c5cab; --accent-soft:#e3edf9; --figure:#fcfcfb; --code:#eef1f5;
  --display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f1216; --surface:#161a20; --fg:#e6e8ec; --fg2:#b4bac5; --muted:#8a92a0; --rule:#2a303a;
  --accent:#6da7ec; --accent-soft:#16263b; --figure:#fcfcfb; --code:#1c2129; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f1216; --surface:#161a20; --fg:#e6e8ec; --fg2:#b4bac5; --muted:#8a92a0; --rule:#2a303a;
  --accent:#6da7ec; --accent-soft:#16263b; --figure:#fcfcfb; --code:#1c2129; color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:46rem;margin:0 auto;padding-block:2.5rem 4rem}
.wide{max-width:64rem}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
h1,h2,h3{font-family:var(--display);line-height:1.15;text-wrap:balance;letter-spacing:-.01em}
h1{font-size:clamp(2rem,5vw,3rem);margin:.2rem 0 .6rem;font-weight:700}
h2{font-size:1.55rem;margin:2.8rem 0 .8rem;padding-top:1.2rem;border-top:1px solid var(--rule);font-weight:700}
h3{font-size:1.15rem;margin:2rem 0 .5rem;font-weight:600}
p,li{color:var(--fg)}
.eyebrow{font:500 .75rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:1.15rem;color:var(--fg2)}
nav.top{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;
  font:500 .8rem var(--mono);color:var(--muted);padding-bottom:1rem;border-bottom:1px solid var(--rule)}
nav.top a{text-decoration:none}
img{max-width:100%;height:auto;display:block;margin:1.5rem auto;background:var(--figure);
  border:1px solid var(--rule);border-radius:6px;padding:.5rem}
.table-scroll{overflow-x:auto;margin:1.2rem 0}
table{border-collapse:collapse;font-size:.86rem;min-width:100%}
th,td{padding:.45rem .65rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--fg2);background:var(--surface)}
td{font-variant-numeric:tabular-nums}
code{font:.85em var(--mono);background:var(--code);padding:.1em .35em;border-radius:3px}
pre{background:var(--code);padding:1rem;border-radius:6px;overflow-x:auto;font-size:.82rem;line-height:1.5}
pre code{background:none;padding:0}
blockquote{margin:1rem 0;padding:.2rem 1rem;border-left:3px solid var(--accent);color:var(--fg2)}
hr{border:0;border-top:1px solid var(--rule);margin:2rem 0}
hr + h2{border-top:0;padding-top:0;margin-top:1rem}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);font-size:.85rem;color:var(--muted)}
/* landing */
.hero{padding-block:2rem 1rem}
.tier{margin-top:2.5rem}
.tier h2{border:0;padding:0;margin:0 0 .8rem;font-size:1rem;font-family:var(--mono);font-weight:500;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.rows{display:grid;gap:0;border-top:1px solid var(--rule)}
.row{display:grid;grid-template-columns:3.2rem minmax(0,1fr) auto;gap:1rem;align-items:baseline;
  padding:.85rem 0;border-bottom:1px solid var(--rule);text-decoration:none;color:inherit}
.row .n{font:600 1rem var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.row .t{font-weight:600;min-width:0}
.row .q{display:block;font-weight:400;color:var(--fg2);font-size:.92rem;margin-top:.2rem}
.row .s{font:500 .72rem var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
a.row:hover .t{color:var(--accent)}
a.row .s{color:var(--accent);background:var(--accent-soft);padding:.2rem .45rem;border-radius:3px}
.principles{display:grid;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));gap:1rem 2rem;margin-top:1rem}
.principles div{font-size:.92rem;color:var(--fg2)}
.principles b{display:block;color:var(--fg);font-family:var(--display);font-size:1rem}
@media (max-width:480px){.row{grid-template-columns:2.4rem minmax(0,1fr)}.row .s{grid-column:2}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'
         '&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400'
         '&display=swap">')


def page(title: str, body: str, description: str, standalone: bool = True) -> str:
    head = (f"<title>{escape(title)}</title>\n"
            '<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<meta name="description" content="{escape(description)}">\n{FONTS}\n<style>{CSS}</style>\n')
    if not standalone:  # fragment form (no document skeleton)
        return head + body
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</head>\n<body>\n{body}\n</body>\n</html>\n")


def rewrite_links(src: str, folder: str) -> str:
    """Point relative Markdown links that the site does not serve to the file on GitHub.

    Served on the site: figures/, the STATIC_DIRS, and ../NN-slug/ project folders (other articles).
    Everything else relative (PREREGISTRATION.md, results/*.csv, src/...) goes to the repository.
    """
    def fix(m: re.Match) -> str:
        bang, text, target = m.group(1), m.group(2), m.group(3)
        if bang or re.match(r"^(https?:|mailto:|#|/)", target):
            return m.group(0)
        if target.startswith("figures/") or target.split("/")[0] in STATIC_DIRS:
            return m.group(0)
        if target in ("../", "./") or re.match(r"^\.\./\d\d-[^/]+/?$", target):  # site home or another article
            return m.group(0)
        path = target[3:] if target.startswith("../") else f"{folder}/{target}"
        kind = "tree" if path.endswith("/") else "blob"
        return f"[{text}]({REPO_URL}/{kind}/main/{path})"
    return re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", fix, src)


def render_article(folder: str, standalone: bool = True, home_href: str = "../") -> tuple[str, str]:
    src = (ROOT / folder / "README.md").read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", src, re.M).group(1).strip()
    src = re.sub(r"^# .+\n", "", src, count=1, flags=re.M)
    src = src.replace("](../README.md)", f"]({home_href})")
    src = rewrite_links(src, folder)
    html = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])
    html = re.sub(r"<table>", '<div class="table-scroll"><table>', html)
    html = html.replace("</table>", "</table></div>")
    html = re.sub(r'<img alt="([^"]*)" src="([^"]+)"', r'<img alt="\1" loading="lazy" src="\2"', html)
    num = int(folder[:2])
    body = (f'<div class="wrap">\n<nav class="top"><a href="{home_href}">← 30 Quant Research Projects</a>'
            f'<span>Project {num:02d} / 30</span></nav>\n'
            f'<p class="eyebrow" style="margin-top:2rem">Working paper · Project {num:02d}</p>\n'
            f"<h1>{escape(title)}</h1>\n{html}\n"
            f'<footer>Code, data and every number in this article: '
            f'<a href="{REPO_URL}/tree/main/{folder}">{folder} on GitHub</a>.</footer>\n</div>')
    desc = next(d for n, f, t, d in PROJECTS if f == folder)
    return page(title, body, desc, standalone), title


def render_index() -> str:
    tiers = []
    for rng, label in TIERS.items():
        rows = []
        for n, folder, title, q in PROJECTS:
            if n not in rng:
                continue
            if folder:
                rows.append(f'<a class="row" href="{folder}/"><span class="n">{n:02d}</span>'
                            f'<span class="t">{escape(title)}<span class="q">{escape(q)}</span></span>'
                            f'<span class="s">Read</span></a>')
            else:
                rows.append(f'<div class="row"><span class="n">{n:02d}</span><span class="t">{escape(title)}</span>'
                            f'<span class="s">Planned</span></div>')
        tiers.append(f'<section class="tier"><h2>{label} · {rng.start:02d}–{rng.stop - 1:02d}</h2>'
                     f'<div class="rows">{"".join(rows)}</div></section>')
    done = sum(1 for p in PROJECTS if p[1])
    body = f"""<div class="wrap wide">
<header class="hero">
<p class="eyebrow">Self-directed research programme · {done} of 30 published</p>
<h1>30 Quant Research Projects</h1>
<p class="lede">Thirty questions in quantitative finance, from Monte Carlo and portfolio construction to
stochastic volatility, statistical arbitrage, market making and deep hedging. Each one is answered out of
sample, against honest baselines, with reproducible code and a section on what does not work.</p>
<div class="principles">
<div><b>A question, not a tutorial</b>Every project states a falsifiable question and answers it with a number.</div>
<div><b>Out of sample only</b>Rolling or walk-forward estimation, no look-ahead, fixed seeds.</div>
<div><b>Proper statistics</b>Formal tests, effect sizes and the textbook baseline in every comparison.</div>
<div><b>One command to reproduce</b>Every table and figure rebuilds from the repository.</div>
</div>
</header>
{"".join(tiers)}
<footer>Source: <a href="{REPO_URL}">{REPO_URL.split("github.com/")[1]}</a>. Not investment advice.</footer>
</div>"""
    return page("30 Quant Research Projects", body,
                "A self-directed programme of 30 out-of-sample research projects in quantitative finance.")


def main():
    (DOCS / "index.html").write_text(render_index(), encoding="utf-8")
    for n, folder, *_ in PROJECTS:
        if not folder:
            continue
        out = DOCS / folder
        out.mkdir(parents=True, exist_ok=True)
        html, _ = render_article(folder)
        (out / "index.html").write_text(html, encoding="utf-8")
        if (out / "figures").exists():
            shutil.rmtree(out / "figures")
        shutil.copytree(ROOT / folder / "figures", out / "figures")
        for d in STATIC_DIRS:
            if (ROOT / folder / d).is_dir():
                if (out / d).exists():
                    shutil.rmtree(out / d)
                shutil.copytree(ROOT / folder / d, out / d)
    (DOCS / ".nojekyll").write_text("")
    print("site built in", DOCS)


if __name__ == "__main__":
    main()
