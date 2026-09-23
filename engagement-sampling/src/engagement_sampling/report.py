"""The study report: one self-contained HTML file."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts

FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500;600&'
         'family=IBM+Plex+Sans:wght@400;500;600;700&'
         'family=IBM+Plex+Serif:wght@500;600&display=swap">')

CSS = """
:root{color-scheme:light;
 --bg:#f3f6f9;--card:#fff;--figure:#fcfcfb;--sunken:#eef2f7;
 --ink:#10151c;--ink2:#55606e;--ink3:#7b8797;--line:#dde4ec;--line2:#c8d3df;
 --accent:#1f6fd0;--accent-soft:#e9f1fc;--accent-ink:#1a5aa8;
 --warn:#c04a33;--warn-soft:#fbeeea;--good:#1b7f4f;--good-soft:#e6f4ec;
 --serif:"IBM Plex Serif",Georgia,serif;
 --sans:"IBM Plex Sans",ui-sans-serif,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
 --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --bg:#0d1117;--card:#161c24;--figure:#fcfcfb;--sunken:#11171e;
 --ink:#eef2f7;--ink2:#9aa6b4;--ink3:#78848f;--line:#232c37;--line2:#33404e;
 --accent:#4d97ec;--accent-soft:#16273d;--accent-ink:#8fbef4;
 --warn:#e2765c;--warn-soft:#2b1a16;--good:#5cc48c;--good-soft:#12271c;}}
:root[data-theme="dark"]{color-scheme:dark;
 --bg:#0d1117;--card:#161c24;--figure:#fcfcfb;--sunken:#11171e;
 --ink:#eef2f7;--ink2:#9aa6b4;--ink3:#78848f;--line:#232c37;--line2:#33404e;
 --accent:#4d97ec;--accent-soft:#16273d;--accent-ink:#8fbef4;
 --warn:#e2765c;--warn-soft:#2b1a16;--good:#5cc48c;--good-soft:#12271c;}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
 font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding-inline:20px;padding-block:44px 90px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--ink3);margin:0 0 10px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,4.4vw,40px);line-height:1.12;
 letter-spacing:-.015em;margin:0;text-wrap:balance}
.lede{color:var(--ink2);margin:12px 0 0;max-width:66ch;font-size:16px}
.rule{height:1px;background:var(--line);margin:32px 0 0}
h2{font-family:var(--serif);font-weight:600;font-size:23px;margin:46px 0 4px;letter-spacing:-.01em}
h3{font-family:var(--mono);font-size:11.5px;font-weight:500;letter-spacing:.13em;
 text-transform:uppercase;color:var(--ink3);margin:28px 0 10px}
p{margin:10px 0;color:var(--ink2);max-width:76ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
 background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:18px 0}
.tile{background:var(--card);padding:15px 17px}
.tile .k{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
.tile .v{font-family:var(--mono);font-size:25px;font-weight:600;letter-spacing:-.02em;margin-top:5px;
 color:var(--ink);font-variant-numeric:tabular-nums}
.tile .s{font-size:12.5px;color:var(--ink2);margin-top:3px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card);margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:8px 13px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
td{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
th{font-family:var(--mono);font-weight:500;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
 color:var(--ink3);position:sticky;top:0;background:var(--card);z-index:1;border-bottom:1px solid var(--line2)}
td:first-child,th:first-child{text-align:left;font-family:var(--sans);font-size:13px;font-weight:500}
tbody tr:last-child td{border-bottom:none}
tr.total td{font-weight:600;border-top:1px solid var(--line2);background:var(--accent-soft);color:var(--accent-ink)}
.figure{background:var(--figure);border:1px solid var(--line);border-radius:8px;padding:10px;margin:16px 0}
.figure img{display:block;width:100%;height:auto;border-radius:3px}
.note{border-left:2px solid var(--accent);background:var(--accent-soft);padding:13px 17px;margin:20px 0;
 border-radius:0 6px 6px 0;color:var(--ink2);font-size:14px}
.note strong{color:var(--ink)}
.note.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.note.good{border-left-color:var(--good);background:var(--good-soft)}
code{font-family:var(--mono);font-size:.88em;background:var(--accent-soft);color:var(--accent-ink);
 padding:2px 5px;border-radius:4px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink3);
 font-size:13px;max-width:76ch}
@media(max-width:520px){.wrap{padding-block:30px 60px}th,td{padding:7px 10px}}
"""


def table(df: pd.DataFrame, index_name="", int_cols=None, pct_cols=None, round_cols=None) -> str:
    int_cols, pct_cols, round_cols = int_cols or [], pct_cols or [], round_cols or []
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for idx, row in df.iterrows():
        cls = ' class="total"' if str(idx) == "TOTAL" else ""
        cells = [f"<td>{html.escape(str(idx))}</td>"] if index_name else []
        for c in df.columns:
            v = row[c]
            if c in pct_cols and pd.notna(v):
                s = f"{float(v)*100:.1f}%"
            elif c in int_cols and pd.notna(v):
                s = f"{int(v):,}"
            elif c in round_cols and pd.notna(v):
                s = f"{float(v):,.3f}"
            elif isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v):
                s = charts.fmt(v)
            elif isinstance(v, (bool, np.bool_)):
                s = "yes" if v else ""
            else:
                s = html.escape("" if pd.isna(v) else str(v))
            cells.append(f"<td>{s}</td>")
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    idx_head = f"<th>{html.escape(index_name)}</th>" if index_name else ""
    return (f'<div class="scroll"><table><thead><tr>{idx_head}{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def build_report(result: dict, out_path: Path) -> Path:
    cfg = result["config"]
    frames, funnel, plan = result["frames"], result["funnel"], result["plan"]
    sizes, sample, summary = result["sizes"], result["sample"], result["summary"]
    total = cfg.engagement.total_name
    order = cfg.dataset_order
    fig_dir = out_path.parent / "05_charts"

    c = {
        "volume": charts.volume_by_period(result["counts"], order, fig_dir / "volume.png"),
        "target": charts.target_vs_cap(sizes, cfg.design.cap, fig_dir / "target.png"),
        "weights": charts.design_weights(sizes, fig_dir / "weights.png"),
        "quota": charts.quota_allocation(plan, order, fig_dir / "quota.png"),
        "dist": charts.distribution_panels(frames, total, order, fig_dir / "distribution.png"),
        "conc": charts.concentration_curves(frames, total, order, fig_dir / "concentration.png"),
        "contrib": charts.contribution_bars(result["contribution"], order, fig_dir / "contribution.png"),
    }
    if not result["correlations"].empty:
        c["corr"] = charts.correlation_bars(result["correlations"], order, fig_dir / "correlation.png")
    if not result["overlap"].empty and len(cfg.design.priority_measures) > 1:
        c["overlap"] = charts.overlap_chart(result["overlap"], order, fig_dir / "overlap.png")

    frame_n = sum(len(f) for f in frames.values())
    deff = result["design_effect"]
    deff_row = deff.iloc[0] if not deff.empty else None
    collapsed = int(funnel.get("duplicates_collapsed", pd.Series([0])).sum())

    tiles = "".join([
        f'<div class="tile"><div class="k">frame</div><div class="v">{frame_n:,}</div>'
        f'<div class="s">{collapsed:,} duplicates collapsed</div></div>',
        f'<div class="tile"><div class="k">drawn</div><div class="v">{len(sample):,}</div>'
        f'<div class="s">{len(sample)/max(frame_n,1)*100:.1f}% of the frame</div></div>',
        f'<div class="tile"><div class="k">effective n</div>'
        f'<div class="v">{deff_row["effective_n"]:,.0f}</div>'
        f'<div class="s">design effect {deff_row["design_effect"]:.2f}</div></div>'
        if deff_row is not None else "",
        f'<div class="tile"><div class="k">strata</div><div class="v">{len(plan):,}</div>'
        f'<div class="s">{plan["period"].nunique()} periods × {plan["dataset"].nunique()} datasets</div></div>',
    ])

    issues = result["plan_issues"] + result["sample_issues"]
    validation = (f'<div class="note good"><strong>Validation passed.</strong> Quotas sum to '
                  f'every period target, no stratum was over-drawn, every inclusion probability '
                  f'lies in (0,&nbsp;1], and no duplicate text survives in the sample.</div>'
                  if not issues else
                  '<div class="note warn"><strong>Validation raised '
                  f'{len(issues)} problem(s).</strong><br>' +
                  "<br>".join(html.escape(i) for i in issues[:12]) + '</div>')

    cover = result["coverage"]
    cover_note = ""
    if not cover.empty:
        us = cover["sample_units"].sum() / cover["frame_units"].sum() * 100
        ms = cover["sample_mass"].sum() / cover["frame_mass"].sum() * 100
        cover_note = (f'<div class="note"><strong>The sample holds {us:.1f}% of the units and '
                      f'{ms:.1f}% of the engagement.</strong> That gap is the design working as '
                      f'intended: the priority pool concentrates the budget on the tail, where '
                      f'the influential content is. It is also why the weights matter — an '
                      f'unweighted average over this sample describes the sample, not the corpus.</div>')

    filters = ", ".join(f"{f.name}" for f in cfg.frame.filters) or "none"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(cfg.name)}</title>{FONTS}<style>{CSS}</style></head>
<body><div class="wrap">

<p class="eyebrow">Sampling report</p>
<h1>{html.escape(cfg.name)}</h1>
<p class="lede">{len(frames)} datasets, {plan['period'].nunique()} periods.
Frame filters: {html.escape(filters)}. Target {cfg.design.rate*100:g}% per period
{'capped at ' + format(cfg.design.cap, ',') if cfg.design.cap else 'uncapped'},
allocated {cfg.design.allocation}ly, with a priority pool of the top
{cfg.design.top_pct*100:g}% on {', '.join(f'<code>{html.escape(m)}</code>' for m in cfg.design.priority_measures)}.</p>
<div class="tiles">{tiles}</div>
{validation}
<div class="rule"></div>

<h2>The frame</h2>
{table(funnel.set_index("dataset"), "dataset",
       int_cols=[c for c in funnel.columns if c.startswith(("raw","after","blank","unparsed","duplicates","frame"))])}
<div class="figure"><img src="data:image/png;base64,{c['volume']}" alt="frame size by period"></div>

<h2>Where the engagement is</h2>
<div class="figure"><img src="data:image/png;base64,{c['contrib']}" alt="platform contribution"></div>
<div class="figure"><img src="data:image/png;base64,{c['dist']}" alt="distribution by dataset"></div>
<div class="figure"><img src="data:image/png;base64,{c['conc']}" alt="concentration curves"></div>
<h3>Concentration of {html.escape(total)}</h3>
{table(result['concentration'].assign(p=lambda d:(d.p*100).map('top {:g}%'.format))
        .set_index('dataset')[['p','n_top','threshold','sum_top','share_of_total','mean_top']],
       'dataset', int_cols=['n_top'], pct_cols=['share_of_total'])}

{('<h2>Do the two engagement measures agree?</h2>'
  '<div class="figure"><img src="data:image/png;base64,' + c['corr'] + '" alt="correlation"></div>'
  + table(result['correlations'].set_index('dataset')[['n','pearson','spearman','both_zero_pct']],
          'dataset', int_cols=['n'], round_cols=['pearson','spearman'])) if 'corr' in c else ''}

{('<div class="figure"><img src="data:image/png;base64,' + c['overlap'] + '" alt="measure overlap"></div>'
  '<p>Where this sits near zero, the priority measures are selecting different units, so the '
  'priority pool is close to the sum of their sizes rather than their union.</p>') if 'overlap' in c else ''}

<h2>The plan</h2>
<div class="figure"><img src="data:image/png;base64,{c['target']}" alt="target sample size per period"></div>
<div class="figure"><img src="data:image/png;base64,{c['weights']}" alt="design weight per period"></div>
<p>A period whose uncapped target exceeds the cap is sampled at a lower rate than the rest,
so each of its units stands for more of the corpus. Those are the marked bars, and the
difference is carried through as a design weight on every unit drawn from them.</p>
{table(sizes.set_index("period")[["period_total","target_raw","target","capped","realized_rate","period_weight"]],
       "period", int_cols=["period_total","target"], pct_cols=["realized_rate"], round_cols=["period_weight"])}
<div class="figure"><img src="data:image/png;base64,{c['quota']}" alt="quota allocation"></div>

<h2>The sample</h2>
{cover_note}
{table(cover.set_index("dataset"), "dataset", int_cols=["frame_units","sample_units"],
       pct_cols=["unit_share","mass_share"]) if not cover.empty else ""}
<h3>Precision</h3>
{table(deff.set_index("scope"), "scope", int_cols=["n"], round_cols=["design_effect","cv_squared"])
 if not deff.empty else ""}
<h3>How each unit was selected</h3>
{table(sample.groupby(["dataset","selection_phase"]).size().unstack(fill_value=0)
       if not sample.empty else pd.DataFrame(), "dataset")}

<footer>Generated {datetime.now():%Y-%m-%d %H:%M}. Every table here is written to
<code>output/</code> as CSV; <code>03_sample/sample.csv.gz</code> carries
<code>inclusion_prob</code> and <code>design_weight</code> for every selected unit, and
<code>00_frame/config_used.json</code> records the exact configuration that produced this run.</footer>
</div></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
