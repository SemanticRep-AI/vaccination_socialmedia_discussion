#!/usr/bin/env python3
"""
Build docs/visualizations.html.

One chart for every analysis the reference notebook performs, plus the few the
hybrid design adds. Reads the CSVs in examples/reference_run/ so the page can
be rebuilt without re-running either pipeline.

    python scripts/make_visualizations.py
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "examples" / "reference_run"
OUT = ROOT / "docs" / "visualizations.html"
PNGS = ROOT / "docs" / "figures"

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT, WARN, GOOD = SERIES[0], "#c04a33", "#1b7f4f"
ORDER = ["pro_vaccine", "anti_vaccine", "politicians", "journalists"]


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=.8, alpha=.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="600", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def save(fig, name):
    PNGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNGS / f"{name}.png", dpi=140, bbox_inches="tight", facecolor=SURFACE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fmt(x):
    x = float(x)
    for d, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= d:
            return f"{x/d:.1f}{s}"
    return f"{x:,.0f}"


def xticks(ax, labels):
    step = max(1, len(labels) // 18)
    ax.set_xticks(np.arange(len(labels))[::step])
    ax.set_xticklabels(list(labels)[::step], rotation=45, ha="right")


FIGS: dict[str, str] = {}

# ---------------------------------------------------------------- 1. volume
counts = pd.read_csv(DATA / "all_monthly_counts.csv")
wide = counts.pivot_table(index="month", columns="dataset", values="count").fillna(0)
wide = wide[[d for d in ORDER if d in wide.columns]]
fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
x = np.arange(len(wide))
for i, col in enumerate(wide.columns):
    ax.plot(x, wide[col], lw=2.2, color=SERIES[i], marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.annotate(col, (x[-1], wide[col].iloc[-1]), xytext=(7, 0), textcoords="offset points",
                color=SERIES[i], fontsize=9.5, fontweight="600", va="center")
xticks(ax, wide.index)
ax.set_xlim(-.5, len(x) - .5 + len(x) * .09); ax.set_ylim(bottom=0)
style(ax, "Posts per month, English-language, before deduplication", "month", "posts")
fig.tight_layout(); FIGS["volume"] = save(fig, "01_volume")

# ------------------------------------------------- 2. engagement descriptives
desc = pd.read_csv(DATA / "measure_descriptives.csv")
d = desc[(desc.measure == "content_engagement") & (desc.scope == "nonzero")].set_index("dataset")
d = d.reindex([o for o in ORDER if o in d.index])
fig, ax = plt.subplots(figsize=(10.5, 5), facecolor=SURFACE)
pcts = ["p50", "p75", "p90", "p95", "p99"]
x = np.arange(len(d)); w = .15
for j, p in enumerate(pcts):
    ax.bar(x + (j - 2) * w, d[p], w, color=SERIES[j], label=p.upper())
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(d.index)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
style(ax, "content_engagement percentiles, posts with a non-zero value",
      "", "engagement (log scale)")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=5)
fig.tight_layout(); FIGS["percentiles"] = save(fig, "02_percentiles")

# ------------------------------------------------------- 3. zero inflation
z = desc[(desc.scope == "all") & (desc.measure.isin(["content_engagement", "source_engagement"]))]
piv = z.pivot_table(index="dataset", columns="measure", values="nonzero_pct")
piv = piv.reindex([o for o in ORDER if o in piv.index])
fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
x = np.arange(len(piv)); w = .36
ax.bar(x - w/2, piv["content_engagement"], w, color=SERIES[0], label="content_engagement")
ax.bar(x + w/2, piv["source_engagement"], w, color=SERIES[1], label="platform engagement column")
for i in range(len(piv)):
    for off, col in ((-w/2, "content_engagement"), (w/2, "source_engagement")):
        ax.text(x[i] + off, piv[col].iloc[i], f"{piv[col].iloc[i]:.0f}%", ha="center",
                va="bottom", fontsize=9, color=INK_SOFT, fontweight="600")
ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylim(0, 60)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
style(ax, "Share of posts with any engagement at all", "", "% non-zero")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
fig.tight_layout(); FIGS["zeroes"] = save(fig, "03_zero_inflation")

# ------------------------------------------------------------ 4. correlation
cb = pd.read_csv(DATA / "correlations_both.csv").set_index("dataset")
cb = cb.reindex([o for o in ORDER if o in cb.index])
fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
x = np.arange(len(cb)); w = .36
ax.bar(x - w/2, cb["pearson"], w, color=SERIES[0], label="Pearson (levels)")
ax.bar(x + w/2, cb["spearman"], w, color=SERIES[1], label="Spearman (ranks)")
for i in range(len(cb)):
    ax.text(x[i] - w/2, cb["pearson"].iloc[i], f"{cb['pearson'].iloc[i]:.3f}", ha="center",
            va="bottom", fontsize=9, color=INK_SOFT, fontweight="600")
    ax.text(x[i] + w/2, cb["spearman"].iloc[i], f"{cb['spearman'].iloc[i]:.3f}", ha="center",
            va="bottom", fontsize=9, color=INK_SOFT, fontweight="600")
ax.set_xticks(x); ax.set_xticklabels(cb.index); ax.set_ylim(0, .8)
style(ax, "Do the two engagement measures agree?", "", "correlation")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
fig.tight_layout(); FIGS["correlation"] = save(fig, "04_correlation")

# ----------------------------------------------------------- 5. contribution
cont = pd.read_csv(DATA / "platform_contribution.csv")
piv = cont.pivot_table(index="dataset", columns="measure", values="share").fillna(0)
piv = piv.reindex([o for o in ORDER if o in piv.index])
piv = piv[[c for c in piv.columns if piv[c].sum() > 0]]
fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE)
y = np.arange(len(piv)); left = np.zeros(len(piv))
for i, col in enumerate(piv.columns):
    v = piv[col].to_numpy() * 100
    ax.barh(y, v, left=left, color=SERIES[i], height=.6, edgecolor=SURFACE, linewidth=1.2,
            label=col.replace("_engagement", ""))
    for j, (val, l) in enumerate(zip(v, left)):
        if val > 8:
            ax.text(l + val/2, y[j], f"{val:.1f}%", ha="center", va="center",
                    color="white", fontsize=9.5, fontweight="700")
    left += v
ax.set_yticks(y); ax.set_yticklabels(piv.index); ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
style(ax, "Which platform the engagement actually comes from", "share of total engagement", "")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=6,
          loc="lower center", bbox_to_anchor=(.5, -.45))
fig.tight_layout(); FIGS["contribution"] = save(fig, "05_contribution")

# ------------------------------------------------------- 6. sample size + cap
mss = pd.read_csv(DATA / "monthly_sample_sizes.csv")
fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
x = np.arange(len(mss))
capped = mss["two_percent"] > 2000
ax.bar(x, mss["two_percent"], color=GRID, width=.72, label="10% of the month")
ax.bar(x, mss["monthly_sample_size"], color=[WARN if c else ACCENT for c in capped],
       width=.72, label="after the cap")
ax.axhline(2000, color=WARN, lw=1.5, ls="--")
ax.text(len(x) - .4, 2000, "  cap 2,000", color=WARN, fontsize=9, va="center", fontweight="600")
for i, c in enumerate(capped):
    if c:
        rate = mss["monthly_sample_size"].iloc[i] / mss["monthly_total"].iloc[i] * 100
        ax.annotate(f"{rate:.1f}%", (i, mss["monthly_sample_size"].iloc[i]), xytext=(0, 6),
                    textcoords="offset points", ha="center", color=WARN,
                    fontsize=8.5, fontweight="700")
xticks(ax, mss["month"])
style(ax, "Reference design: target sample size per month", "month", "posts to sample")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
fig.tight_layout(); FIGS["target"] = save(fig, "06_target")

# ------------------------------------------------------------- 7. quotas
q = pd.read_csv(DATA / "dataset_quotas.csv")
piv = q.pivot_table(index="month", columns="dataset", values="sample_quota").fillna(0)
piv = piv[[d for d in ORDER if d in piv.columns]]
fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
x = np.arange(len(piv)); bottom = np.zeros(len(piv))
for i, col in enumerate(piv.columns):
    v = piv[col].to_numpy(float)
    ax.bar(x, v, bottom=bottom, width=.72, color=SERIES[i], edgecolor=SURFACE,
           linewidth=1.4, label=col)
    bottom += v
xticks(ax, piv.index)
style(ax, "Reference design: quota by month and dataset", "month", "posts to sample")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=4)
fig.tight_layout(); FIGS["quota"] = save(fig, "07_quota")

# -------------------------------------------------------- 8. selection overlap
summ = pd.read_csv(DATA / "baseline_sampling_summary.csv")
summ["overlap_pct"] = summ["selected_by_both_n"] / summ["unique_high_engagement_n"] * 100
fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
periods = sorted(summ["month"].unique())
x = np.arange(len(periods)); sets = [d for d in ORDER if d in set(summ["dataset"])]
w = .8 / len(sets)
for i, name in enumerate(sets):
    sub = summ[summ.dataset == name].set_index("month").reindex(periods)
    ax.bar(x + i*w - .4 + w/2, sub["overlap_pct"].fillna(0), w, color=SERIES[i], label=name)
xticks(ax, periods)
ax.set_ylim(0, max(12, float(summ["overlap_pct"].max()) * 1.2))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
style(ax, "How often the two engagement measures pick the same post",
      "month", "% of the priority pool chosen by both")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=4)
fig.tight_layout(); FIGS["overlap"] = save(fig, "08_overlap")

# ------------------------------------------------------ 9. duplication by month
dl = pd.read_csv(DATA / "baseline_dedup_loss.csv")
fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
x = np.arange(len(dl))
ax.bar(x, dl["target"], color=GRID, width=.72, label="drawn")
ax.bar(x, dl["after_dedup"], color=ACCENT, width=.72, label="left after deduplication")
for i in range(len(dl)):
    ax.annotate(f"-{dl['pct_lost'].iloc[i]:.0f}%", (i, dl["target"].iloc[i]), xytext=(0, 5),
                textcoords="offset points", ha="center", color=WARN, fontsize=8.5,
                fontweight="700")
xticks(ax, dl["month"])
style(ax, "Deduplicating after the draw: what reaches the coders",
      "month", "posts")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
fig.tight_layout(); FIGS["dedup"] = save(fig, "09_dedup")

# ------------------------------------------- 10. the decile chart (the key one)
dec = pd.read_csv(DATA / "dedup_by_decile.csv")
fig, ax = plt.subplots(figsize=(10.5, 5), facecolor=SURFACE)
x = np.arange(len(dec))
colors = [GOOD if r < .25 else WARN for r in dec["drop_rate"]]
ax.bar(x, dec["drop_rate"] * 100, color=colors, width=.72)
for i, r in enumerate(dec["drop_rate"]):
    ax.text(x[i], r*100, f"{r*100:.0f}%", ha="center", va="bottom",
            fontsize=9.5, color=INK_SOFT, fontweight="700")
ax.set_xticks(x); ax.set_xticklabels([f"D{int(v)}" for v in dec["decile"]])
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
style(ax, "Which posts deduplication removes, by engagement decile",
      "content_engagement decile  (D1 lowest — D10 highest)", "% removed as duplicates")
fig.tight_layout(); FIGS["decile"] = save(fig, "10_decile")

# ------------------------------------------------------------ 11. frame effect
fun = pd.read_csv(DATA / "hybrid_funnel.csv").set_index("dataset")
fun = fun.reindex([o for o in ORDER if o in fun.index])
fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
y = np.arange(len(fun))
ax.barh(y, fun["after_language"], color=GRID, height=.6, label="English-language posts")
ax.barh(y, fun["frame_rows"], color=ACCENT, height=.6, label="unique texts (frame)")
for i in range(len(fun)):
    pct = fun["duplicates_collapsed"].iloc[i] / fun["after_language"].iloc[i] * 100
    ax.text(fun["after_language"].iloc[i], y[i], f"  {pct:.0f}% duplicates",
            va="center", fontsize=9, color=WARN, fontweight="600")
ax.set_yticks(y); ax.set_yticklabels(fun.index)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
style(ax, "Deduplicating the frame instead: how much of the corpus is reposts", "posts", "")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="lower right")
fig.tight_layout(); FIGS["frame"] = save(fig, "11_frame")

# --------------------------------------------------------- 12. design effect
dp = pd.read_csv(DATA / "hybrid_design_effect_by_phase.csv")
overall = pd.read_csv(DATA / "hybrid_design_effect.csv").iloc[0]
fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
labels = list(dp["selection_phase"]) + ["overall"]
vals = list(dp["effective_n"]) + [overall["effective_n"]]
raw = list(dp["n"]) + [overall["n"]]
y = np.arange(len(labels))
ax.barh(y, raw, color=GRID, height=.6, label="posts drawn")
ax.barh(y, vals, color=[SERIES[0], SERIES[2], SERIES[3]][:len(labels)], height=.6,
        label="effective sample size")
for i, (r, v) in enumerate(zip(raw, vals)):
    ax.text(r, y[i], f"  deff {r/v:.2f}", va="center", fontsize=9,
            color=INK_SOFT, fontweight="600")
ax.set_yticks(y); ax.set_yticklabels(labels)
style(ax, "What the unequal weights cost in precision", "posts", "")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="lower right")
fig.tight_layout(); FIGS["deff"] = save(fig, "12_design_effect")

# ------------------------------------------------------------- 13. coverage
cov = pd.read_csv(DATA / "hybrid_coverage.csv").set_index("dataset")
cov = cov.reindex([o for o in ORDER if o in cov.index])
fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
x = np.arange(len(cov)); w = .36
ax.bar(x - w/2, cov["unit_share"]*100, w, color=SERIES[0], label="share of posts")
ax.bar(x + w/2, cov["mass_share"]*100, w, color=SERIES[2], label="share of engagement")
for i in range(len(cov)):
    ax.text(x[i]-w/2, cov["unit_share"].iloc[i]*100, f"{cov['unit_share'].iloc[i]*100:.0f}%",
            ha="center", va="bottom", fontsize=9, color=INK_SOFT, fontweight="600")
    ax.text(x[i]+w/2, cov["mass_share"].iloc[i]*100, f"{cov['mass_share'].iloc[i]*100:.0f}%",
            ha="center", va="bottom", fontsize=9, color=INK_SOFT, fontweight="600")
ax.set_xticks(x); ax.set_xticklabels(cov.index); ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
style(ax, "What a 10% sample captures", "", "")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
fig.tight_layout(); FIGS["coverage"] = save(fig, "13_coverage")

print(f"built {len(FIGS)} figures")
json.dump(list(FIGS), open(PNGS / "index.json", "w"), indent=2)

# ===========================================================================
# The page
# ===========================================================================

FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&'
         'family=IBM+Plex+Serif:wght@500;600&display=swap">')

CSS = """
:root{color-scheme:light;
 --bg:#f3f6f9;--card:#fff;--figure:#fcfcfb;--ink:#10151c;--ink2:#55606e;--ink3:#7b8797;
 --line:#dde4ec;--line2:#c8d3df;--accent:#1f6fd0;--accent-soft:#e9f1fc;--accent-ink:#1a5aa8;
 --warn:#c04a33;--warn-soft:#fbeeea;--good:#1b7f4f;--good-soft:#e6f4ec;
 --serif:"IBM Plex Serif",Georgia,serif;
 --sans:"IBM Plex Sans",ui-sans-serif,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
 --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --bg:#0d1117;--card:#161c24;--figure:#fcfcfb;--ink:#eef2f7;--ink2:#9aa6b4;--ink3:#78848f;
 --line:#232c37;--line2:#33404e;--accent:#4d97ec;--accent-soft:#16273d;--accent-ink:#8fbef4;
 --warn:#e2765c;--warn-soft:#2b1a16;--good:#5cc48c;--good-soft:#12271c;}}
:root[data-theme="dark"]{color-scheme:dark;
 --bg:#0d1117;--card:#161c24;--figure:#fcfcfb;--ink:#eef2f7;--ink2:#9aa6b4;--ink3:#78848f;
 --line:#232c37;--line2:#33404e;--accent:#4d97ec;--accent-soft:#16273d;--accent-ink:#8fbef4;
 --warn:#e2765c;--warn-soft:#2b1a16;--good:#5cc48c;--good-soft:#12271c;}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;
 line-height:1.62;-webkit-font-smoothing:antialiased}
.shell{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:44px 96px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--ink3);margin:0 0 10px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(29px,4.6vw,42px);line-height:1.1;
 letter-spacing:-.015em;margin:0;text-wrap:balance}
.lede{color:var(--ink2);margin:13px 0 0;max-width:66ch;font-size:16.5px}
.rule{height:1px;background:var(--line);margin:34px 0 0}
.cols{display:grid;grid-template-columns:212px minmax(0,1fr);gap:44px;margin-top:34px;align-items:start}
@media(max-width:900px){.cols{grid-template-columns:minmax(0,1fr);gap:0}}
nav.toc{position:sticky;top:22px;max-height:calc(100vh - 44px);overflow-y:auto}
@media(max-width:900px){nav.toc{position:static;max-height:none;margin-bottom:28px;
 border:1px solid var(--line);border-radius:8px;background:var(--card);padding:14px 16px}}
nav.toc .t{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--ink3);margin:0 0 9px}
nav.toc ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px}
nav.toc a{display:flex;gap:9px;padding:5px 9px;border-radius:5px;text-decoration:none;
 color:var(--ink2);font-size:13.5px;border-left:2px solid transparent}
nav.toc a span{font-family:var(--mono);font-size:11px;color:var(--ink3);flex:none;min-width:14px}
nav.toc a:hover{background:var(--accent-soft);color:var(--accent-ink);border-left-color:var(--accent)}
section{scroll-margin-top:22px}
h2{font-family:var(--serif);font-weight:600;font-size:24px;margin:52px 0 6px;letter-spacing:-.01em;
 text-wrap:balance}
section:first-of-type h2{margin-top:0}
p{margin:11px 0;color:var(--ink2);max-width:76ch}
strong{color:var(--ink);font-weight:600}
.figure{background:var(--figure);border:1px solid var(--line);border-radius:8px;padding:10px;margin:18px 0}
.figure img{display:block;width:100%;height:auto;border-radius:3px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
 background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:20px 0}
.tile{background:var(--card);padding:15px 17px}
.tile .k{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3)}
.tile .v{font-family:var(--mono);font-size:25px;font-weight:600;letter-spacing:-.02em;margin-top:5px;
 color:var(--ink);font-variant-numeric:tabular-nums}
.tile .v.warn{color:var(--warn)}.tile .v.good{color:var(--good)}
.tile .s{font-size:12.5px;color:var(--ink2);margin-top:3px}
.note{border-left:2px solid var(--accent);background:var(--accent-soft);padding:13px 17px;margin:20px 0;
 border-radius:0 6px 6px 0;color:var(--ink2);font-size:14px}
.note strong{color:var(--ink)}
.note.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.note.good{border-left-color:var(--good);background:var(--good-soft)}
code{font-family:var(--mono);font-size:.87em;background:var(--accent-soft);color:var(--accent-ink);
 padding:2px 5px;border-radius:4px}
footer{margin-top:64px;padding-top:22px;border-top:1px solid var(--line);color:var(--ink3);
 font-size:13px;max-width:76ch}
@media print{nav.toc{display:none}.cols{display:block}.figure{break-inside:avoid}
 body{background:#fff}h2{break-after:avoid}}
@media(max-width:520px){.shell{padding-block:30px 64px}}
"""

hf = json.load(open(DATA / "headline_findings.json"))
fun = pd.read_csv(DATA / "hybrid_funnel.csv")
dl = pd.read_csv(DATA / "baseline_dedup_loss.csv")
ov = pd.read_csv(DATA / "baseline_sampling_summary.csv")
overall = pd.read_csv(DATA / "hybrid_design_effect.csv").iloc[0]

SECTIONS = [
    ("volume", "1", "The corpus over time",
     FIGS["volume"],
     "<p>Four search profiles, thirteen months, English-language posts only. Volume peaks in "
     "September 2025 and again in August 2026; both are the months where a fixed cap on the "
     "sample later changes the sampling rate. August 2025 is a single day of data and is "
     "effectively an artefact of the export window rather than a month.</p>"),

    ("engagement", "2", "How engagement is distributed",
     FIGS["percentiles"],
     "<p>Engagement is not merely skewed, it is skewed across orders of magnitude: the note the "
     "axis is logarithmic. Between the median and the 99th percentile of the non-zero posts "
     "there are three to four factors of ten. Any summary that quotes a mean of this variable "
     "is quoting the top fraction of a percent of the posts.</p>"),

    ("zeroes", "3", "Most posts have no engagement at all",
     FIGS["zeroes"],
     "<p>This is the fact that drives most of what follows. The platform's own engagement column "
     "is non-zero for only 6&ndash;14% of posts, and the derived total for 18&ndash;42%. A rule "
     "that selects the &lsquo;top 5% by engagement&rsquo; is therefore often choosing among "
     "hundreds of posts tied on the same value, and how those ties are broken decides who gets "
     "coded.</p>"),

    ("correlation", "4", "The two engagement measures disagree",
     FIGS["correlation"],
     "<p>The derived total and the platform's own engagement column are close to unrelated. "
     "Pearson correlation runs 0.02&ndash;0.16; the rank correlation, which is the form that "
     "matters because the design selects on order rather than magnitude, is higher but still "
     "far from agreement. Two measures this different are not redundant &mdash; which is a good "
     "reason to use both, and a reason not to expect the same posts from each.</p>"),

    ("contribution", "5", "The corpus is one platform",
     FIGS["contribution"],
     "<p>Facebook, YouTube, Pinterest and LinkedIn contribute nothing: every source metric behind "
     "them is zero on every row. Instagram contributes a rounding error. In practice "
     "<code>content_engagement</code> is <code>x_engagement</code>, and the multi-platform "
     "formula is insurance for a future export rather than a description of this one.</p>"),

    ("target", "6", "Target sample size, and where the cap binds",
     FIGS["target"],
     "<p>Ten per cent of each month, capped at 2,000. Two months exceed the cap and are "
     "therefore sampled at 9.1% and 7.6% instead of 10%. A post drawn from those months stands "
     "for more of the corpus than one drawn from any other month, which is a design weight, and "
     "unweighted totals pooled across months are biased against them.</p>"),

    ("quota", "7", "How each month's budget is split",
     FIGS["quota"],
     "<p>Each month's target is divided across the four datasets in proportion to their share of "
     "that month's posts, with the fractional remainders allocated largest-first so the quotas "
     "sum exactly to the target. This part of the reference implementation is textbook and "
     "carried through unchanged.</p>"),

    ("overlap", "8", "The two measures rarely pick the same post",
     FIGS["overlap"],
     "<p>Across all 51 strata, only <strong>2.6%</strong> of the priority pool is chosen by both "
     "measures, and in 14 strata the overlap is exactly zero. The selection rule ranks "
     "&lsquo;chosen by both&rsquo; ahead of &lsquo;chosen by one&rsquo;, but there is almost "
     "never anything in the first group &mdash; so in practice the rule reduces to a random draw "
     "from the union.</p>"),

    ("dedup", "9", "Deduplicating after the draw",
     FIGS["dedup"],
     "<p>Roughly 43% of every month's sample is removed as duplicate text after it has been "
     "drawn. A month targeted at 2,000 posts delivers 1,111. The shortfall is not a rounding "
     "issue; it is larger than the sample of most published content analyses.</p>"),

    ("decile", "10", "What deduplication removes",
     FIGS["decile"],
     "<p>This is the chart that matters most. Duplicates are reposts, and a repost carries the "
     "original's impression count, so duplicate groups sit almost entirely in the upper half of "
     "the engagement distribution. Deduplicating after the draw removes 75&ndash;88% of the top "
     "five deciles and almost nothing from the bottom four. Measured on the drawn sample it "
     "strips out <strong>82.9% of the engagement mass</strong> &mdash; precisely the content the "
     "design exists to capture.</p>"),

    ("frame", "11", "Deduplicating the frame instead",
     FIGS["frame"],
     "<p>Collapsing duplicates before the draw rather than after turns the same fact into a "
     "property of the population instead of a loss in the sample. Between 26% and 60% of each "
     "dataset is reposts. What remains is a frame of unique texts in which every quota can "
     "actually be met, every unit has a known probability of selection, and repost volume "
     "survives as a <code>duplicate_count</code> column rather than being discarded.</p>"),

    ("deff", "12", "What the design costs in precision",
     FIGS["deff"],
     "<p>Mixing a near-census of the tail with a sparse random sample of everything else makes "
     "the weights unequal, and unequal weights cost precision. Reserving a fifth of each "
     "stratum's quota for the remainder pool brings the pooled design effect from 39.4 to "
     "<strong>4.05</strong>, raising the effective sample size from about 200 to "
     "<strong>2,008</strong>. Each phase on its own is efficient; it is the mixture that has a "
     "cost, and it is the price of covering the tail.</p>"),

    ("coverage", "13", "What a 10% sample actually captures",
     FIGS["coverage"],
     "<p>Ten per cent of the posts, but 77&ndash;92% of all the engagement in the corpus. That "
     "gap is the design working as intended rather than a happy accident, and it is the reason "
     "the weights have to travel with the sample: an unweighted average over these posts "
     "describes the sample, not the corpus.</p>"),
]

toc = "".join(
    f'<li><a href="#{sid}"><span>{num}</span>{html.escape(title)}</a></li>'
    for sid, num, title, _, _ in SECTIONS)

body = "".join(
    f'<section id="{sid}"><h2>{html.escape(title)}</h2>{prose}'
    f'<div class="figure"><img src="data:image/png;base64,{fig}" alt="{html.escape(title)}"></div>'
    f'</section>'
    for sid, num, title, fig, prose in SECTIONS)

tiles = f"""
<div class="tiles">
  <div class="tile"><div class="k">posts analysed</div><div class="v">170,866</div>
    <div class="s">English-language, 4 datasets</div></div>
  <div class="tile"><div class="k">reposts</div><div class="v warn">52%</div>
    <div class="s">89,600 duplicate texts</div></div>
  <div class="tile"><div class="k">engagement lost</div><div class="v warn">82.9%</div>
    <div class="s">if deduplicating after the draw</div></div>
  <div class="tile"><div class="k">measure overlap</div><div class="v warn">2.6%</div>
    <div class="s">of the priority pool</div></div>
  <div class="tile"><div class="k">effective n</div><div class="v good">2,008</div>
    <div class="s">after reserving a remainder draw</div></div>
</div>"""

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sampling Design Visualisations</title>{FONTS}<style>{CSS}</style></head>
<body><div class="shell">

<p class="eyebrow">Vaccine discourse 2025&ndash;2026 &middot; sampling design</p>
<h1>Sampling Design Visualisations</h1>
<p class="lede">Thirteen charts covering every analysis in the reference pipeline, measured on the
four exports it was written for. Each one answers a question the design has to get right:
how the corpus is shaped, whether the engagement measures agree, how the sample is allocated,
and what the sample is left holding once duplicates are removed.</p>
{tiles}
<div class="rule"></div>

<div class="cols">
<nav class="toc" aria-label="Contents"><div class="t">Charts</div><ul>{toc}</ul></nav>
<main>{body}

<section id="reading">
<h2>Reading these together</h2>
<div class="note warn"><p><strong>Three of these charts describe the same problem.</strong>
Chart 3 shows that most posts have no engagement, so top-percentile rules select among large
groups of tied posts. Chart 9 shows a 43% shortfall against target once duplicates are removed.
Chart 10 shows that the removal is concentrated in the top half of the engagement distribution.
Together they mean the realised sample is both smaller than designed and systematically less
viral than designed &mdash; and neither effect is visible from the sample alone.</p></div>
<div class="note good"><p><strong>Two changes fix all three.</strong> Deduplicate the frame
before drawing, and reserve part of every stratum's quota for a random draw from outside the
priority pool. The first makes quotas attainable and keeps repost volume as data; the second
keeps every post reachable, which is what makes the weights in chart 12 valid.</p></div>
</section>
</main></div>

<footer>Generated {datetime.now():%Y-%m-%d} from <code>examples/reference_run/</code>.
Rebuild with <code>python scripts/make_visualizations.py</code>. Individual PNGs are written to
<code>docs/figures/</code>.</footer>
</div></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f}K)")
