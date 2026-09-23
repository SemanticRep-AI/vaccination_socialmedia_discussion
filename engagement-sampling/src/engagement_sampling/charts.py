"""
Chart builders.

Every chart returns a base64 PNG so a report can be a single file. Colours come
from one validated categorical order; series are labelled directly as well as
coloured, so nothing depends on distinguishing two hues.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = SERIES[0]
WARN = "#c04a33"


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="600", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def save(fig, path: Path | None = None) -> str:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x/div:.2f}{suf}".replace(".00", "")
    return f"{x:,.0f}" if abs(x) >= 10 else f"{x:,.2f}"


def _periods(ax, labels):
    step = max(1, len(labels) // 18)
    ax.set_xticks(np.arange(len(labels))[::step])
    ax.set_xticklabels(list(labels)[::step], rotation=45, ha="right")


# ---------------------------------------------------------------------------

def volume_by_period(counts: pd.DataFrame, order: list[str], path=None) -> str:
    """Units per period, one line per dataset."""
    wide = counts.pivot_table(index="period", columns="dataset",
                              values="count", aggfunc="sum").fillna(0)
    wide = wide[[d for d in order if d in wide.columns]]
    fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
    x = np.arange(len(wide.index))
    for i, col in enumerate(wide.columns):
        c = SERIES[i % len(SERIES)]
        ax.plot(x, wide[col].to_numpy(), lw=2.2, color=c, marker="o", ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round")
        ax.annotate(col, xy=(x[-1], wide[col].iloc[-1]), xytext=(7, 0),
                    textcoords="offset points", color=c, fontsize=9.5,
                    fontweight="600", va="center")
    _periods(ax, wide.index)
    ax.set_xlim(-0.5, len(x) - 0.5 + len(x) * 0.09)
    ax.set_ylim(bottom=0)
    style(ax, "Frame size by period", "period", "units in frame")
    fig.tight_layout()
    return save(fig, path)


def target_vs_cap(sizes: pd.DataFrame, cap: int | None, path=None) -> str:
    """Target sample size per period, and where the cap bites."""
    fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
    x = np.arange(len(sizes))
    capped = sizes["capped"].to_numpy(bool) if "capped" in sizes else np.zeros(len(sizes), bool)
    ax.bar(x, sizes["target_raw"], color=GRID, width=.72, label="uncapped target")
    ax.bar(x, sizes["target"], color=[WARN if c else ACCENT for c in capped],
           width=.72, label="actual target")
    if cap:
        ax.axhline(cap, color=WARN, lw=1.5, ls="--")
        ax.text(len(x) - .4, cap, f"  cap {cap:,}", color=WARN, fontsize=9,
                va="center", fontweight="600")
    for i, c in enumerate(capped):
        if c:
            ax.annotate(f"{sizes['realized_rate'].iloc[i]*100:.1f}%",
                        xy=(i, sizes["target"].iloc[i]), xytext=(0, 6),
                        textcoords="offset points", ha="center", color=WARN,
                        fontsize=8.5, fontweight="700")
    _periods(ax, sizes["period"])
    style(ax, "Target sample size per period", "period", "units")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
    fig.tight_layout()
    return save(fig, path)


def design_weights(sizes: pd.DataFrame, path=None) -> str:
    """The weight each period's units carry, which the cap makes unequal."""
    fig, ax = plt.subplots(figsize=(11.5, 4.4), facecolor=SURFACE)
    x = np.arange(len(sizes))
    capped = sizes["capped"].to_numpy(bool) if "capped" in sizes else np.zeros(len(sizes), bool)
    ax.bar(x, sizes["period_weight"], color=[WARN if c else ACCENT for c in capped], width=.72)
    base = sizes.loc[~capped, "period_weight"].mean() if (~capped).any() else np.nan
    if np.isfinite(base):
        ax.axhline(base, color=INK_SOFT, lw=1.3, ls="--")
        ax.text(0.2, base, f" uncapped periods ≈ {base:.1f}", color=INK_SOFT,
                fontsize=9, va="bottom")
    for i, c in enumerate(capped):
        if c:
            ax.annotate(f"{sizes['period_weight'].iloc[i]:.1f}",
                        xy=(i, sizes["period_weight"].iloc[i]), xytext=(0, 5),
                        textcoords="offset points", ha="center", color=WARN,
                        fontsize=8.5, fontweight="700")
    _periods(ax, sizes["period"])
    style(ax, "Design weight by period — each sampled unit stands for this many",
          "period", "1 / sampling rate")
    fig.tight_layout()
    return save(fig, path)


def quota_allocation(plan: pd.DataFrame, order: list[str], path=None) -> str:
    """How each period's target splits across datasets."""
    wide = plan.pivot_table(index="period", columns="dataset",
                            values="quota", aggfunc="sum").fillna(0)
    wide = wide[[d for d in order if d in wide.columns]]
    fig, ax = plt.subplots(figsize=(11.5, 5), facecolor=SURFACE)
    x = np.arange(len(wide.index))
    bottom = np.zeros(len(wide))
    for i, col in enumerate(wide.columns):
        v = wide[col].to_numpy(float)
        ax.bar(x, v, bottom=bottom, width=.72, color=SERIES[i % len(SERIES)],
               edgecolor=SURFACE, linewidth=1.4, label=col)
        bottom += v
    _periods(ax, wide.index)
    style(ax, "Sample quota by period and dataset", "period", "units to draw")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=len(wide.columns))
    fig.tight_layout()
    return save(fig, path)


def distribution_panels(frames: dict, measure: str, order: list[str], path=None) -> str:
    """Log-scale distribution of a measure, one panel per dataset."""
    sets = [d for d in order if d in frames]
    cols = 2 if len(sets) > 1 else 1
    rows = int(np.ceil(len(sets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.7 * cols, 3.5 * rows),
                             facecolor=SURFACE, squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, sets):
        v = frames[name]
        v = pd.to_numeric(v[measure], errors="coerce").fillna(0)
        v = v[v > 0].to_numpy(float)
        if v.size < 5:
            style(ax, f"{name} — no data")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        med, p95 = float(np.median(v)), float(np.percentile(v, 95))
        ax.hist(np.log10(np.maximum(v, 1e-9)), bins=42, color=ACCENT,
                edgecolor=SURFACE, linewidth=.6)
        ax.axvline(np.log10(max(med, 1e-9)), color=INK_SOFT, lw=1.6, ls="--")
        ax.axvline(np.log10(max(p95, 1e-9)), color=SERIES[1], lw=2.2)
        style(ax, "", "value (log scale)", "units")
        ax.set_title(f"{name}\n{v.size:,} non-zero  ·  median {fmt(med)} (dashed)"
                     f"  ·  p95 {fmt(p95)} (orange)", color=INK, fontsize=10.5,
                     fontweight="600", loc="left", pad=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda t, _: fmt(10 ** t)))
    for ax in flat[len(sets):]:
        ax.set_visible(False)
    fig.suptitle(f"{measure} — distribution among units with a non-zero value",
                 color=INK, fontsize=13, fontweight="700", x=0.008, ha="left", y=1.004)
    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    return save(fig, path)


def concentration_curves(frames: dict, measure: str, order: list[str], path=None) -> str:
    """Share of the measure's mass held by the top X% of units."""
    fig, ax = plt.subplots(figsize=(9.8, 5.4), facecolor=SURFACE)
    grid = np.logspace(np.log10(0.1), np.log10(100), 260)
    marks = []
    for i, name in enumerate([d for d in order if d in frames]):
        v = pd.to_numeric(frames[name][measure], errors="coerce").fillna(0)
        v = np.sort(v[v > 0].to_numpy(float))[::-1]
        if v.size < 20 or v.sum() <= 0:
            continue
        cum = np.cumsum(v) / v.sum() * 100
        g = grid[grid >= 100.0 / v.size]
        y = cum[np.clip((g / 100 * v.size).astype(int), 1, v.size) - 1]
        c = SERIES[i % len(SERIES)]
        ax.plot(g, y, lw=2.2, color=c, solid_capstyle="round", zorder=3)
        if g[0] <= 5.0:
            at5 = float(np.interp(5.0, g, y))
            ax.plot([5], [at5], "o", ms=7.5, color=c, zorder=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.8)
            marks.append((name, c, at5))
    ax.axvline(5, color=INK_SOFT, lw=1.4, ls="--", zorder=1)
    ax.text(5.4, 2, "top 5%", color=INK_SOFT, fontsize=9.5, rotation=90, va="bottom")
    marks.sort(key=lambda m: -m[2])
    ax.text(0.985, 0.30, "at the top 5% of units", transform=ax.transAxes, ha="right",
            va="top", fontsize=9.5, color=INK_SOFT, fontweight="600")
    for j, (name, c, val) in enumerate(marks):
        ax.text(0.985, 0.245 - j * .055, f"{name}   {val:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=c, fontweight="650")
    ax.set_xscale("log"); ax.set_xlim(0.1, 100); ax.set_ylim(0, 103)
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 25, 50, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    style(ax, f"Concentration of {measure}",
          "top X% of units, ranked by the measure", "cumulative share of the total")
    fig.tight_layout()
    return save(fig, path)


def contribution_bars(contribution: pd.DataFrame, order: list[str], path=None) -> str:
    """Each platform measure's share of the total."""
    wide = contribution.pivot_table(index="dataset", columns="measure",
                                    values="share", aggfunc="sum").fillna(0)
    wide = wide.reindex([d for d in order if d in wide.index])
    fig, ax = plt.subplots(figsize=(10.5, 0.75 * len(wide) + 2), facecolor=SURFACE)
    y = np.arange(len(wide))
    left = np.zeros(len(wide))
    for i, col in enumerate(wide.columns):
        v = wide[col].to_numpy(float) * 100
        ax.barh(y, v, left=left, color=SERIES[i % len(SERIES)], height=.62,
                edgecolor=SURFACE, linewidth=1.2, label=col.replace("_engagement", ""))
        for j, (val, l) in enumerate(zip(v, left)):
            if val > 6:
                ax.text(l + val / 2, y[j], f"{val:.0f}%", ha="center", va="center",
                        color="white", fontsize=9.5, fontweight="700")
        left += v
    ax.set_yticks(y); ax.set_yticklabels(wide.index)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    style(ax, "Where the engagement mass comes from", "share of total engagement", "")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT,
              ncol=min(6, len(wide.columns)), loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    return save(fig, path)


def correlation_bars(corr: pd.DataFrame, order: list[str], path=None) -> str:
    """Pearson against Spearman for the two engagement measures."""
    df = corr.set_index("dataset").reindex([d for d in order if d in set(corr["dataset"])])
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
    x = np.arange(len(df)); w = .36
    ax.bar(x - w/2, df["pearson"], w, color=SERIES[0], label="Pearson (levels)")
    ax.bar(x + w/2, df["spearman"], w, color=SERIES[1], label="Spearman (ranks)")
    for i, (p, s) in enumerate(zip(df["pearson"], df["spearman"])):
        ax.text(x[i] - w/2, p, f"{p:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK_SOFT, fontweight="600")
        ax.text(x[i] + w/2, s, f"{s:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK_SOFT, fontweight="600")
    ax.set_xticks(x); ax.set_xticklabels(df.index)
    ax.set_ylim(0, max(1.0, float(np.nanmax(df[["pearson", "spearman"]].to_numpy())) * 1.25))
    style(ax, "Agreement between the derived total and the platform's own engagement column",
          "", "correlation")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
    fig.tight_layout()
    return save(fig, path)


def overlap_chart(overlap: pd.DataFrame, order: list[str], path=None) -> str:
    """How much the priority measures agree, stratum by stratum."""
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    periods = sorted(overlap["period"].unique())
    x = np.arange(len(periods))
    sets = [d for d in order if d in set(overlap["dataset"])]
    w = .8 / max(len(sets), 1)
    for i, name in enumerate(sets):
        sub = overlap[overlap["dataset"] == name].set_index("period").reindex(periods)
        ax.bar(x + i * w - .4 + w / 2, sub["jaccard"].fillna(0) * 100, w,
               color=SERIES[i % len(SERIES)], label=name)
    _periods(ax, periods)
    style(ax, "Overlap between the priority measures (Jaccard, per stratum)",
          "period", "% of the priority pool chosen by every measure")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=len(sets))
    fig.tight_layout()
    return save(fig, path)
