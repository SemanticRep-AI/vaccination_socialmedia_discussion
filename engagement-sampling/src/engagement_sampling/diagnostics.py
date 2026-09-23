"""
Diagnostics.

Two kinds. Frame diagnostics describe the population before anything is drawn
and justify the design. Sample diagnostics describe what the draw achieved and
what it costs in precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sample import stratum_seed, top_indices


# ---------------------------------------------------------------------------
# Frame diagnostics
# ---------------------------------------------------------------------------

DEFAULT_PERCENTILES = [50, 75, 90, 95, 99, 99.9]


def describe_measures(frames: dict[str, pd.DataFrame], measures: list[str],
                      percentiles: list[float] = None) -> pd.DataFrame:
    """
    Shape of each measure, over the whole frame and over its non-zero part.

    Both are reported because most of these measures are zero for most rows. A
    median taken over every row says only how common a zero is; the non-zero
    view is the one that describes the behaviour being measured.
    """
    percentiles = percentiles or DEFAULT_PERCENTILES
    rows = []
    for name, df in frames.items():
        for measure in measures:
            if measure not in df.columns:
                continue
            full = pd.to_numeric(df[measure], errors="coerce").fillna(0.0)
            for scope, v in (("all", full), ("nonzero", full[full > 0])):
                arr = v.to_numpy(dtype=float)
                row = {"dataset": name, "measure": measure, "scope": scope,
                       "n": arr.size,
                       "nonzero_pct": round(float((full > 0).mean() * 100), 3),
                       "total": float(arr.sum()) if arr.size else 0.0,
                       "mean": float(arr.mean()) if arr.size else 0.0,
                       "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                       "max": float(arr.max()) if arr.size else 0.0}
                for p in percentiles:
                    row[f"p{p:g}"] = float(np.percentile(arr, p)) if arr.size else 0.0
                rows.append(row)
    return pd.DataFrame(rows)


def top_p_magnitude(values: pd.Series, p: float) -> dict:
    """
    How much of a measure sits in its top p fraction.

    threshold       the value a unit must reach to be in the top p
    share_of_total  how much of the measure's mass those units hold
    """
    arr = np.sort(pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(float))[::-1]
    n = arr.size
    if n == 0:
        return {"p": p, "threshold": 0.0, "n_top": 0, "sum_top": 0.0,
                "share_of_total": 0.0, "mean_top": 0.0}
    k = max(1, int(np.ceil(n * p)))
    head = arr[:k]
    total = float(arr.sum())
    return {"p": p, "threshold": float(head[-1]), "n_top": int(k),
            "sum_top": float(head.sum()),
            "share_of_total": float(head.sum() / total) if total > 0 else 0.0,
            "mean_top": float(head.mean())}


def concentration_table(frames: dict[str, pd.DataFrame], measure: str,
                        cuts=(0.01, 0.02, 0.05, 0.10, 0.25)) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        if measure not in df.columns:
            continue
        active = df.loc[df[measure] > 0, measure]
        for p in cuts:
            rows.append({"dataset": name, "measure": measure, **top_p_magnitude(active, p)})
    return pd.DataFrame(rows)


def correlation_table(frames: dict[str, pd.DataFrame], a: str, b: str) -> pd.DataFrame:
    """
    Agreement between two measures, by Pearson and by Spearman.

    Both are reported because they answer different questions here. Pearson on
    a variable this heavy-tailed is dominated by a handful of extreme posts and
    understates any relationship in the body of the distribution. Spearman ranks
    first, which is also the form the sampling rule actually uses - the design
    selects on order, not on magnitude - so it is the relevant coefficient for
    judging whether two measures would pick the same posts.
    """
    rows = []
    for name, df in frames.items():
        if a not in df.columns or b not in df.columns:
            continue
        x = pd.to_numeric(df[a], errors="coerce")
        y = pd.to_numeric(df[b], errors="coerce")
        ok = x.notna() & y.notna()
        rows.append({
            "dataset": name, "measure_a": a, "measure_b": b, "n": int(ok.sum()),
            "pearson": float(x[ok].corr(y[ok])) if ok.sum() > 2 else np.nan,
            "spearman": float(x[ok].corr(y[ok], method="spearman")) if ok.sum() > 2 else np.nan,
            "both_zero_pct": round(float(((x == 0) & (y == 0))[ok].mean() * 100), 2),
        })
    return pd.DataFrame(rows)


def platform_contribution(frames: dict[str, pd.DataFrame], measures: list[str],
                          total: str) -> pd.DataFrame:
    """Each measure's share of the total, which shows where the data actually is."""
    rows = []
    for name, df in frames.items():
        grand = float(df[total].sum()) if total in df.columns else 0.0
        for measure in measures:
            if measure not in df.columns:
                continue
            value = float(df[measure].sum())
            rows.append({"dataset": name, "measure": measure, "total": value,
                         "share": value / grand if grand > 0 else 0.0,
                         "nonzero_posts": int((df[measure] > 0).sum())})
    return pd.DataFrame(rows)


def measure_overlap(frames: dict[str, pd.DataFrame], measures: list[str],
                    top_pct: float, seed: int) -> pd.DataFrame:
    """
    How much the priority measures agree about which units are important.

    If the overlap is near zero the measures are selecting disjoint sets, the
    priority pool is roughly the sum of their sizes rather than their union,
    and any rule that ranks "chosen by both" ahead of "chosen by one" has
    almost nothing to rank.
    """
    rows = []
    if len(measures) < 2:
        return pd.DataFrame(rows)
    for name, df in frames.items():
        for period, stratum in df.groupby("period", sort=True):
            n = len(stratum)
            k = max(1, int(np.ceil(n * top_pct)))
            sets = []
            for i, measure in enumerate(measures):
                if measure not in stratum.columns:
                    continue
                idx, _ = top_indices(stratum, measure, k, stratum_seed(seed, measure, str(i)))
                sets.append(set(idx))
            if len(sets) < 2:
                continue
            inter = set.intersection(*sets)
            union = set.union(*sets)
            rows.append({"dataset": name, "period": period, "stratum_n": n, "top_k": k,
                         "union": len(union), "intersection": len(inter),
                         "jaccard": len(inter) / len(union) if union else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sample diagnostics
# ---------------------------------------------------------------------------

def design_effect(sample: pd.DataFrame, weight_col: str = "design_weight") -> pd.DataFrame:
    """
    The precision cost of unequal weights.

    Kish's approximation, deff = 1 + CV^2(w). The effective sample size is the
    size of a simple random sample that would estimate a mean about as
    precisely, so it is the number to quote when describing the study's power -
    not the row count.
    """
    rows = []
    for scope, g in [("overall", sample)] + list(sample.groupby("period")):
        w = pd.to_numeric(g[weight_col], errors="coerce").dropna().to_numpy(float)
        if w.size == 0:
            continue
        cv2 = (w.std(ddof=1) / w.mean()) ** 2 if w.mean() > 0 and w.size > 1 else 0.0
        deff = 1 + cv2
        rows.append({"scope": str(scope), "n": int(w.size), "mean_weight": float(w.mean()),
                     "min_weight": float(w.min()), "max_weight": float(w.max()),
                     "cv_squared": float(cv2), "design_effect": float(deff),
                     "effective_n": float(w.size / deff)})
    return pd.DataFrame(rows)


def coverage(sample: pd.DataFrame, frames: dict[str, pd.DataFrame],
             measure: str) -> pd.DataFrame:
    """
    What the sample captures of the frame: units, and mass of the measure.

    A design that prioritises the tail captures a small share of the units and
    a large share of the mass. Reporting both is what makes that trade explicit
    rather than accidental.
    """
    rows = []
    for name, df in frames.items():
        s = sample[sample["dataset"] == name]
        if s.empty or measure not in df.columns:
            continue
        frame_mass = float(df[measure].sum())
        rows.append({
            "dataset": name,
            "frame_units": len(df), "sample_units": len(s),
            "unit_share": len(s) / len(df) if len(df) else 0.0,
            "frame_mass": frame_mass,
            "sample_mass": float(s[measure].sum()),
            "mass_share": float(s[measure].sum()) / frame_mass if frame_mass > 0 else 0.0,
            "weighted_mass_estimate": float((s[measure] * s["design_weight"]).sum()),
        })
    return pd.DataFrame(rows)


def weighted_total(sample: pd.DataFrame, measure: str,
                   by: list[str] | None = None,
                   weight_col: str = "design_weight") -> pd.DataFrame:
    """
    Horvitz-Thompson estimate of a frame total, with its standard error.

    This is the payoff of carrying inclusion probabilities: the sample
    estimates population quantities, rather than only describing itself.
    """
    by = by or []
    groups = sample.groupby(by) if by else [((), sample)]
    rows = []
    for key, g in groups:
        w = pd.to_numeric(g[weight_col], errors="coerce").fillna(0).to_numpy(float)
        y = pd.to_numeric(g[measure], errors="coerce").fillna(0).to_numpy(float)
        n = y.size
        total = float((w * y).sum())
        if n > 1:
            per = w * y
            var = n * per.var(ddof=1)
            se = float(np.sqrt(max(var, 0)))
        else:
            se = float("nan")
        row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}
        rows.append({**row, "measure": measure, "n": n, "total_estimate": total,
                     "se": se, "mean_estimate": total / w.sum() if w.sum() else 0.0})
    return pd.DataFrame(rows)


def design_effect_by(sample: pd.DataFrame, by: str,
                     weight_col: str = "design_weight") -> pd.DataFrame:
    """
    Design effect within each level of `by`.

    Reported alongside the overall figure because the overall number mixes two
    very different phases. The priority phase is close to a census and costs
    almost nothing in precision; the remainder phase is a small simple random
    sample standing in for a large pool. Quoting only the pooled design effect
    makes a deliberately tail-heavy design look like a broken one.
    """
    rows = []
    for level, g in sample.groupby(by):
        w = pd.to_numeric(g[weight_col], errors="coerce").dropna().to_numpy(float)
        if w.size == 0:
            continue
        cv2 = (w.std(ddof=1) / w.mean()) ** 2 if w.mean() > 0 and w.size > 1 else 0.0
        rows.append({by: str(level), "n": int(w.size), "mean_weight": float(w.mean()),
                     "min_weight": float(w.min()), "max_weight": float(w.max()),
                     "design_effect": float(1 + cv2),
                     "effective_n": float(w.size / (1 + cv2))})
    return pd.DataFrame(rows)


def cross_dataset_overlap(sample: pd.DataFrame, key: str = "_content_key") -> pd.DataFrame:
    """
    How often the same text is selected under more than one search profile.

    The datasets are overlapping search profiles, not a partition, so a post can
    legitimately belong to several. That is not an error, but it matters twice:
    a coder should not be shown the same text repeatedly, and a pooled estimate
    across datasets double-counts these units unless the union frame is used.
    """
    if key not in sample.columns:
        return pd.DataFrame()
    per_text = sample.groupby(key)["dataset"].nunique()
    shared = per_text[per_text > 1]
    rows = [{"metric": "unique texts", "value": int(per_text.size)},
            {"metric": "texts in more than one dataset", "value": int(shared.size)},
            {"metric": "rows those texts account for", "value": int(shared.sum())},
            {"metric": "share of sample rows", "value": round(float(shared.sum()) / len(sample), 4)}]
    return pd.DataFrame(rows)
