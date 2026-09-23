"""
Sample size and quota allocation.

Two steps. First decide how many units each period should contribute; then
split that number across the datasets present in the period. Both steps are
exact: quotas sum to the period target, and no stratum is asked for more units
than it contains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DesignSpec


def stratum_counts(frames: dict[str, pd.DataFrame], dataset_order: list[str],
                   sort_measure: str) -> pd.DataFrame:
    """One row per dataset x period: size, and the dispersion Neyman needs."""
    rows = []
    for name in dataset_order:
        df = frames.get(name)
        if df is None or df.empty:
            continue
        grouped = df.groupby("period", sort=True)
        agg = grouped.agg(count=(sort_measure, "size"),
                          sd=(sort_measure, "std"),
                          total=(sort_measure, "sum"))
        for period, r in agg.iterrows():
            rows.append({"dataset": name, "period": period,
                         "count": int(r["count"]),
                         "sd": float(r["sd"]) if np.isfinite(r["sd"]) else 0.0,
                         "measure_total": float(r["total"])})
    return pd.DataFrame(rows).sort_values(["period", "dataset"]).reset_index(drop=True)


def period_sample_sizes(counts: pd.DataFrame, design: DesignSpec) -> pd.DataFrame:
    """
    Target sample size per period: rate x period total, capped and floored.

    The cap is what makes the design unequal-probability. A period whose target
    is cut by the cap is sampled at a lower rate than one that is not, so the
    units in it each stand for more of the population. `realized_rate` and
    `period_weight` record that, and every downstream weight is built from them.
    """
    out = (counts.groupby("period", as_index=False)["count"]
                 .sum().rename(columns={"count": "period_total"}))
    out["target_raw"] = out["period_total"] * design.rate
    target = out["target_raw"]
    if design.cap is not None:
        target = np.minimum(target, design.cap)
    if design.floor is not None:
        target = np.maximum(target, design.floor)
    out["target"] = np.minimum(target.round(), out["period_total"]).astype(int)
    out["capped"] = design.cap is not None and (out["target_raw"] > design.cap)
    out["realized_rate"] = out["target"] / out["period_total"]
    out["period_weight"] = 1.0 / out["realized_rate"]
    return out


def _largest_remainder(raw: np.ndarray, target: int, capacity: np.ndarray) -> np.ndarray:
    """
    Round a real-valued allocation to integers summing exactly to `target`,
    never exceeding each stratum's capacity.

    Floor first, then hand the leftover units to the largest fractional parts.
    If a stratum is at capacity it is skipped and the unit moves on, so a
    period whose largest stratum is small still reaches its target rather than
    silently under-filling.
    """
    alloc = np.minimum(np.floor(raw), capacity).astype(int)
    remainder = raw - np.floor(raw)
    shortfall = int(target - alloc.sum())
    if shortfall <= 0:
        while shortfall < 0:                      # over-allocated: take back
            eligible = np.where(alloc > 0)[0]
            if eligible.size == 0:
                break
            victim = eligible[np.argmin(remainder[eligible])]
            alloc[victim] -= 1
            shortfall += 1
        return alloc
    order = np.argsort(-remainder, kind="mergesort")
    cycles = 0
    while shortfall > 0 and cycles < len(order) + target:
        progressed = False
        for i in order:
            if shortfall == 0:
                break
            if alloc[i] < capacity[i]:
                alloc[i] += 1
                shortfall -= 1
                progressed = True
        cycles += 1
        if not progressed:                         # every stratum at capacity
            break
    return alloc


def allocate_quotas(counts: pd.DataFrame, sizes: pd.DataFrame,
                    design: DesignSpec) -> pd.DataFrame:
    """
    Split each period's target across its datasets.

    proportional  share of the period's units (self-weighting within a period)
    neyman        share of units x within-stratum SD, which puts more of the
                  budget where the measure varies most and lowers the variance
                  of a total; it makes weights unequal within a period, which
                  is why the weights below are computed per stratum, not per
                  period.
    """
    df = counts.merge(sizes[["period", "period_total", "target",
                             "realized_rate", "period_weight", "capped"]],
                      on="period", how="left")

    if design.allocation == "neyman":
        df["_basis"] = df["count"] * df["sd"].replace(0, np.nan)
        df["_basis"] = df["_basis"].fillna(df["count"].astype(float))
    else:
        df["_basis"] = df["count"].astype(float)

    df["share"] = df.groupby("period")["_basis"].transform(lambda s: s / s.sum())
    df["raw_quota"] = df["share"] * df["target"]

    if design.min_stratum_quota:
        df["raw_quota"] = np.maximum(df["raw_quota"],
                                     np.minimum(design.min_stratum_quota, df["count"]))

    quotas = []
    for period, g in df.groupby("period", sort=True):
        alloc = _largest_remainder(g["raw_quota"].to_numpy(),
                                   int(g["target"].iloc[0]),
                                   g["count"].to_numpy())
        quotas.append(pd.Series(alloc, index=g.index))
    df["quota"] = pd.concat(quotas).sort_index().astype(int)

    df["stratum_rate"] = df["quota"] / df["count"]
    df["stratum_weight"] = np.where(df["quota"] > 0, 1.0 / df["stratum_rate"], np.nan)

    cols = ["period", "dataset", "count", "sd", "period_total", "target", "capped",
            "share", "raw_quota", "quota", "stratum_rate", "stratum_weight",
            "realized_rate", "period_weight"]
    return df[cols].sort_values(["period", "dataset"]).reset_index(drop=True)


def validate_plan(plan: pd.DataFrame) -> list[str]:
    """Checks that must hold before a single unit is drawn."""
    problems = []
    per_period = plan.groupby("period").agg(alloc=("quota", "sum"),
                                            target=("target", "first"),
                                            capacity=("count", "sum"))
    for period, r in per_period.iterrows():
        if r["alloc"] != r["target"] and r["alloc"] != r["capacity"]:
            problems.append(
                f"{period}: quotas sum to {r['alloc']}, target is {r['target']}")
    over = plan[plan["quota"] > plan["count"]]
    for _, r in over.iterrows():
        problems.append(
            f"{r['period']}/{r['dataset']}: quota {r['quota']} exceeds population {r['count']}")
    bad = plan[(plan["quota"] > 0) & (~np.isfinite(plan["stratum_weight"]))]
    for _, r in bad.iterrows():
        problems.append(f"{r['period']}/{r['dataset']}: non-finite design weight")
    return problems
