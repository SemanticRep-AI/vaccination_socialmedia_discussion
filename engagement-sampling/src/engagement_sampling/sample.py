"""
Drawing the sample.

The design is a two-phase stratified draw. Within each dataset x period
stratum the frame is split into a priority pool - the top `top_pct` of any of
the priority measures - and a remainder pool. Units are taken from the priority
pool first and the quota is completed at random from the remainder.

The point of separating the two is that each phase has a known, computable
inclusion probability, so the sample supports design-based estimation rather
than only description. Every returned row carries `inclusion_prob` and
`design_weight`, and the weights reproduce the frame totals by construction.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def stratum_seed(master_seed: int, *parts: str) -> int:
    """
    A stable seed per stratum, derived from the master seed and the stratum's
    identity. Using one seed for every stratum would couple their draws to row
    order; deriving each from a hash keeps them independent and still makes the
    whole run reproducible from a single number.
    """
    key = "|".join([str(master_seed), *parts]).encode("utf-8")
    return int(hashlib.blake2b(key, digest_size=4).hexdigest(), 16)


def _jitter(index: pd.Index, seed: int) -> np.ndarray:
    """A deterministic value in [0,1) per row, used only to break ties."""
    rng = np.random.default_rng(seed)
    return rng.random(len(index))


# ---------------------------------------------------------------------------
# Priority pool
# ---------------------------------------------------------------------------

def top_indices(stratum: pd.DataFrame, measure: str, k: int, seed: int) -> tuple[pd.Index, dict]:
    """
    The k highest rows on `measure`, with ties broken at random rather than by
    file order.

    This matters more than it sounds. In this corpus the platform's own
    engagement column is zero for most rows, so the value at a top-5% cutoff is
    often shared by hundreds of posts. Taking the first k in file order then
    selects on position in the export - which is publication time - and quietly
    turns a measure-based rule into a recency rule. A seeded random tie-break
    keeps the selection exchangeable among tied units, which is what the
    inclusion probabilities below assume.
    """
    values = stratum[measure].to_numpy(dtype=float)
    if k <= 0 or len(values) == 0:
        return stratum.index[:0], {"cutoff": np.nan, "tied_at_cutoff": 0, "tied_taken": 0}

    k = min(k, len(values))
    order = np.lexsort((_jitter(stratum.index, seed), -values))
    chosen = stratum.index[order[:k]]

    cutoff = float(np.sort(values)[::-1][k - 1])
    tied = int((values == cutoff).sum())
    above = int((values > cutoff).sum())
    return chosen, {"cutoff": cutoff, "tied_at_cutoff": tied,
                    "tied_taken": max(k - above, 0)}


# ---------------------------------------------------------------------------
# One stratum
# ---------------------------------------------------------------------------

def sample_stratum(stratum: pd.DataFrame, quota: int, measures: list[str],
                   top_pct: float, seed: int,
                   remainder_reserve: float = 0.20) -> tuple[pd.DataFrame, dict]:
    """
    Draw `quota` units from one stratum and record how each was selected.

    The quota is split in two before anything is drawn. A reserve of
    `remainder_reserve` is set aside for a simple random sample of the units
    OUTSIDE the priority pool; the rest goes to the pool. Both halves then have
    a known, strictly positive inclusion probability:

      priority half   all of the pool if it fits (pi = 1), otherwise a simple
                      random sample of it (pi = take / |P|)
      remainder half  a simple random sample of everything else (pi = r / |R|)

    Reserving that slice is what keeps the design estimable. Draw only from the
    priority pool and every other unit in the stratum has pi = 0 - the sample
    then describes the pool and says nothing about the stratum, and the weights
    will not reproduce the frame.
    """
    n = len(stratum)
    quota = int(min(quota, n))
    if n == 0 or quota == 0:
        return stratum.iloc[:0].copy(), {"population": n, "quota": 0, "drawn": 0}

    k = max(1, int(np.ceil(n * top_pct)))
    flags, tie_info = {}, {}
    for i, measure in enumerate(measures):
        idx, info = top_indices(stratum, measure, k, stratum_seed(seed, measure, str(i)))
        flags[measure] = stratum.index.isin(idx)
        tie_info[measure] = info

    if flags:
        in_priority = np.logical_or.reduce(list(flags.values()))
        n_measures_hit = np.sum(list(flags.values()), axis=0)
    else:                                                      # no priority rule
        in_priority = np.zeros(n, dtype=bool)
        n_measures_hit = np.zeros(n, dtype=int)

    work = stratum.copy()
    work["priority_measures_hit"] = n_measures_hit
    for measure, flag in flags.items():
        work[f"top_{measure}"] = flag

    priority = work[in_priority]
    remainder = work[~in_priority]
    rng = np.random.default_rng(seed)

    if quota >= n:
        selected = work.copy()
        selected["selection_phase"] = np.where(in_priority, "priority", "remainder")
        selected["inclusion_prob"] = 1.0

    else:
        # reserve part of the quota for the remainder pool, so nothing has pi = 0
        reserve = int(np.ceil(quota * remainder_reserve)) if len(remainder) else 0
        reserve = min(reserve, len(remainder), quota)
        priority_take = min(quota - reserve, len(priority))
        reserve = min(quota - priority_take, len(remainder))

        if priority_take >= len(priority):
            a = priority.copy()
            a["inclusion_prob"] = 1.0
        else:
            a = priority.sample(n=priority_take, random_state=rng).copy()
            a["inclusion_prob"] = priority_take / len(priority)
        a["selection_phase"] = "priority"

        if reserve > 0:
            b = remainder.sample(n=reserve, random_state=rng).copy()
            b["inclusion_prob"] = reserve / len(remainder)
        else:
            b = remainder.iloc[:0].copy()
            b["inclusion_prob"] = pd.Series(dtype=float)
        b["selection_phase"] = "remainder"

        selected = pd.concat([a, b]) if len(b) else a

    selected["design_weight"] = 1.0 / selected["inclusion_prob"]
    if "duplicate_count" in selected.columns:
        selected["cluster_weight"] = selected["design_weight"] * selected["duplicate_count"]
    else:
        selected["cluster_weight"] = selected["design_weight"]

    info = {
        "population": n,
        "quota": quota,
        "drawn": len(selected),
        "priority_pool": int(len(priority)),
        "priority_drawn": int((selected["selection_phase"] == "priority").sum()),
        "remainder_drawn": int((selected["selection_phase"] == "remainder").sum()),
        "top_k_per_measure": k,
        "selected_by_all_measures": int((work["priority_measures_hit"] == len(measures)).sum())
        if measures else 0,
        "priority_overlap_pct": round(
            float((work["priority_measures_hit"] == len(measures)).sum())
            / max(len(priority), 1) * 100, 2) if measures else 0.0,
        "remainder_pool": int(len(remainder)),
        "min_inclusion_prob": float(selected["inclusion_prob"].min()),
        "max_design_weight": float(selected["design_weight"].max()),
        "tied_at_cutoff": int(sum(t["tied_at_cutoff"] for t in tie_info.values())),
        "tied_taken": int(sum(t["tied_taken"] for t in tie_info.values())),
    }
    return selected, info


# ---------------------------------------------------------------------------
# All strata
# ---------------------------------------------------------------------------

def draw_sample(frames: dict[str, pd.DataFrame], plan: pd.DataFrame,
                measures: list[str], top_pct: float, seed: int,
                remainder_reserve: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the design over every stratum in the plan."""
    drawn, summaries = [], []

    for _, row in plan.iterrows():
        dataset, period, quota = row["dataset"], str(row["period"]), int(row["quota"])
        df = frames.get(dataset)
        if df is None:
            continue
        stratum = df[df["period"] == period]
        if stratum.empty:
            continue

        s = stratum_seed(seed, dataset, period)
        selected, info = sample_stratum(stratum, quota, measures, top_pct, s,
                                        remainder_reserve)
        if selected.empty:
            continue

        selected = selected.assign(dataset=dataset, period=period, stratum_seed=s)
        drawn.append(selected)
        summaries.append({"period": period, "dataset": dataset,
                          "planned_quota": quota, **info})

    if not drawn:
        return pd.DataFrame(), pd.DataFrame(summaries)

    sample = pd.concat(drawn, ignore_index=True)
    return sample, pd.DataFrame(summaries)


def validate_sample(sample: pd.DataFrame, plan: pd.DataFrame,
                    frames: dict[str, pd.DataFrame]) -> list[str]:
    """Checks that must hold after the draw."""
    problems = []
    if sample.empty:
        return ["sample is empty"]

    got = sample.groupby(["period", "dataset"]).size()
    for _, r in plan.iterrows():
        key = (str(r["period"]), r["dataset"])
        n = int(got.get(key, 0))
        if n != int(r["quota"]):
            problems.append(f"{key[0]}/{key[1]}: drew {n}, quota was {int(r['quota'])}")

    if (sample["inclusion_prob"] <= 0).any() or (sample["inclusion_prob"] > 1).any():
        problems.append("inclusion probabilities outside (0, 1]")

    key = "_content_key"
    if key in sample.columns:
        within = sample.duplicated(subset=["dataset", key]).sum()
        if within:
            problems.append(f"{int(within)} texts repeat within a single dataset")

    # weighted frame size should recover the frame, stratum by stratum
    for (period, dataset), g in sample.groupby(["period", "dataset"]):
        frame_n = len(frames[dataset][frames[dataset]["period"] == period])
        est = g["design_weight"].sum()
        if abs(est - frame_n) / max(frame_n, 1) > 0.10:
            problems.append(
                f"{period}/{dataset}: weights estimate {est:,.0f} units, frame holds {frame_n:,}")
    return problems
