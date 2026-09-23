"""End-to-end run: frame, diagnose, plan, draw, report."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import allocate, diagnostics, frame as frame_mod
from .config import StudyConfig
from .sample import draw_sample, validate_sample


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=df.index.name is not None)
    return path


def run_study(cfg: StudyConfig, input_dir: Path, out_dir: Path, cache_dir: Path,
              refresh: bool = False, write_coding_files: bool = True) -> dict:
    """
    Run the whole study and return every artefact it produced.

    The order is deliberate: the frame is finished before anything is measured,
    and everything is measured before anything is drawn, so no diagnostic is
    computed on a population the sample was already taken from.
    """
    started = datetime.now()
    result: dict[str, object] = {"config": cfg, "started": started}

    # ---- frame ------------------------------------------------------------
    print("\n[1/5] building the frame")
    frames, funnel, missing = frame_mod.build_frame(cfg, input_dir, cache_dir, refresh)
    d00 = out_dir / "00_frame"
    _write(funnel, d00 / "funnel.csv")
    (d00 / "config_used.json").parent.mkdir(parents=True, exist_ok=True)
    (d00 / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    if missing:
        (d00 / "missing_columns.txt").write_text("\n".join(missing), encoding="utf-8")
    result["frames"], result["funnel"] = frames, funnel

    total = cfg.engagement.total_name
    measures = cfg.engagement.measure_names
    priority = cfg.design.priority_measures

    # ---- frame diagnostics ------------------------------------------------
    print("[2/5] diagnosing the frame")
    d01 = out_dir / "01_diagnostics"
    described = diagnostics.describe_measures(frames, measures + [total, cfg.engagement.source_name])
    contribution = diagnostics.platform_contribution(frames, measures, total)
    concentration = diagnostics.concentration_table(frames, total)
    overlap = diagnostics.measure_overlap(frames, priority, cfg.design.top_pct, cfg.design.seed)
    correlations = pd.DataFrame()
    if cfg.engagement.source_column:
        correlations = diagnostics.correlation_table(frames, total, cfg.engagement.source_name)
        _write(correlations, d01 / "correlations.csv")
    _write(described, d01 / "measure_descriptives.csv")
    _write(contribution, d01 / "platform_contribution.csv")
    _write(concentration, d01 / "concentration.csv")
    if not overlap.empty:
        _write(overlap, d01 / "measure_overlap.csv")
    result.update(descriptives=described, contribution=contribution,
                  concentration=concentration, overlap=overlap, correlations=correlations)

    # ---- plan -------------------------------------------------------------
    print("[3/5] allocating quotas")
    counts = allocate.stratum_counts(frames, cfg.dataset_order, total)
    sizes = allocate.period_sample_sizes(counts, cfg.design)
    plan = allocate.allocate_quotas(counts, sizes, cfg.design)
    issues = allocate.validate_plan(plan)
    d02 = out_dir / "02_plan"
    _write(sizes, d02 / "period_sample_sizes.csv")
    _write(plan, d02 / "quota_plan.csv")
    (d02 / "validation.txt").write_text(
        "\n".join(issues) if issues else "plan validated: no problems found\n", encoding="utf-8")
    for problem in issues:
        print(f"    PLAN WARNING: {problem}")
    result.update(counts=counts, sizes=sizes, plan=plan, plan_issues=issues)

    # ---- draw -------------------------------------------------------------
    print("[4/5] drawing the sample")
    sample, summary = draw_sample(frames, plan, priority, cfg.design.top_pct,
                                  cfg.design.seed, cfg.design.remainder_reserve)
    problems = validate_sample(sample, plan, frames)
    for problem in problems:
        print(f"    SAMPLE WARNING: {problem}")

    d03 = out_dir / "03_sample"
    keep = [c for c in cfg.keep_columns if c in sample.columns]
    keep += [c for c in ["dataset", "period", "duplicate_count", *measures, total,
                         cfg.engagement.source_name, "selection_phase",
                         "priority_measures_hit", "inclusion_prob", "design_weight",
                         "cluster_weight", "stratum_seed"] if c in sample.columns]
    d03.mkdir(parents=True, exist_ok=True)
    sample[keep].to_csv(d03 / "sample.csv.gz", index=False, compression="gzip")
    _write(summary, d03 / "sampling_summary.csv")

    deff = diagnostics.design_effect(sample) if not sample.empty else pd.DataFrame()
    deff_phase = (diagnostics.design_effect_by(sample, "selection_phase")
                  if not sample.empty else pd.DataFrame())
    overlap_ds = (diagnostics.cross_dataset_overlap(sample) if not sample.empty
                  else pd.DataFrame())
    if not deff_phase.empty:
        _write(deff_phase, d03 / "design_effect_by_phase.csv")
    if not overlap_ds.empty:
        _write(overlap_ds, d03 / "cross_dataset_overlap.csv")
    cover = diagnostics.coverage(sample, frames, total) if not sample.empty else pd.DataFrame()
    if not deff.empty:
        _write(deff, d03 / "design_effect.csv")
    if not cover.empty:
        _write(cover, d03 / "coverage.csv")
    (d03 / "validation.txt").write_text(
        "\n".join(problems) if problems else "sample validated: no problems found\n",
        encoding="utf-8")
    result.update(sample=sample, summary=summary, design_effect=deff,
                  design_effect_phase=deff_phase, cross_dataset=overlap_ds,
                  coverage=cover, sample_issues=problems)

    # ---- coder files ------------------------------------------------------
    if write_coding_files and not sample.empty:
        d04 = out_dir / "04_coding"
        d04.mkdir(parents=True, exist_ok=True)
        cols = [c for c in [cfg.frame.id_column, cfg.frame.text_column,
                            "dataset", "period", "design_weight"] if c in sample.columns]
        # A post can match more than one search profile. Coders should see each
        # text once, so the coder files are made unique across datasets while the
        # analytic sample keeps every dataset membership.
        coder = sample
        if "_content_key" in sample.columns:
            coder = sample.drop_duplicates(subset="_content_key", keep="first")
        for period, g in coder.groupby("period"):
            g[cols].to_csv(d04 / f"coding_{period}.csv", index=False)
        dropped = len(sample) - len(coder)
        print(f"    coder files: {coder['period'].nunique()} periods, {len(coder):,} texts "
              f"({dropped:,} cross-dataset repeats removed) -> {d04}")

    result["elapsed"] = (datetime.now() - started).total_seconds()
    return result
