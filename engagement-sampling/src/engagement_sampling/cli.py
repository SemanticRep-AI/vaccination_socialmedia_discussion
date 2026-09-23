#!/usr/bin/env python3
"""Command line interface."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .config import StudyConfig
from .pipeline import run_study


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="engagement-sampling",
        description="Design-based sampling for large social media corpora.")
    p.add_argument("--config", required=True, help="study configuration (.yaml or .json)")
    p.add_argument("--input", default="data", help="folder holding the exports")
    p.add_argument("--output", default="output", help="where results are written")
    p.add_argument("--cache", default="cache", help="parsed-export cache")
    p.add_argument("--refresh", action="store_true", help="re-read the exports")
    p.add_argument("--rate", type=float, help="override the sampling rate")
    p.add_argument("--cap", type=int, help="override the per-period cap")
    p.add_argument("--top-pct", type=float, help="override the priority pool size")
    p.add_argument("--seed", type=int, help="override the master seed")
    p.add_argument("--no-report", action="store_true", help="skip the HTML report")
    p.add_argument("--no-coding-files", action="store_true", help="skip the per-period coder files")
    p.add_argument("--no-open", action="store_true", help="do not open the report")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = StudyConfig.load(args.config)

    overrides = {k: v for k, v in
                 {"rate": args.rate, "cap": args.cap,
                  "top_pct": args.top_pct, "seed": args.seed}.items() if v is not None}
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, design=replace(cfg.design, **overrides))
        print(f"design overrides: {overrides}")

    out_dir = Path(args.output)
    result = run_study(cfg, Path(args.input), out_dir, Path(args.cache),
                       refresh=args.refresh,
                       write_coding_files=not args.no_coding_files)

    if not args.no_report:
        print("[5/5] writing the report")
        from .report import build_report
        path = build_report(result, out_dir / "report.html")
        print(f"    {path}")
        if not args.no_open:
            try:
                webbrowser.open(path.resolve().as_uri())
            except Exception:
                pass
    else:
        print("[5/5] report skipped")

    sample = result["sample"]
    plan = result["plan"]
    print(f"\ndone in {result['elapsed']:.0f}s")
    print(f"  frame      {sum(len(f) for f in result['frames'].values()):,} units")
    print(f"  planned    {int(plan['quota'].sum()):,} units")
    print(f"  drawn      {len(sample):,} units")
    if not result["design_effect"].empty:
        row = result["design_effect"].iloc[0]
        print(f"  effective  {row['effective_n']:,.0f} (design effect {row['design_effect']:.2f})")
    issues = result["plan_issues"] + result["sample_issues"]
    print(f"  validation {'no problems found' if not issues else f'{len(issues)} problem(s)'}")
    print(f"  output     {out_dir.resolve()}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
