"""Shared fixtures. Plain functions so the suite runs under unittest or pytest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engagement_sampling.config import (DesignSpec, EngagementMeasure, EngagementSpec,
                                        FilterSpec, FrameSpec, StudyConfig)


def engagement_spec() -> EngagementSpec:
    return EngagementSpec(
        measures=[
            EngagementMeasure("alpha_engagement", {"a_likes": 1.0, "a_views": 0.1}),
            EngagementMeasure("beta_engagement", {"b_shares": 1.0}),
        ],
        total_name="total_engagement",
        source_column="engagement",
        source_name="source_engagement",
    )


def study() -> StudyConfig:
    return StudyConfig(
        name="test study",
        datasets={"one": "one", "two": "two"},
        engagement=engagement_spec(),
        frame=FrameSpec(filters=[FilterSpec("language", ["lang"], ["en"])]),
        design=DesignSpec(rate=0.10, cap=100, priority_measures=["total_engagement"],
                          top_pct=0.05, seed=7),
        keep_columns=["url", "content"],
    )


def synthetic_corpus(n_per_stratum: int = 400) -> pd.DataFrame:
    """Heavy-tailed, zero-inflated, with reposts concentrated in the tail."""
    rng = np.random.default_rng(0)
    rows = []
    for dataset in ("one", "two"):
        for month in ("01", "02", "03"):
            for i in range(n_per_stratum):
                viral = rng.random() < 0.06
                likes = float(rng.pareto(1.2) * 500) if viral else float(rng.integers(0, 3))
                rows.append({
                    "url": f"http://x/{dataset}/{month}/{i}",
                    "lang": "en" if i % 50 else "fr",
                    "published": f"2026-{month}-{1 + i % 27:02d} 12:00:00",
                    "content": (f"viral text {i % 7}" if viral
                                else f"text {dataset} {month} {i}"),
                    "a_likes": likes,
                    "a_views": likes * 20,
                    "b_shares": 0.0,
                    "engagement": likes / 10 if viral else float(rng.integers(0, 2)),
                })
    return pd.DataFrame(rows)


def write_corpus(tmp: Path, df: pd.DataFrame) -> Path:
    """Split the corpus into two dataset files the loader can discover."""
    tmp.mkdir(parents=True, exist_ok=True)
    half = len(df) // 2
    df.iloc[:half].to_csv(tmp / "export_one.csv", index=False)
    df.iloc[half:].to_csv(tmp / "export_two.csv", index=False)
    return tmp
