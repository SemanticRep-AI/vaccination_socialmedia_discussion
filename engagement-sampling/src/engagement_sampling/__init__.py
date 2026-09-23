"""
engagement_sampling
===================

A configurable, design-based sampler for large social media corpora.

The package builds a sampling frame from platform exports, derives engagement
measures from a declared formula, allocates a stratified sample across periods
and datasets, and draws it while recording the inclusion probability of every
selected unit - so the resulting sample supports population estimation, not
only description.

    from engagement_sampling import StudyConfig, run_study

    cfg = StudyConfig.load("config/study.yaml")
    result = run_study(cfg, Path("data"), Path("output"), Path("cache"))
"""
from .config import (StudyConfig, EngagementSpec, EngagementMeasure,
                     FrameSpec, FilterSpec, DesignSpec)
from .pipeline import run_study

__version__ = "1.0.0"
__all__ = ["StudyConfig", "EngagementSpec", "EngagementMeasure", "FrameSpec",
           "FilterSpec", "DesignSpec", "run_study", "__version__"]
