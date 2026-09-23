"""
Configuration objects.

Everything the pipeline needs to know about a study - which datasets exist,
how engagement is built, which filters apply, how large the sample should be -
is declared here and nowhere else. A study is fully described by one YAML or
JSON file; no module contains a dataset name, a column name or a weight.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngagementMeasure:
    """
    One derived engagement column, built as a weighted sum of source columns.

    Weights below 1 down-weight passive metrics (views, impressions), which are
    two to three orders of magnitude larger than deliberate actions and would
    otherwise determine the total on their own.
    """
    name: str
    components: dict[str, float]
    aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class EngagementSpec:
    """The full engagement model: per-platform measures plus their total."""
    measures: list[EngagementMeasure]
    total_name: str = "content_engagement"
    source_column: str | None = "engagement"
    source_name: str = "source_engagement"

    @property
    def measure_names(self) -> list[str]:
        return [m.name for m in self.measures]


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterSpec:
    """
    A keep-if-value-matches filter over one or more interchangeable columns.

    `columns` holds alternative encodings of the same fact - a country name and
    its ISO code, say. A row is kept when any of them matches. `allow_blank`
    decides what happens to rows where every listed column is empty.
    """
    name: str
    columns: list[str]
    keep: list[str]
    allow_blank: bool = False


@dataclass(frozen=True)
class FrameSpec:
    """How the sampling frame is built from the raw exports."""
    date_column: str = "published"
    period: str = "M"                       # M, W, Q - any pandas period alias
    text_column: str = "content"
    id_column: str = "url"
    filters: list[FilterSpec] = field(default_factory=list)
    deduplicate: bool = True
    dedup_normalise: bool = True            # casefold + collapse whitespace
    drop_missing_text: bool = True


# ---------------------------------------------------------------------------
# Sample design
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignSpec:
    """
    The sample design.

    rate / cap       target size per period: min(rate x period total, cap)
    allocation       'proportional' (share of period volume) or 'neyman'
                     (share of volume x within-stratum SD of the sort measure)
    priority_measures  columns whose top `top_pct` form the priority pool;
                     a post in the top of ANY of them is a candidate
    top_pct          size of that pool, as a fraction of the stratum
    remainder_reserve  minimum share of each stratum's quota drawn at random
                     from OUTSIDE the priority pool. Without it, a stratum
                     whose priority pool is larger than its quota is sampled
                     entirely from the pool, every other unit has an inclusion
                     probability of zero, and the sample can no longer estimate
                     anything about the stratum as a whole. Reserving a slice
                     keeps every unit reachable and the design estimable.
    seed             master seed; every stratum derives its own from it
    min_stratum_quota  never allocate fewer than this to a non-empty stratum
    """
    rate: float = 0.10
    cap: int | None = 2000
    floor: int | None = None
    allocation: str = "proportional"
    priority_measures: list[str] = field(default_factory=lambda: ["content_engagement"])
    top_pct: float = 0.05
    remainder_reserve: float = 0.20
    seed: int = 42
    min_stratum_quota: int = 0


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudyConfig:
    name: str
    datasets: dict[str, str]                # key -> filename substring
    engagement: EngagementSpec
    frame: FrameSpec = field(default_factory=FrameSpec)
    design: DesignSpec = field(default_factory=DesignSpec)
    keep_columns: list[str] = field(default_factory=list)
    dataset_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.dataset_order:
            object.__setattr__(self, "dataset_order", list(self.datasets))
        unknown = set(self.dataset_order) - set(self.datasets)
        if unknown:
            raise ValueError(f"dataset_order names unknown datasets: {sorted(unknown)}")
        known = set(self.engagement.measure_names) | {
            self.engagement.total_name, self.engagement.source_name}
        missing = [m for m in self.design.priority_measures if m not in known]
        if missing:
            raise ValueError(
                f"design.priority_measures refers to columns the engagement spec "
                f"does not produce: {missing}. Available: {sorted(known)}")
        if not 0 < self.design.rate <= 1:
            raise ValueError("design.rate must be in (0, 1]")
        if not 0 < self.design.top_pct <= 1:
            raise ValueError("design.top_pct must be in (0, 1]")
        if not 0 <= self.design.remainder_reserve < 1:
            raise ValueError("design.remainder_reserve must be in [0, 1)")
        if self.design.allocation not in {"proportional", "neyman"}:
            raise ValueError("design.allocation must be 'proportional' or 'neyman'")

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StudyConfig":
        eng = d["engagement"]
        measures = [EngagementMeasure(
            name=m["name"],
            components={k: float(v) for k, v in m["components"].items()},
            aliases=m.get("aliases", {}),
        ) for m in eng["measures"]]
        engagement = EngagementSpec(
            measures=measures,
            total_name=eng.get("total_name", "content_engagement"),
            source_column=eng.get("source_column", "engagement"),
            source_name=eng.get("source_name", "source_engagement"),
        )
        fr = d.get("frame", {})
        frame = FrameSpec(
            date_column=fr.get("date_column", "published"),
            period=fr.get("period", "M"),
            text_column=fr.get("text_column", "content"),
            id_column=fr.get("id_column", "url"),
            filters=[FilterSpec(**f) for f in fr.get("filters", [])],
            deduplicate=fr.get("deduplicate", True),
            dedup_normalise=fr.get("dedup_normalise", True),
            drop_missing_text=fr.get("drop_missing_text", True),
        )
        design = DesignSpec(**d.get("design", {}))
        return cls(
            name=d["name"],
            datasets=d["datasets"],
            engagement=engagement,
            frame=frame,
            design=design,
            keep_columns=d.get("keep_columns", []),
            dataset_order=d.get("dataset_order", []),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StudyConfig":
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:                       # pragma: no cover
                raise ImportError(
                    "Reading a YAML config needs PyYAML (pip install pyyaml). "
                    "A JSON config works with no extra dependency."
                ) from exc
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
