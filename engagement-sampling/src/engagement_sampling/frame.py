"""
Building the sampling frame: load, filter, score, deduplicate.

The frame is the population the sample is drawn from. Every decision that
changes who can be selected happens here, and every one of them is counted, so
the funnel from raw export to frame is fully auditable.
"""
from __future__ import annotations

import hashlib
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EngagementSpec, FrameSpec, StudyConfig

warnings.filterwarnings("ignore", message="Workbook contains no default style")

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9]")


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return _NONWORD.sub("", str(name).lower())


def resolve_column(df: pd.DataFrame, wanted: str,
                   aliases: dict[str, list[str]] | None = None) -> str | None:
    """Find a column despite differences in case, punctuation or known aliases."""
    if wanted in df.columns:
        return wanted
    lookup = {_norm(c): c for c in df.columns}
    if _norm(wanted) in lookup:
        return lookup[_norm(wanted)]
    for alt in (aliases or {}).get(wanted, []):
        if alt in df.columns:
            return alt
        if _norm(alt) in lookup:
            return lookup[_norm(alt)]
    return None


def numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def text(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return (df[col].astype(str).str.strip().str.lower()
            .replace({"nan": "", "none": "", "null": "", "<na>": ""}))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_files(input_dir: Path, datasets: dict[str, str]) -> dict[str, Path]:
    """Match each dataset key to a file by filename substring."""
    candidates = sorted(p for p in input_dir.iterdir()
                        if p.suffix.lower() in {".xlsx", ".csv", ".gz", ".parquet"}
                        and not p.name.startswith("~$"))
    found: dict[str, Path] = {}
    for key, needle in datasets.items():
        hits = [p for p in candidates if _norm(needle) in _norm(p.name)]
        if hits:
            found[key] = hits[0]
    return found


def load_raw(key: str, path: Path, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """
    Read one export, caching the parsed result as compressed CSV.

    Everything is read as text. Each step converts only the columns it uses,
    which keeps type inference from guessing wrong on a wide mixed export.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{key}.csv.gz"
    if cache.exists() and not refresh and cache.stat().st_mtime >= path.stat().st_mtime:
        log(f"{key}: cache")
        return pd.read_csv(cache, dtype=str, low_memory=False).fillna("")

    log(f"{key}: reading {path.name} (first run only)")
    t0 = time.time()
    if path.suffix.lower() == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        header = [str(h) for h in next(rows)]
        df = pd.DataFrame(list(rows), columns=header)
        wb.close()
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path).astype(str)
    else:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    df.to_csv(cache, index=False, compression="gzip")
    log(f"{key}: {len(df):,} x {df.shape[1]} in {time.time() - t0:.0f}s, cached")
    return pd.read_csv(cache, dtype=str, low_memory=False).fillna("")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def apply_filter(df: pd.DataFrame, spec) -> tuple[pd.DataFrame, dict]:
    """Keep rows matching a FilterSpec across any of its interchangeable columns."""
    wanted = {v.strip().lower() for v in spec.keep}
    present = [c for c in (resolve_column(df, c) for c in spec.columns) if c]
    if not present:
        raise KeyError(
            f"Filter '{spec.name}' needs one of {spec.columns}; none is in the export.")

    values = [text(df, c) for c in present]
    match = np.logical_or.reduce([v.isin(wanted).to_numpy() for v in values])
    blank = np.logical_and.reduce([(v == "").to_numpy() for v in values])
    keep = match | (blank if spec.allow_blank else False)

    return df[keep].copy(), {
        "filter": spec.name,
        "kept": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "blank": int(blank.sum()),
        "blank_kept": int((blank & keep).sum()),
    }


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------

def add_engagement(df: pd.DataFrame, spec: EngagementSpec) -> tuple[pd.DataFrame, list[str]]:
    """Add one column per measure, their total, and the export's own column."""
    out = df.copy()
    missing: list[str] = []
    for measure in spec.measures:
        total = pd.Series(0.0, index=out.index)
        for source, weight in measure.components.items():
            col = resolve_column(out, source, measure.aliases)
            if col is None:
                missing.append(f"{measure.name} <- {source}")
                continue
            total = total + numeric(out, col) * weight
        out[measure.name] = total
    out[spec.total_name] = out[spec.measure_names].sum(axis=1)
    if spec.source_column:
        out[spec.source_name] = numeric(out, resolve_column(out, spec.source_column))
    return out, missing


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

def add_period(df: pd.DataFrame, frame: FrameSpec) -> tuple[pd.DataFrame, int]:
    col = resolve_column(df, frame.date_column)
    if col is None:
        raise KeyError(f"Required column '{frame.date_column}' is not in the export.")
    out = df.copy()
    ts = pd.to_datetime(out[col].astype(str).str.strip(), errors="coerce", format="mixed")
    if ts.isna().all():
        ts = pd.to_datetime(out[col], errors="coerce")
    out["_timestamp"] = ts
    out["period"] = ts.dt.to_period(frame.period).astype(str)
    unparsed = int(ts.isna().sum())
    out = out[ts.notna()].copy()
    return out, unparsed


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _content_key(series: pd.Series, normalise: bool) -> pd.Series:
    s = series.astype(str)
    if normalise:
        s = s.str.strip().str.casefold().map(lambda t: _WS.sub(" ", t))
    return s.map(lambda t: hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest())


def deduplicate_frame(df: pd.DataFrame, frame: FrameSpec,
                      sort_measure: str) -> tuple[pd.DataFrame, dict]:
    """
    Collapse identical text to one frame unit before any sampling happens.

    This is the order that matters. Duplicates in this corpus are reposts, and
    a repost inherits the original's impression count, so duplicate groups sit
    overwhelmingly in the upper tail of engagement. Deduplicating a sample
    after it is drawn therefore removes the high-engagement content the design
    exists to capture, and leaves a realized sample smaller than its target by
    an amount that varies by stratum. Deduplicating the frame first keeps every
    quota met and every inclusion probability known.

    The surviving row is the highest-engagement member of its group. The size
    of the group is retained as `duplicate_count`, so repost volume is still
    measurable and can be carried into estimation as a cluster size.
    """
    col = resolve_column(df, frame.text_column)
    if col is None:
        return df.assign(duplicate_count=1), {
            "deduplicated": False, "reason": f"no '{frame.text_column}' column"}

    out = df.copy()
    before = len(out)
    blank_text = out[col].astype(str).str.strip().eq("") | out[col].isna()
    n_blank = int(blank_text.sum())
    if frame.drop_missing_text:
        out = out[~blank_text].copy()

    out["_content_key"] = _content_key(out[col], frame.dedup_normalise)
    group = out.groupby("_content_key", sort=False)
    out["duplicate_count"] = group["_content_key"].transform("size")

    out = (out.sort_values(sort_measure, ascending=False, kind="mergesort")
              .drop_duplicates(subset="_content_key", keep="first")
              .sort_index())

    return out, {
        "deduplicated": True,
        "rows_in": before,
        "blank_text_dropped": n_blank,
        "duplicates_collapsed": before - n_blank - len(out),
        "rows_out": len(out),
        "max_group": int(out["duplicate_count"].max()) if len(out) else 0,
        "pct_collapsed": round((before - n_blank - len(out)) / max(before, 1) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_frame(cfg: StudyConfig, input_dir: Path, cache_dir: Path,
                refresh: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    """Build one frame per dataset, plus the funnel that documents the losses."""
    files = discover_files(input_dir, cfg.datasets)
    if not files:
        raise FileNotFoundError(
            f"No export in {input_dir} matches any of: {', '.join(cfg.datasets.values())}")

    frames: dict[str, pd.DataFrame] = {}
    funnel_rows: list[dict] = []
    all_missing: list[str] = []

    for key in [k for k in cfg.dataset_order if k in files]:
        raw = load_raw(key, files[key], cache_dir, refresh)
        row = {"dataset": key, "raw_rows": len(raw)}
        df = raw

        for spec in cfg.frame.filters:
            df, info = apply_filter(df, spec)
            row[f"after_{spec.name}"] = info["kept"]
            row[f"blank_{spec.name}"] = info["blank"]

        df, missing = add_engagement(df, cfg.engagement)
        all_missing += [f"{key}: {m}" for m in missing]

        df, unparsed = add_period(df, cfg.frame)
        row["unparsed_dates"] = unparsed

        if cfg.frame.deduplicate:
            df, dedup = deduplicate_frame(df, cfg.frame, cfg.engagement.total_name)
            row["blank_text_dropped"] = dedup.get("blank_text_dropped", 0)
            row["duplicates_collapsed"] = dedup.get("duplicates_collapsed", 0)
        else:
            df = df.assign(duplicate_count=1)
            row["blank_text_dropped"] = 0
            row["duplicates_collapsed"] = 0

        row["frame_rows"] = len(df)
        row["first_period"] = df["period"].min() if len(df) else ""
        row["last_period"] = df["period"].max() if len(df) else ""
        funnel_rows.append(row)
        frames[key] = df

        log(f"{key:<14} {len(raw):>8,} raw -> {len(df):>8,} frame "
            f"({row['duplicates_collapsed']:,} duplicates collapsed)")

    return frames, pd.DataFrame(funnel_rows), all_missing
