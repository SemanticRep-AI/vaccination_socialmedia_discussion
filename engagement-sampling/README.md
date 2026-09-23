# engagement-sampling

Design-based sampling for large social media corpora.

Builds a sampling frame from platform exports, derives engagement measures from a
declared formula, allocates a stratified sample across time periods and datasets,
and draws it while recording the inclusion probability of every selected unit — so
the resulting sample supports population estimation, not only description.

Written for a content-analysis study of vaccine discourse across four search
profiles, but nothing in the code is specific to that study. Datasets, columns,
weights, filters, period granularity and design parameters are all declared in a
configuration file.

---

## Quick start

```bash
pip install -r requirements.txt
cp /path/to/exports/*.xlsx data/
python run.py --config config/vaccine_2026.yaml
```

The run prints its progress, writes everything to `output/`, and opens
`output/report.html`. The first run parses the workbooks and takes a few minutes;
later runs read the cache in `cache/` and take about twenty seconds.

```bash
python run.py --config config/study.yaml --rate 0.05      # override a design parameter
python run.py --config config/study.yaml --refresh        # re-read the exports
python run.py --config config/study.yaml --no-open        # for servers and CI
```

---

## What it produces

| Path | Contents |
|---|---|
| `output/report.html` | Every table and chart in one self-contained page |
| `output/03_sample/sample.csv.gz` | The sample, with `inclusion_prob` and `design_weight` per unit |
| `output/04_coding/coding_<period>.csv` | One file per period for human coders, deduplicated across datasets |
| `output/02_plan/quota_plan.csv` | Quota per dataset per period, with the realised sampling rate |
| `output/01_diagnostics/` | Descriptives, correlations, platform contribution, concentration, measure overlap |
| `output/00_frame/funnel.csv` | Row counts at each filter stage — the audit trail |
| `output/00_frame/config_used.json` | The exact configuration that produced the run |
| `output/*/validation.txt` | What the plan and sample validators checked, and found |

---

## The statistical design

The sample is a **two-phase stratified draw**. Each dataset × period stratum is split
into a *priority pool* — the top `top_pct` of any of the priority measures — and a
*remainder pool*. Units are taken from the priority pool first, and a reserved share
of each quota is drawn at random from the remainder.

Four properties follow from that, and the package enforces all four.

**Every unit has a known, positive inclusion probability.** The priority phase is a
census or a simple random sample of the pool; the remainder phase is a simple random
sample of everything else. `inclusion_prob` and `design_weight = 1/π` are written on
every selected row. Weighted, the sample reproduces the frame — in the reference
study, 81,264 against an actual 81,266.

**Nothing is unreachable.** `remainder_reserve` guarantees a slice of every quota goes
to units outside the priority pool. Without it, a stratum whose pool exceeds its quota
is sampled entirely from the pool, everything else has π = 0, and the stratum becomes
inestimable. Reserving 20% took the pooled design effect from 39.4 to 4.05 and the
effective sample size from about 200 to 2,008.

**Ties are broken at random, not by row order.** These measures are heavily
zero-inflated, so a top-percentile cutoff is often shared by hundreds of units. Taking
the first *k* in file order selects on position — which, in a chronologically ordered
export, means recency. Ties are broken with a seed derived from the master seed, which
keeps tied units exchangeable and the run reproducible.

**Duplicates are collapsed before the draw, not after.** Reposts carry the original's
impression count, so duplicate groups sit in the upper tail. Removing them after
sampling deletes the high-engagement content the design exists to capture — measured
on the reference study, 82.9% of the engagement mass. Collapsing them into the frame
keeps quotas attainable and retains group size as `duplicate_count`.

Reported alongside the sample: design effect and effective sample size (Kish),
coverage of both units and engagement mass, and Horvitz–Thompson totals with standard
errors via `diagnostics.weighted_total`.

---

## Configuration

One file describes a study completely. No module contains a dataset name, a column
name or a weight.

```yaml
name: My Study
datasets:
  treatment: treatment_export
  control:   control_export

frame:
  period: M                 # M, W, Q — any pandas period alias
  deduplicate: true
  filters:
    - name: language
      columns: [lang]       # interchangeable encodings; a row matching any is kept
      keep: [en]

engagement:
  total_name: content_engagement
  measures:
    - name: x_engagement
      components:
        article_extended_attributes.twitter_likes: 1.0
        article_extended_attributes.twitter_impressions: 0.1   # passive metric

design:
  rate: 0.10                # target = min(rate × period total, cap)
  cap: 2000
  allocation: proportional  # or neyman
  priority_measures: [content_engagement, source_engagement]
  top_pct: 0.05
  remainder_reserve: 0.20
  seed: 42
```

`config/vaccine_2026.yaml` is a complete worked example with comments.

---

## Layout

```
engagement-sampling/
├── run.py                        convenience entry point
├── config/vaccine_2026.yaml      the worked example study
├── src/engagement_sampling/
│   ├── config.py                 study definition and validation
│   ├── frame.py                  load, filter, score, deduplicate
│   ├── allocate.py               sample sizes and quota allocation
│   ├── sample.py                 the draw, and inclusion probabilities
│   ├── diagnostics.py            frame and sample statistics
│   ├── charts.py                 chart builders
│   ├── report.py                 the HTML report
│   ├── pipeline.py               end-to-end orchestration
│   └── cli.py                    command line interface
├── tests/                        55 tests, stdlib unittest
├── scripts/make_visualizations.py rebuilds docs/visualizations.html
├── docs/
│   ├── comparison.html           two pipelines compared, with measurements
│   ├── visualizations.html       13 charts covering the design
│   └── methodology.md            the design written out
├── examples/reference_run/       measured outputs both pipelines produced
├── data/                         put exports here (git-ignored)
├── cache/                        parsed exports (git-ignored)
└── output/                       results (git-ignored)
```

---

## Tests

```bash
python -m unittest discover -s tests -t tests     # stdlib, no dependencies
pytest tests                                       # also works
```

55 tests covering configuration validation, largest-remainder allocation, tie
handling, inclusion probabilities, deduplication order, and a full run from two CSV
exports to a validated sample — including that the same seed reproduces the same
sample and a different seed does not.

---

## Requirements

Python 3.9+, `pandas`, `numpy`, `openpyxl`, `matplotlib`. `PyYAML` only if the config
is YAML rather than JSON.

## Licence

MIT. See `LICENSE`.
