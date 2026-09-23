# Methodology

How the sample is drawn, and why each step is the way it is.

---

## 1. The frame

The frame is the set of units the sample can be drawn from. Everything that changes
who is eligible happens here, and every exclusion is counted into
`output/00_frame/funnel.csv`.

**Filters.** Each filter names one or more interchangeable columns and a set of values
to keep. A row survives if any listed column matches. This handles the common case
where the same fact is encoded twice — a country name in one column and its ISO code
in another. Rows where every listed column is blank are dropped unless `allow_blank`
says otherwise, and the count of blanks is reported either way so the decision is
visible rather than buried.

**Engagement.** Each measure is a weighted sum of source columns declared in the
config. Passive metrics — views, impressions — carry a weight of 0.1 because they are
two to three orders of magnitude larger than deliberate actions such as a like or a
share, and at weight 1 they decide the total by themselves. A source column that is
absent is treated as zero and recorded in `missing_columns.txt`, so an export with a
slightly different schema still runs and still tells you what it could not find.

**Periods.** The publication timestamp is parsed and reduced to a period key. The
granularity is configurable; monthly is the default. Unparseable timestamps are
counted and excluded rather than silently coerced.

**Deduplication.** Identical text is collapsed to a single frame unit before anything
is drawn. The surviving row is the highest-engagement member of its group; the group
size is kept as `duplicate_count`.

The order matters more than it appears. In social media corpora duplicates are
reposts, and a repost inherits the original's impression count, so duplicate groups
sit overwhelmingly in the upper tail of engagement. Deduplicating a sample *after* it
is drawn therefore:

- leaves the realised sample smaller than its target, by an amount that varies
  unpredictably by stratum; and
- removes the high-engagement content the design was built to capture.

In the reference study, deduplicating after the draw removed 43% of the sample and
82.9% of its engagement mass, with removal rates of 75–88% in the top five deciles
against roughly 1% in the bottom four. Deduplicating the frame instead makes the same
fact a property of the population — 26% to 60% of each dataset is reposts — while
keeping every quota attainable and every inclusion probability known.

---

## 2. Sample size

For each period:

```
target = min(rate × period_total, cap)
```

bounded below by `floor` if set, and never above the period's own size.

The cap is what makes the design unequal-probability. A period whose uncapped target
exceeds the cap is sampled at a lower rate than one that is not, so its units each
stand for more of the population. The realised rate and its reciprocal are recorded
per period (`realized_rate`, `period_weight`) and carried into every unit's weight.

A design without a cap is self-weighting across periods and simpler to analyse. A cap
exists to bound coder workload, so it is a budget decision rather than a statistical
one — but it has statistical consequences, and they should be recorded rather than
absorbed.

---

## 3. Allocating the quota

Each period's target is split across the datasets present in it.

**Proportional** allocation gives each dataset a share equal to its share of the
period's units. Within a period this is self-weighting: every unit has the same
probability of selection regardless of dataset.

**Neyman** allocation gives each dataset a share proportional to `N_h × S_h`, where
`S_h` is the within-stratum standard deviation of the sort measure. This minimises the
variance of an estimated total for a fixed sample size, which matters when one dataset
is far more variable than another. It makes weights unequal *within* a period, which
is why weights are computed per stratum rather than per period.

**Rounding.** Real-valued allocations are floored, then the leftover units are handed
to the largest fractional parts until the period target is met exactly. A stratum
already at capacity is skipped and the unit passes to the next, so a period whose
largest dataset is small still reaches its target instead of quietly under-filling.

Before anything is drawn, `validate_plan` asserts that quotas sum to each period's
target, that no quota exceeds its stratum's population, and that every weight is
finite. Failures are written to `output/02_plan/validation.txt` and surfaced on the
report.

---

## 4. Drawing the sample

Within each dataset × period stratum:

1. Compute the priority pool: the top `top_pct` of **any** priority measure. Using
   more than one measure is a coverage decision — two measures that rank units
   differently will each surface content the other misses.
2. Split the quota. A share `remainder_reserve` is set aside for the remainder pool;
   the rest goes to the priority pool.
3. Draw each half, and record the inclusion probability.

| Case | Priority phase | Remainder phase |
|---|---|---|
| quota ≥ stratum | census, π = 1 | census, π = 1 |
| pool fits the priority half | all of it, π = 1 | SRS, π = r / \|R\| |
| pool exceeds the priority half | SRS of the pool, π = take / \|P\| | SRS, π = r / \|R\| |

**Why the reserve exists.** Draw only from the priority pool and every unit outside it
has π = 0. The sample then describes the pool and says nothing about the stratum, and
the weights will not reproduce the frame. Reserving a slice keeps every unit reachable.
In the reference study this took the pooled design effect from 39.4 to 4.05 and the
effective sample size from about 200 to 2,008.

**Ties.** Ranking uses a seeded random tie-break rather than row order. These measures
are heavily zero-inflated — the platform's own engagement column is zero for 86–94% of
posts — so the value at a top-percentile cutoff is often shared by hundreds of units.
Taking the first *k* in file order selects on position in the export, which in a
chronologically ordered file means recency. In the reference study 5.2% of top-5% slots
were filled this way overall and 15.3% in the most zero-inflated dataset. A random
tie-break also keeps tied units exchangeable, which is what the inclusion probabilities
above assume.

**Seeding.** Each stratum derives its seed from a hash of the master seed and its own
identity. One seed reused across strata couples their draws to row order; per-stratum
seeds keep them independent while leaving the whole run reproducible from one number.

After the draw, `validate_sample` asserts that every quota was met, that all
probabilities lie in (0, 1], that no text repeats within a dataset, and that the
weights reproduce each stratum's frame size to within 10%.

---

## 5. Estimation

Every selected unit carries:

| Column | Meaning |
|---|---|
| `inclusion_prob` | π, the probability this unit entered the sample |
| `design_weight` | 1/π — how many frame units this one stands for |
| `cluster_weight` | `design_weight × duplicate_count`, for corpus-level counts |
| `selection_phase` | `priority` or `remainder` |
| `priority_measures_hit` | how many priority measures selected it |

A population total is the Horvitz–Thompson estimator, `Σ wᵢyᵢ`, available as
`diagnostics.weighted_total` with a standard error. **An unweighted mean over this
sample describes the sample, not the corpus**: the design deliberately over-samples
the tail, so unweighted figures are biased upward for anything correlated with
engagement.

**Effective sample size.** Unequal weights cost precision. Kish's approximation,
`deff = 1 + CV²(w)`, gives an effective n of `n / deff` — the size of a simple random
sample that would estimate a mean about as precisely. That is the number to report in
a methods section, not the row count.

The design effect is also reported per phase, because the pooled figure mixes two very
different things: the priority phase is close to a census and costs almost nothing,
the remainder phase is a clean simple random sample. Quoting only the pooled number
makes a deliberately tail-heavy design look broken.

**Coverage.** The report states what share of the frame's units and what share of its
engagement mass the sample holds. A design that prioritises the tail captures few units
and most of the mass — in the reference study, 10% of units and 77–92% of engagement.
Reporting both makes that trade explicit rather than accidental.

---

## 6. Overlapping datasets

Where the datasets are search profiles rather than a partition, one post can belong to
several. This is not an error, but it has two consequences:

- A coder should not be shown the same text repeatedly. The files in `04_coding/` are
  deduplicated across datasets; the analytic sample keeps every dataset membership.
- A pooled estimate across datasets double-counts these units unless the union frame is
  used. `03_sample/cross_dataset_overlap.csv` reports how many units are affected.

---

## 7. Reproducibility

A run is fully determined by the config file and the master seed. The configuration
actually used is written to `output/00_frame/config_used.json` alongside the results,
so a set of outputs can always be traced to the settings that produced it. The test
suite asserts that the same seed reproduces the same sample and that a different seed
does not.
