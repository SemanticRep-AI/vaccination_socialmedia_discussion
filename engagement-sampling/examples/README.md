# Examples

### `reference_run/`

Aggregate outputs measured by running both pipelines on the source exports: monthly
counts, quota plans, correlations, percentiles, duplication rates, design effects.
Every figure quoted in `docs/comparison.html` and `docs/visualizations.html` is
reproducible from these files, and `scripts/make_visualizations.py` rebuilds the
charts from them.

These are aggregates only — counts and statistics, at most 64 rows per file. No post
text, no URLs, no author names.

### `example_sample_head.csv`

The shape of a drawn sample: 150 rows showing every column the pipeline attaches,
including `inclusion_prob`, `design_weight`, `cluster_weight`, `selection_phase` and
`duplicate_count`.

**Generated from the synthetic corpus in `tests/_support.py`, not from real data.**
The posts are fabricated. It exists to document the output schema, so it can live in
a public repository without redistributing anyone's content.

### `example_report.html`

A run report, for reference. Contains aggregate tables and charts only.

---

## A note on committing data

`data/`, `cache/` and `output/` are git-ignored, so source exports and drawn samples
stay out of the repository by default. Keep it that way: platform terms of service
generally prohibit redistributing post content in bulk, and a drawn sample from a
study like this carries post text and author handles for identifiable people. Share
post IDs if you need to make a sample reproducible for reviewers.
