"""Frame construction: filters, engagement, and the order of deduplication."""
import unittest

import numpy as np
import pandas as pd

from _support import engagement_spec, study
from engagement_sampling.config import FilterSpec, FrameSpec
from engagement_sampling.frame import (add_engagement, add_period, apply_filter,
                                       deduplicate_frame, resolve_column)


class TestColumnResolution(unittest.TestCase):

    def test_finds_an_exact_name(self):
        df = pd.DataFrame({"a.b_c": [1]})
        self.assertEqual(resolve_column(df, "a.b_c"), "a.b_c")

    def test_tolerates_case_and_punctuation_drift(self):
        df = pd.DataFrame({"A.B-C": [1]})
        self.assertEqual(resolve_column(df, "a_b_c"), "A.B-C")

    def test_falls_back_to_a_declared_alias(self):
        df = pd.DataFrame({"impressions": [1]})
        self.assertEqual(resolve_column(df, "impression",
                                        {"impression": ["impressions"]}), "impressions")

    def test_returns_none_when_absent(self):
        self.assertIsNone(resolve_column(pd.DataFrame({"x": [1]}), "y"))


class TestFilters(unittest.TestCase):

    def test_matches_any_of_the_interchangeable_columns(self):
        df = pd.DataFrame({"country": ["United States", "Canada", ""],
                           "code": ["", "ca", "us"]})
        spec = FilterSpec("country", ["country", "code"], ["united states", "us"])
        kept, info = apply_filter(df, spec)
        self.assertEqual(len(kept), 2)
        self.assertEqual(info["dropped"], 1)

    def test_blank_rows_are_dropped_unless_allowed(self):
        df = pd.DataFrame({"country": ["us", "", ""]})
        strict = FilterSpec("country", ["country"], ["us"], allow_blank=False)
        loose = FilterSpec("country", ["country"], ["us"], allow_blank=True)
        self.assertEqual(len(apply_filter(df, strict)[0]), 1)
        self.assertEqual(len(apply_filter(df, loose)[0]), 3)

    def test_raises_when_no_listed_column_exists(self):
        with self.assertRaisesRegex(KeyError, "language"):
            apply_filter(pd.DataFrame({"x": [1]}),
                         FilterSpec("language", ["lang"], ["en"]))


class TestEngagement(unittest.TestCase):

    def test_applies_the_declared_weights(self):
        df = pd.DataFrame({"a_likes": ["10"], "a_views": ["100"], "b_shares": ["5"],
                           "engagement": ["7"]})
        out, missing = add_engagement(df, engagement_spec())
        self.assertEqual(missing, [])
        self.assertAlmostEqual(out["alpha_engagement"].iloc[0], 10 + 100 * 0.1)
        self.assertAlmostEqual(out["total_engagement"].iloc[0], 25.0)
        self.assertAlmostEqual(out["source_engagement"].iloc[0], 7.0)

    def test_a_missing_source_column_is_zero_and_reported(self):
        df = pd.DataFrame({"a_likes": ["10"], "engagement": ["0"]})
        out, missing = add_engagement(df, engagement_spec())
        self.assertAlmostEqual(out["alpha_engagement"].iloc[0], 10.0)
        self.assertAlmostEqual(out["beta_engagement"].iloc[0], 0.0)
        self.assertTrue(any("b_shares" in m for m in missing))

    def test_non_numeric_values_become_zero(self):
        df = pd.DataFrame({"a_likes": ["", "None", "abc", "4"], "a_views": ["0"] * 4,
                           "b_shares": ["0"] * 4, "engagement": ["0"] * 4})
        out, _ = add_engagement(df, engagement_spec())
        self.assertEqual(list(out["alpha_engagement"]), [0.0, 0.0, 0.0, 4.0])


class TestDeduplication(unittest.TestCase):

    def frame(self):
        return pd.DataFrame({
            "content": ["viral", "VIRAL  ", "viral", "unique a", "unique b", ""],
            "total_engagement": [10.0, 900.0, 50.0, 5.0, 7.0, 3.0],
        })

    def test_collapses_normalised_duplicates(self):
        out, info = deduplicate_frame(self.frame(), FrameSpec(), "total_engagement")
        self.assertEqual(len(out), 3)
        self.assertEqual(info["duplicates_collapsed"], 2)
        self.assertEqual(info["blank_text_dropped"], 1)

    def test_keeps_the_highest_engagement_member_of_a_group(self):
        out, _ = deduplicate_frame(self.frame(), FrameSpec(), "total_engagement")
        self.assertEqual(out["total_engagement"].max(), 900.0)
        self.assertIn(900.0, list(out["total_engagement"]))

    def test_group_size_survives_as_duplicate_count(self):
        out, _ = deduplicate_frame(self.frame(), FrameSpec(), "total_engagement")
        kept = out[out["total_engagement"] == 900.0].iloc[0]
        self.assertEqual(kept["duplicate_count"], 3)

    def test_case_sensitivity_can_be_turned_off(self):
        spec = FrameSpec(dedup_normalise=False)
        out, _ = deduplicate_frame(self.frame(), spec, "total_engagement")
        self.assertEqual(len(out), 4)          # "VIRAL  " is now its own text

    def test_deduplicating_the_frame_preserves_the_engagement_mass(self):
        """
        The property that the alternative ordering breaks. Collapsing duplicates
        before the draw keeps the highest-engagement member of each group, so
        the tail of the distribution survives into the frame.
        """
        df = self.frame()
        before_max = df["total_engagement"].max()
        out, _ = deduplicate_frame(df, FrameSpec(), "total_engagement")
        self.assertEqual(out["total_engagement"].max(), before_max)


class TestPeriods(unittest.TestCase):

    def test_parses_mixed_timestamp_formats(self):
        df = pd.DataFrame({"published": ["2026-01-02 03:04:05",
                                         "2026-02-03 04:05:06.789000"]})
        out, unparsed = add_period(df, FrameSpec())
        self.assertEqual(unparsed, 0)
        self.assertEqual(list(out["period"]), ["2026-01", "2026-02"])

    def test_unreadable_dates_are_counted_and_removed(self):
        df = pd.DataFrame({"published": ["2026-01-02", "not a date"]})
        out, unparsed = add_period(df, FrameSpec())
        self.assertEqual(unparsed, 1)
        self.assertEqual(len(out), 1)

    def test_period_granularity_is_configurable(self):
        df = pd.DataFrame({"published": ["2026-01-02", "2026-02-03"]})
        out, _ = add_period(df, FrameSpec(period="Q"))
        self.assertEqual(out["period"].iloc[0], "2026Q1")

    def test_raises_when_the_date_column_is_absent(self):
        with self.assertRaisesRegex(KeyError, "published"):
            add_period(pd.DataFrame({"x": [1]}), FrameSpec())


if __name__ == "__main__":
    unittest.main()
