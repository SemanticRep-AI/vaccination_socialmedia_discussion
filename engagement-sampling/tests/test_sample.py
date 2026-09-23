"""
The sampling step is where the statistics live, so these tests check the
properties an estimate depends on rather than the shape of the output.
"""
import unittest

import numpy as np
import pandas as pd

from _support import synthetic_corpus  # noqa: F401  (path setup)
from engagement_sampling.sample import (sample_stratum, stratum_seed, top_indices)


def stratum(n=1000, zeros=0.9, seed=0):
    """A stratum with a long tail and a large block of ties at zero."""
    rng = np.random.default_rng(seed)
    v = np.where(rng.random(n) < zeros, 0.0, rng.pareto(1.3, n) * 100)
    return pd.DataFrame({"measure": v, "other": rng.random(n) * v,
                         "row_order": np.arange(n)})


class TestTieHandling(unittest.TestCase):

    def test_takes_exactly_k_rows(self):
        s = stratum()
        idx, _ = top_indices(s, "measure", 50, seed=1)
        self.assertEqual(len(idx), 50)

    def test_every_value_above_the_cutoff_is_taken(self):
        s = stratum()
        idx, info = top_indices(s, "measure", 200, seed=1)
        above = s[s["measure"] > info["cutoff"]]
        self.assertTrue(set(above.index).issubset(set(idx)))

    def test_ties_are_broken_at_random_not_by_row_order(self):
        """
        With 90% of the stratum tied at zero, taking the first k in row order
        would select the lowest row numbers every time. A seeded tie-break
        should instead spread the tied picks across the stratum, and two seeds
        should disagree about which tied rows are taken.
        """
        s = stratum(n=1000, zeros=0.9)
        a, info = top_indices(s, "measure", 300, seed=1)
        b, _ = top_indices(s, "measure", 300, seed=2)
        self.assertGreater(info["tied_at_cutoff"], 100)
        self.assertNotEqual(set(a), set(b))
        tied_rows = s.loc[list(set(a) & set(s[s["measure"] == info["cutoff"]].index)),
                          "row_order"]
        self.assertGreater(tied_rows.mean(), len(s) * 0.25)

    def test_the_same_seed_reproduces_the_same_rows(self):
        s = stratum()
        self.assertEqual(list(top_indices(s, "measure", 100, seed=9)[0]),
                         list(top_indices(s, "measure", 100, seed=9)[0]))


class TestStratumSeed(unittest.TestCase):

    def test_is_stable_and_distinct_per_stratum(self):
        self.assertEqual(stratum_seed(42, "a", "2026-01"), stratum_seed(42, "a", "2026-01"))
        self.assertNotEqual(stratum_seed(42, "a", "2026-01"), stratum_seed(42, "a", "2026-02"))
        self.assertNotEqual(stratum_seed(42, "a", "2026-01"), stratum_seed(43, "a", "2026-01"))


class TestSampleStratum(unittest.TestCase):

    def draw(self, n=1000, quota=100, reserve=0.2, top_pct=0.05):
        return sample_stratum(stratum(n), quota, ["measure", "other"],
                              top_pct, seed=3, remainder_reserve=reserve)

    def test_draws_exactly_the_quota(self):
        sel, info = self.draw()
        self.assertEqual(len(sel), 100)
        self.assertEqual(info["drawn"], 100)

    def test_every_inclusion_probability_is_a_probability(self):
        sel, _ = self.draw()
        self.assertTrue((sel["inclusion_prob"] > 0).all())
        self.assertTrue((sel["inclusion_prob"] <= 1).all())

    def test_weights_recover_the_stratum_size(self):
        """The property that makes Horvitz-Thompson estimation valid."""
        sel, _ = self.draw(n=1000, quota=100)
        self.assertAlmostEqual(sel["design_weight"].sum(), 1000, delta=1000 * 0.05)

    def test_reserve_keeps_the_remainder_reachable(self):
        """
        With a priority pool larger than the quota and no reserve, every unit
        outside the pool has probability zero and the sample cannot describe
        the stratum. The reserve is what prevents that.
        """
        without, _ = self.draw(quota=60, reserve=0.0, top_pct=0.10)
        with_reserve, _ = self.draw(quota=60, reserve=0.2, top_pct=0.10)
        self.assertEqual((without["selection_phase"] == "remainder").sum(), 0)
        self.assertGreater((with_reserve["selection_phase"] == "remainder").sum(), 0)
        self.assertLess(abs(with_reserve["design_weight"].sum() - 1000),
                        abs(without["design_weight"].sum() - 1000))

    def test_a_quota_at_or_above_the_population_is_a_census(self):
        sel, _ = sample_stratum(stratum(50), 50, ["measure"], 0.05, seed=1)
        self.assertEqual(len(sel), 50)
        self.assertTrue((sel["inclusion_prob"] == 1.0).all())

    def test_priority_units_outrank_the_remainder_on_the_measure(self):
        sel, _ = self.draw()
        p = sel[sel.selection_phase == "priority"]["measure"]
        r = sel[sel.selection_phase == "remainder"]["measure"]
        if len(r):
            self.assertGreater(p.mean(), r.mean())

    def test_an_empty_stratum_returns_nothing_rather_than_raising(self):
        sel, info = sample_stratum(stratum(0), 10, ["measure"], 0.05, seed=1)
        self.assertEqual(len(sel), 0)
        self.assertEqual(info["drawn"], 0)

    def test_cluster_weight_accounts_for_collapsed_duplicates(self):
        s = stratum(500)
        s["duplicate_count"] = 3
        sel, _ = sample_stratum(s, 50, ["measure"], 0.05, seed=1)
        self.assertTrue(np.allclose(sel["cluster_weight"],
                                    sel["design_weight"] * 3))


if __name__ == "__main__":
    unittest.main()
