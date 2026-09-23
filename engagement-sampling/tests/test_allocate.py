"""Quotas must sum to the target and never exceed a stratum's population."""
import unittest

import numpy as np
import pandas as pd

from _support import study  # noqa: F401  (path setup)
from engagement_sampling.allocate import (_largest_remainder, allocate_quotas,
                                          period_sample_sizes, validate_plan)
from engagement_sampling.config import DesignSpec


def counts():
    return pd.DataFrame([
        {"dataset": "a", "period": "2026-01", "count": 800, "sd": 10.0, "measure_total": 1.0},
        {"dataset": "b", "period": "2026-01", "count": 200, "sd": 90.0, "measure_total": 1.0},
        {"dataset": "a", "period": "2026-02", "count": 50_000, "sd": 10.0, "measure_total": 1.0},
    ])


class TestLargestRemainder(unittest.TestCase):

    def test_hits_the_target_exactly(self):
        raw = np.array([3.4, 2.5, 4.1])
        out = _largest_remainder(raw, 10, np.array([100, 100, 100]))
        self.assertEqual(out.sum(), 10)
        self.assertTrue((out >= np.floor(raw)).all())

    def test_never_exceeds_capacity(self):
        out = _largest_remainder(np.array([8.0, 2.0]), 10, np.array([3, 100]))
        self.assertEqual(out.sum(), 10)
        self.assertLessEqual(out[0], 3)

    def test_stops_when_every_stratum_is_full(self):
        out = _largest_remainder(np.array([5.0, 5.0]), 10, np.array([2, 3]))
        self.assertEqual(out.sum(), 5)


class TestAllocation(unittest.TestCase):

    def test_cap_lowers_the_rate_and_raises_the_weight(self):
        sizes = period_sample_sizes(counts(), DesignSpec(rate=0.10, cap=2000))
        jan = sizes[sizes.period == "2026-01"].iloc[0]
        feb = sizes[sizes.period == "2026-02"].iloc[0]
        self.assertEqual(jan.target, 100)
        self.assertFalse(bool(jan.capped))
        self.assertEqual(feb.target, 2000)
        self.assertTrue(bool(feb.capped))
        self.assertLess(feb.realized_rate, jan.realized_rate)
        self.assertGreater(feb.period_weight, jan.period_weight)

    def test_quotas_sum_to_the_period_target(self):
        c, design = counts(), DesignSpec(rate=0.10, cap=2000)
        plan = allocate_quotas(c, period_sample_sizes(c, design), design)
        for _, g in plan.groupby("period"):
            self.assertEqual(g["quota"].sum(), g["target"].iloc[0])
        self.assertEqual(validate_plan(plan), [])

    def test_neyman_moves_budget_to_the_more_variable_stratum(self):
        c = counts()
        prop = DesignSpec(rate=0.10, cap=2000, allocation="proportional")
        ney = DesignSpec(rate=0.10, cap=2000, allocation="neyman")
        p = allocate_quotas(c, period_sample_sizes(c, prop), prop)
        n = allocate_quotas(c, period_sample_sizes(c, ney), ney)
        pick = lambda d: d[(d.period == "2026-01") & (d.dataset == "b")]["quota"].iloc[0]
        self.assertGreater(pick(n), pick(p))

    def test_quota_never_exceeds_the_population(self):
        c = pd.DataFrame([
            {"dataset": "a", "period": "2026-01", "count": 5, "sd": 1.0, "measure_total": 1.0},
            {"dataset": "b", "period": "2026-01", "count": 5000, "sd": 1.0, "measure_total": 1.0},
        ])
        design = DesignSpec(rate=0.99, cap=None)
        plan = allocate_quotas(c, period_sample_sizes(c, design), design)
        self.assertTrue((plan["quota"] <= plan["count"]).all())
        self.assertEqual(validate_plan(plan), [])


if __name__ == "__main__":
    unittest.main()
