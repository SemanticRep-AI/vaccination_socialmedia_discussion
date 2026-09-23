"""The configuration should refuse a study it cannot run."""
import unittest
from dataclasses import replace

from _support import study
from engagement_sampling.config import StudyConfig


class TestStudyConfig(unittest.TestCase):

    def test_dataset_order_defaults_to_declaration_order(self):
        self.assertEqual(study().dataset_order, ["one", "two"])

    def test_rejects_priority_measure_that_cannot_be_produced(self):
        s = study()
        with self.assertRaisesRegex(ValueError, "priority_measures"):
            replace(s, design=replace(s.design, priority_measures=["nope"]))

    def test_rejects_out_of_range_design_values(self):
        s = study()
        for field, value in [("rate", 0.0), ("rate", 1.5), ("top_pct", 0.0),
                             ("top_pct", 2.0), ("remainder_reserve", 1.0),
                             ("remainder_reserve", -0.1), ("allocation", "magic")]:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    replace(s, design=replace(s.design, **{field: value}))

    def test_rejects_unknown_dataset_in_order(self):
        with self.assertRaisesRegex(ValueError, "dataset_order"):
            replace(study(), dataset_order=["one", "ghost"])

    def test_round_trips_through_a_dict(self):
        s = study()
        self.assertEqual(StudyConfig.from_dict(s.to_dict()).to_dict(), s.to_dict())


if __name__ == "__main__":
    unittest.main()
