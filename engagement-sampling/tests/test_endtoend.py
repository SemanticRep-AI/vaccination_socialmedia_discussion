"""A whole study, from two CSV exports to a validated sample."""
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from _support import study, synthetic_corpus, write_corpus
from engagement_sampling.pipeline import run_study


class TestEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        write_corpus(cls.tmp / "data", synthetic_corpus())
        cfg = study()
        cfg = replace(cfg, design=replace(cfg.design, rate=0.20, cap=None))
        cls.result = run_study(cfg, cls.tmp / "data", cls.tmp / "out",
                               cls.tmp / "cache", write_coding_files=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_run_reports_no_validation_problems(self):
        self.assertEqual(self.result["plan_issues"], [])
        self.assertEqual(self.result["sample_issues"], [])

    def test_every_quota_is_met(self):
        drawn = self.result["sample"].groupby(["period", "dataset"]).size()
        for _, row in self.result["plan"].iterrows():
            self.assertEqual(int(drawn.get((row["period"], row["dataset"]), 0)),
                             int(row["quota"]))

    def test_the_language_filter_was_applied(self):
        for df in self.result["frames"].values():
            self.assertTrue((df["lang"] == "en").all())

    def test_the_frame_holds_no_duplicate_text(self):
        for df in self.result["frames"].values():
            self.assertFalse(df["_content_key"].duplicated().any())

    def test_weights_recover_the_frame_size(self):
        frame_n = sum(len(f) for f in self.result["frames"].values())
        estimate = self.result["sample"]["design_weight"].sum()
        self.assertLess(abs(estimate - frame_n) / frame_n, 0.10)

    def test_the_sample_is_reproducible_from_the_seed(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            write_corpus(tmp / "data", synthetic_corpus())
            cfg = study()
            cfg = replace(cfg, design=replace(cfg.design, rate=0.20, cap=None))
            again = run_study(cfg, tmp / "data", tmp / "out", tmp / "cache",
                              write_coding_files=False)
            self.assertEqual(sorted(again["sample"]["url"]),
                             sorted(self.result["sample"]["url"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_different_seed_draws_a_different_sample(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            write_corpus(tmp / "data", synthetic_corpus())
            cfg = study()
            cfg = replace(cfg, design=replace(cfg.design, rate=0.20, cap=None, seed=999))
            other = run_study(cfg, tmp / "data", tmp / "out", tmp / "cache",
                              write_coding_files=False)
            self.assertNotEqual(sorted(other["sample"]["url"]),
                                sorted(self.result["sample"]["url"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_it_writes_the_files_a_run_is_judged_by(self):
        out = self.tmp / "out"
        for path in ["00_frame/funnel.csv", "00_frame/config_used.json",
                     "01_diagnostics/measure_descriptives.csv",
                     "02_plan/quota_plan.csv", "02_plan/validation.txt",
                     "03_sample/sample.csv.gz", "03_sample/sampling_summary.csv",
                     "03_sample/validation.txt"]:
            with self.subTest(path=path):
                self.assertTrue((out / path).exists(), path)

    def test_coder_files_contain_each_text_once(self):
        files = sorted((self.tmp / "out" / "04_coding").glob("coding_*.csv"))
        self.assertTrue(files)
        texts = pd.concat([pd.read_csv(f) for f in files])["content"]
        self.assertEqual(len(texts), texts.nunique())

    def test_the_sample_carries_the_columns_estimation_needs(self):
        sample = pd.read_csv(self.tmp / "out" / "03_sample" / "sample.csv.gz")
        for col in ["inclusion_prob", "design_weight", "cluster_weight",
                    "selection_phase", "dataset", "period"]:
            self.assertIn(col, sample.columns)

    def test_the_report_renders(self):
        from engagement_sampling.report import build_report
        path = build_report(self.result, self.tmp / "out" / "report.html")
        html = path.read_text()
        self.assertTrue(path.exists())
        self.assertIn("<h1>", html)
        self.assertIn("data:image/png;base64,", html)


if __name__ == "__main__":
    unittest.main()
