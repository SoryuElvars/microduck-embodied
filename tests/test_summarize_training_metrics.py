import argparse
import unittest

from evaluation.summarize_training_metrics import (
    ScalarPoint,
    parse_series_arg,
    window_stats,
)


class TrainingMetricsSummaryTests(unittest.TestCase):
    def test_parse_series_keeps_resume_order(self) -> None:
        label, paths = parse_series_arg("candidate_a=/tmp/first,/tmp/resume")

        self.assertEqual(label, "candidate_a")
        self.assertEqual([str(path) for path in paths], ["/tmp/first", "/tmp/resume"])

    def test_parse_series_requires_label_and_run(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_series_arg("candidate_a")

    def test_window_stats_uses_inclusive_checkpoint(self) -> None:
        points = {
            step: ScalarPoint(step, float(step), 0.0, "/tmp/run")
            for step in range(5)
        }

        stats = window_stats(points, checkpoint=4, window_size=3)

        assert stats is not None
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["last"], 4.0)


if __name__ == "__main__":
    unittest.main()
