from __future__ import annotations

import unittest

from evaluation.analyze_pointgoal_benchmark import (
    _exact_mcnemar_two_sided,
    describe,
    percentile,
    wilson_interval,
)


class AnalyzePointGoalBenchmarkTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(percentile([3.0], 0.90), 3.0)

    def test_describe_reports_expected_distribution(self) -> None:
        result = describe([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["p25"], 1.75)
        self.assertEqual(result["p75"], 3.25)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(198, 200)

        self.assertLess(lower, 0.99)
        self.assertGreater(upper, 0.99)

    def test_exact_mcnemar_handles_two_one_sided_discordances(self) -> None:
        self.assertEqual(_exact_mcnemar_two_sided(2, 0), 0.5)
        self.assertEqual(_exact_mcnemar_two_sided(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
