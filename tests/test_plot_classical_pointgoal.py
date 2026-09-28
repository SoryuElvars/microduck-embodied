from __future__ import annotations

import unittest

from evaluation.plot_classical_pointgoal import paired_metric_values


class PlotClassicalPointGoalTests(unittest.TestCase):
    def test_paired_values_exclude_timeout_only_from_one_controller(self) -> None:
        summary = {
            "controllers": {
                "naive_p": {
                    "episodes": [
                        {"episode_id": "e0", "success": True, "metric": 1.0},
                        {"episode_id": "e1", "success": False, "metric": 2.0},
                    ]
                },
                "constrained": {
                    "episodes": [
                        {"episode_id": "e0", "success": True, "metric": 3.0},
                        {"episode_id": "e1", "success": True, "metric": 4.0},
                    ]
                },
            }
        }

        naive, constrained = paired_metric_values(summary, "metric")

        self.assertEqual(naive, [1.0])
        self.assertEqual(constrained, [3.0])


if __name__ == "__main__":
    unittest.main()
