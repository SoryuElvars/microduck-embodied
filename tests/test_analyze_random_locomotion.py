from __future__ import annotations

import unittest

from evaluation.analyze_random_locomotion import (
    episode_passes,
    linear_response,
    normalized_tracking_score,
    success_tolerances,
)


class RandomLocomotionAnalysisTests(unittest.TestCase):
    def test_success_requires_all_axes_and_no_fall(self) -> None:
        command = {"vx": 0.2, "vy": -0.1, "wz": 0.5}
        tolerances = success_tolerances({"command": command})
        passing = {
            "fallen": False,
            "command": command,
            "rmse_vx": tolerances["vx"],
            "rmse_vy": tolerances["vy"],
            "rmse_wz": tolerances["wz"],
        }
        self.assertTrue(episode_passes(passing))

        failing = dict(passing)
        failing["rmse_wz"] += 0.001
        self.assertFalse(episode_passes(failing))

        fallen = dict(passing)
        fallen["fallen"] = True
        self.assertFalse(episode_passes(fallen))

    def test_linear_response_recovers_known_mapping(self) -> None:
        episodes = [
            {
                "command": {"vx": target},
                "mean_actual_vx": 0.5 * target + 0.1,
            }
            for target in (-1.0, 0.0, 1.0)
        ]
        result = linear_response(episodes, "vx", "mean_actual_vx")

        self.assertAlmostEqual(result["slope"], 0.5)
        self.assertAlmostEqual(result["intercept"], 0.1)
        self.assertAlmostEqual(result["correlation"], 1.0)

    def test_normalized_tracking_score_is_one_at_thresholds(self) -> None:
        command = {"vx": 0.2, "vy": -0.1, "wz": 0.5}
        tolerances = success_tolerances({"command": command})
        episode = {
            "command": command,
            "rmse_vx": tolerances["vx"],
            "rmse_vy": tolerances["vy"],
            "rmse_wz": tolerances["wz"],
        }

        self.assertAlmostEqual(normalized_tracking_score(episode), 1.0)

    def test_ignored_small_command_is_not_a_success(self) -> None:
        ignored = {
            "fallen": False,
            "command": {"vx": 0.05, "vy": 0.0, "wz": 0.0},
            "rmse_vx": 0.05,
            "rmse_vy": 0.0,
            "rmse_wz": 0.0,
        }

        self.assertFalse(episode_passes(ignored))


if __name__ == "__main__":
    unittest.main()
