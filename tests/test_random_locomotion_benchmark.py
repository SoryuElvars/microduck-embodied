from __future__ import annotations

import unittest

from evaluation.random_locomotion_benchmark import (
    COMMAND_RANGES,
    TURN_MAGNITUDE_RANGE,
    build_command_plan,
)


class RandomCommandPlanTests(unittest.TestCase):
    def test_400_episode_plan_has_documented_bucket_counts(self) -> None:
        plan = build_command_plan(400, 20260911)

        self.assertEqual(len(plan), 400)
        self.assertEqual(sum(item.bucket == "standing" for item in plan), 100)
        self.assertEqual(sum(item.bucket == "turn_in_place" for item in plan), 60)
        self.assertEqual(sum(item.bucket == "general" for item in plan), 240)

    def test_plan_is_deterministic(self) -> None:
        self.assertEqual(
            build_command_plan(400, 20260911),
            build_command_plan(400, 20260911),
        )
        self.assertNotEqual(
            build_command_plan(400, 20260911),
            build_command_plan(400, 20260912),
        )

    def test_commands_respect_ranges_and_turn_bucket(self) -> None:
        plan = build_command_plan(400, 20260911)
        turns = [item for item in plan if item.bucket == "turn_in_place"]

        for item in plan:
            self.assertGreaterEqual(item.vx, COMMAND_RANGES["vx"][0])
            self.assertLessEqual(item.vx, COMMAND_RANGES["vx"][1])
            self.assertGreaterEqual(item.vy, COMMAND_RANGES["vy"][0])
            self.assertLessEqual(item.vy, COMMAND_RANGES["vy"][1])
            self.assertGreaterEqual(item.wz, COMMAND_RANGES["wz"][0])
            self.assertLessEqual(item.wz, COMMAND_RANGES["wz"][1])

        self.assertTrue(all(item.vx == item.vy == 0.0 for item in turns))
        self.assertTrue(
            all(
                TURN_MAGNITUDE_RANGE[0] <= abs(item.wz) <= TURN_MAGNITUDE_RANGE[1]
                for item in turns
            )
        )
        self.assertEqual(sum(item.wz > 0 for item in turns), 30)
        self.assertEqual(sum(item.wz < 0 for item in turns), 30)


if __name__ == "__main__":
    unittest.main()
