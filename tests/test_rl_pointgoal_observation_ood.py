from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.rl_pointgoal_observation_ood import PROJECT_ROOT, load_protocol, select_specs


class ObservationOodProtocolTests(unittest.TestCase):
    def _protocol(self, filename: str):
        return load_protocol(PROJECT_ROOT / "evaluation" / "pointgoal_protocols" / filename)

    def test_yaw_protocol_is_symmetric_and_quick_is_frozen(self) -> None:
        protocol = self._protocol("week05_rl_yaw_bias_ood_v1.json")
        self.assertEqual(protocol.factor, "robot_yaw_bias")
        self.assertEqual(len(protocol.conditions), 5)
        self.assertEqual(len(select_specs(protocol, "quick")), 10)

    def test_position_and_goal_protocols_cover_both_axes(self) -> None:
        robot = self._protocol("week05_rl_robot_position_bias_ood_v1.json")
        goal = self._protocol("week05_rl_goal_position_bias_ood_v1.json")
        self.assertEqual(len(robot.conditions), 9)
        self.assertEqual(len(goal.conditions), 9)
        self.assertEqual(robot.factor, "robot_position_bias")
        self.assertEqual(goal.factor, "goal_position_bias")

    def test_dropout_is_high_level_and_has_two_nonzero_tiers(self) -> None:
        protocol = self._protocol("week05_rl_observation_dropout_ood_v1.json")
        self.assertEqual(protocol.factor, "robot_state_dropout")
        self.assertEqual(
            [item.dropout_probability for item in protocol.conditions], [0.0, 0.1, 0.2]
        )


if __name__ == "__main__":
    unittest.main()
