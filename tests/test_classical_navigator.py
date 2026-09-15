from __future__ import annotations

import math
import unittest

from navigation.classical_navigator import (
    ConstrainedGoToGoalConfig,
    ConstrainedGoToGoalNavigator,
    goal_error,
    wrap_angle,
)
from navigation.types import GoalState, RobotState


def robot_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    finite: bool = True,
) -> RobotState:
    return RobotState(
        time_s=0.0,
        x_world_m=x,
        y_world_m=y,
        z_world_m=0.125,
        yaw_rad=yaw,
        roll_rad=0.0,
        pitch_rad=0.0,
        vx_body_mps=0.0,
        vy_body_mps=0.0,
        wz_body_radps=0.0,
        tilt_rad=0.0,
        fallen=False,
        finite=finite,
    )


class ClassicalNavigatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ConstrainedGoToGoalConfig()

    def test_wrap_angle_crosses_pi(self) -> None:
        self.assertAlmostEqual(wrap_angle(math.radians(181.0)), math.radians(-179.0))

    def test_goal_error_uses_robot_yaw(self) -> None:
        distance, heading_error = goal_error(
            robot_state(yaw=math.pi / 2),
            GoalState(1.0, 0.0),
        )

        self.assertAlmostEqual(distance, 1.0)
        self.assertAlmostEqual(heading_error, -math.pi / 2)

    def test_mirrored_goals_produce_mirrored_yaw_commands(self) -> None:
        left = ConstrainedGoToGoalNavigator(self.config).compute_command(
            robot_state(), GoalState(1.0, 1.0), 0.1
        )
        right = ConstrainedGoToGoalNavigator(self.config).compute_command(
            robot_state(), GoalState(1.0, -1.0), 0.1
        )

        self.assertAlmostEqual(left.vx_mps, right.vx_mps)
        self.assertAlmostEqual(left.wz_radps, -right.wz_radps)
        self.assertGreater(left.wz_radps, 0.0)
        self.assertEqual(left.vy_mps, 0.0)

    def test_side_goal_keeps_small_forward_motion(self) -> None:
        command = ConstrainedGoToGoalNavigator(self.config).compute_command(
            robot_state(), GoalState(0.0, 1.5), 0.1
        )

        self.assertGreater(command.vx_mps, 0.0)
        self.assertLessEqual(command.vx_mps, self.config.min_turn_speed_mps)
        self.assertGreater(command.wz_radps, 0.0)

    def test_command_changes_are_rate_limited(self) -> None:
        command = ConstrainedGoToGoalNavigator(self.config).compute_command(
            robot_state(), GoalState(2.0, 2.0), 0.1
        )

        self.assertLessEqual(command.vx_mps, self.config.max_forward_accel_mps2 * 0.1)
        self.assertLessEqual(command.wz_radps, self.config.max_yaw_accel_radps2 * 0.1)

    def test_goal_inside_tolerance_commands_stop(self) -> None:
        command = ConstrainedGoToGoalNavigator(self.config).compute_command(
            robot_state(), GoalState(0.1, 0.0), 0.1
        )

        self.assertEqual(command.vx_mps, 0.0)
        self.assertEqual(command.vy_mps, 0.0)
        self.assertEqual(command.wz_radps, 0.0)

    def test_non_finite_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ConstrainedGoToGoalNavigator(self.config).compute_command(
                robot_state(finite=False), GoalState(1.0, 0.0), 0.1
            )


if __name__ == "__main__":
    unittest.main()
