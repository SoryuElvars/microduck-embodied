from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

from navigation.observation_perturbation import (
    ObservationPerturbation,
    ObservationPerturbedNavigator,
)
from navigation.types import GoalState, RobotState, VelocityCommand


@dataclass
class _RecordingNavigator:
    calls: list[tuple[RobotState, GoalState]]

    def reset(self) -> None:
        pass

    def compute_command(self, robot: RobotState, goal: GoalState, dt_s: float) -> VelocityCommand:
        self.calls.append((robot, goal))
        return VelocityCommand(0.0, 0.0, 0.0)


def _robot(*, x: float = 1.0, y: float = 2.0, yaw: float = 0.0, time_s: float = 0.0) -> RobotState:
    return RobotState(time_s, x, y, 0.2, yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


class ObservationPerturbationTests(unittest.TestCase):
    def test_yaw_bias_changes_only_navigator_yaw(self) -> None:
        recorder = _RecordingNavigator([])
        navigator = ObservationPerturbedNavigator(
            recorder,
            ObservationPerturbation("yaw", "robot_yaw_bias", yaw_bias_rad=math.pi / 2),
            episode_seed=42,
        )
        navigator.reset()
        original = _robot(yaw=0.2)
        navigator.compute_command(original, GoalState(3.0, 4.0), 0.2)

        observed, goal = recorder.calls[0]
        self.assertAlmostEqual(observed.yaw_rad, 0.2 + math.pi / 2)
        self.assertEqual(observed.x_world_m, original.x_world_m)
        self.assertEqual(goal, GoalState(3.0, 4.0))

    def test_robot_and_goal_position_biases_are_separate(self) -> None:
        recorder = _RecordingNavigator([])
        robot_bias = ObservationPerturbedNavigator(
            recorder,
            ObservationPerturbation(
                "robot", "robot_position_bias", robot_x_bias_m=0.1, robot_y_bias_m=-0.2
            ),
            episode_seed=42,
        )
        robot_bias.reset()
        robot_bias.compute_command(_robot(), GoalState(3.0, 4.0), 0.2)
        self.assertEqual(recorder.calls[0][0].x_world_m, 1.1)
        self.assertEqual(recorder.calls[0][1], GoalState(3.0, 4.0))

        recorder.calls.clear()
        goal_bias = ObservationPerturbedNavigator(
            recorder,
            ObservationPerturbation(
                "goal", "goal_position_bias", goal_x_bias_m=0.1, goal_y_bias_m=-0.2
            ),
            episode_seed=42,
        )
        goal_bias.reset()
        goal_bias.compute_command(_robot(), GoalState(3.0, 4.0), 0.2)
        self.assertEqual(recorder.calls[0][0], _robot())
        self.assertEqual(recorder.calls[0][1], GoalState(3.1, 3.8))

    def test_dropout_holds_the_last_robot_state(self) -> None:
        recorder = _RecordingNavigator([])
        navigator = ObservationPerturbedNavigator(
            recorder,
            ObservationPerturbation(
                "drop", "robot_state_dropout", dropout_probability=0.999, dropout_seed=1
            ),
            episode_seed=42,
        )
        first = _robot(x=1.0, time_s=0.0)
        second = _robot(x=2.0, time_s=0.2)
        navigator.reset()
        navigator.compute_command(first, GoalState(3.0, 4.0), 0.2)
        navigator.compute_command(second, GoalState(3.0, 4.0), 0.2)
        self.assertEqual(recorder.calls[1][0], first)
        self.assertEqual(navigator.readback()["dropout_count"], 1)


if __name__ == "__main__":
    unittest.main()
