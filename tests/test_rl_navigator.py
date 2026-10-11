import math
import unittest

from navigation.rl_navigator import (
    RLNavigator,
    RLNavigatorConfig,
    pointgoal_observation,
)
from navigation.types import GoalState, RobotState


def _robot(**overrides) -> RobotState:
    values = {
        "time_s": 0.0,
        "x_world_m": 1.0,
        "y_world_m": 2.0,
        "z_world_m": 0.12,
        "yaw_rad": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "vx_body_mps": 0.1,
        "vy_body_mps": -0.02,
        "wz_body_radps": 0.3,
        "tilt_rad": 0.0,
        "fallen": False,
    }
    values.update(overrides)
    return RobotState(**values)


class _RecordingPolicy:
    def __init__(self, action) -> None:
        self.action = action
        self.observations = []

    def __call__(self, observation):
        self.observations.append(tuple(observation))
        return self.action


class RLNavigatorTests(unittest.TestCase):
    def test_observation_rotates_goal_into_robot_body_frame(self):
        observation = pointgoal_observation(
            _robot(yaw_rad=math.pi / 2),
            GoalState(x_world_m=2.0, y_world_m=2.0),
        )

        expected = [0.0, -1.0, 0.1, -0.02, 0.3, 0.0, 0.0]
        for actual, wanted in zip(observation, expected, strict=True):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(len(observation), 7)

    def test_normalized_action_is_bounded_and_rate_limited(self):
        navigator = RLNavigator(_RecordingPolicy([4.0, -3.0]), RLNavigatorConfig())

        first = navigator.compute_command(
            _robot(), GoalState(2.0, 2.0), dt_s=0.1
        )
        second = navigator.compute_command(
            _robot(), GoalState(2.0, 2.0), dt_s=1.0
        )

        self.assertAlmostEqual(first.vx_mps, 0.05)
        self.assertAlmostEqual(first.vy_mps, 0.0)
        self.assertAlmostEqual(first.wz_radps, -0.1)
        self.assertAlmostEqual(second.vx_mps, 0.25)
        self.assertAlmostEqual(second.wz_radps, -0.5)

    def test_previous_applied_command_is_part_of_next_observation(self):
        policy = _RecordingPolicy([1.0, 1.0])
        navigator = RLNavigator(policy, RLNavigatorConfig())

        navigator.compute_command(_robot(), GoalState(2.0, 2.0), dt_s=0.1)
        navigator.compute_command(_robot(), GoalState(2.0, 2.0), dt_s=0.1)

        self.assertEqual(policy.observations[0][5:], (0.0, 0.0))
        for actual, wanted in zip(
            policy.observations[1][5:], (0.05, 0.1), strict=True
        ):
            self.assertAlmostEqual(actual, wanted)

    def test_reset_clears_command_memory(self):
        policy = _RecordingPolicy([1.0, 1.0])
        navigator = RLNavigator(policy, RLNavigatorConfig())
        navigator.compute_command(_robot(), GoalState(2.0, 2.0), dt_s=0.1)

        navigator.reset()
        navigator.compute_command(_robot(), GoalState(2.0, 2.0), dt_s=0.1)

        self.assertEqual(policy.observations[-1][5:], (0.0, 0.0))

    def test_near_goal_does_not_inject_a_classical_stop_rule(self):
        navigator = RLNavigator(_RecordingPolicy([1.0, 0.0]), RLNavigatorConfig())

        command = navigator.compute_command(
            _robot(), GoalState(1.01, 2.0), dt_s=1.0
        )

        self.assertAlmostEqual(command.vx_mps, 0.25)

    def test_invalid_policy_output_is_rejected(self):
        navigator = RLNavigator(_RecordingPolicy([0.0]), RLNavigatorConfig())

        with self.assertRaisesRegex(ValueError, "must return 2 actions"):
            navigator.compute_command(_robot(), GoalState(2.0, 2.0), dt_s=0.1)


if __name__ == "__main__":
    unittest.main()
