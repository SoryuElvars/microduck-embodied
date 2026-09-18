import unittest

from evaluation.plot_moving_turn_checkpoint_screen import (
    checkpoint_from_policy_path,
    command_key,
)


class MovingTurnCheckpointScreenTests(unittest.TestCase):
    def test_checkpoint_is_inferred_from_policy_name(self) -> None:
        self.assertEqual(checkpoint_from_policy_path("models/model_1750.onnx"), 1750)

    def test_commands_are_classified_by_values(self) -> None:
        commands = {
            "straight": {"vx": 0.25, "vy": 0.0, "wz": 0.0},
            "gentle_left": {"vx": 0.25, "vy": 0.0, "wz": 0.25},
            "gentle_right": {"vx": 0.25, "vy": 0.0, "wz": -0.25},
            "turn_left": {"vx": 0.2, "vy": 0.0, "wz": 0.5},
            "turn_right": {"vx": 0.2, "vy": 0.0, "wz": -0.5},
        }

        for expected, command in commands.items():
            self.assertEqual(command_key(command), expected)

    def test_unknown_command_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected moving-turn command"):
            command_key({"vx": 0.0, "vy": 0.0, "wz": 0.5})


if __name__ == "__main__":
    unittest.main()
