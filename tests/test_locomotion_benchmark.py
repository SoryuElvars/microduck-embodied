from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.locomotion_benchmark import (
    DEFAULT_COMMAND_SET,
    PROJECT_ROOT,
    load_command_set,
    path_for_record,
    quaternion_to_euler,
    sample_initial_state,
    wrapped_angle_delta,
)


class CommandSetTests(unittest.TestCase):
    def test_policy_path_can_be_recorded_from_official_or_project_root(self) -> None:
        official_root = Path("/workspace/microduck_rl")
        project_root = Path("/workspace/microduck-embodied")

        self.assertEqual(
            path_for_record(
                official_root / "logs" / "velocity" / "policy.onnx",
                official_root,
                project_root,
            ),
            "logs/velocity/policy.onnx",
        )
        self.assertEqual(
            path_for_record(
                project_root / "artifacts" / "week03" / "policy.onnx",
                official_root,
                project_root,
            ),
            "artifacts/week03/policy.onnx",
        )

    def test_straight_only_contract(self) -> None:
        commands = load_command_set(DEFAULT_COMMAND_SET)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].name, "straight_030")
        self.assertEqual((commands[0].vx, commands[0].vy, commands[0].wz), (0.3, 0.0, 0.0))
        self.assertEqual(commands[0].warmup_s, 1.0)
        self.assertEqual(commands[0].duration_s, 10.0)
        self.assertEqual(commands[0].episodes, 1)
        self.assertEqual(commands[0].initial_state_mode, "fixed")

    def test_duplicate_command_names_are_rejected(self) -> None:
        document = {
            "schema_version": 1,
            "defaults": {
                "warmup_s": 1.0,
                "duration_s": 10.0,
                "episodes": 1,
                "initial_state_mode": "fixed",
            },
            "commands": [
                {"name": "duplicate", "vx": 0.1, "vy": 0.0, "wz": 0.0},
                {"name": "duplicate", "vx": 0.2, "vy": 0.0, "wz": 0.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unique"):
                load_command_set(path)

    def test_command_response_matrix_has_eight_paired_cases(self) -> None:
        path = PROJECT_ROOT / "evaluation" / "command_sets" / "command_response_20.json"
        commands = load_command_set(path)

        self.assertEqual(len(commands), 8)
        self.assertTrue(all(command.episodes == 20 for command in commands))
        self.assertTrue(
            all(command.initial_state_mode == "official_reset" for command in commands)
        )
        self.assertEqual(
            [(command.vx, command.vy, command.wz) for command in commands],
            [
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.3, 0.0, 0.0),
                (-0.3, 0.0, 0.0),
                (0.0, 0.2, 0.0),
                (0.0, -0.2, 0.0),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, -0.5),
            ],
        )

    def test_checkpoint_probe_has_three_paired_five_episode_cases(self) -> None:
        path = (
            PROJECT_ROOT
            / "evaluation"
            / "command_sets"
            / "checkpoint_probe_5.json"
        )
        commands = load_command_set(path)

        self.assertEqual(len(commands), 3)
        self.assertTrue(all(command.episodes == 5 for command in commands))
        self.assertTrue(
            all(command.initial_state_mode == "official_reset" for command in commands)
        )
        self.assertEqual(
            [(command.vx, command.vy, command.wz) for command in commands],
            [
                (0.3, 0.0, 0.0),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, -0.5),
            ],
        )

    def test_angular_std025_gate_has_three_paired_five_episode_cases(self) -> None:
        path = (
            PROJECT_ROOT
            / "evaluation"
            / "command_sets"
            / "angular_std025_gate_5.json"
        )
        commands = load_command_set(path)

        self.assertEqual(len(commands), 3)
        self.assertTrue(all(command.episodes == 5 for command in commands))
        self.assertTrue(all(command.lead_in_s == 0.0 for command in commands))
        self.assertTrue(
            all(command.initial_state_mode == "official_reset" for command in commands)
        )
        self.assertEqual(
            [(command.vx, command.vy, command.wz) for command in commands],
            [
                (0.3, 0.0, 0.0),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, -0.5),
            ],
        )

    def test_pointgoal_gate_establishes_gait_before_moving_turns(self) -> None:
        path = (
            PROJECT_ROOT
            / "evaluation"
            / "command_sets"
            / "pointgoal_moving_turn_5.json"
        )
        commands = load_command_set(path)

        self.assertEqual(len(commands), 5)
        self.assertTrue(all(command.episodes == 5 for command in commands))
        self.assertTrue(all(command.lead_in_s == 2.0 for command in commands))
        self.assertTrue(
            all(
                (command.lead_in_vx, command.lead_in_vy, command.lead_in_wz)
                == (0.25, 0.0, 0.0)
                for command in commands
            )
        )
        self.assertEqual(
            [(command.vx, command.vy, command.wz) for command in commands],
            [
                (0.25, 0.0, 0.0),
                (0.25, 0.0, 0.25),
                (0.25, 0.0, -0.25),
                (0.2, 0.0, 0.5),
                (0.2, 0.0, -0.5),
            ],
        )

    def test_checkpoint_response_has_eight_paired_five_episode_cases(self) -> None:
        path = (
            PROJECT_ROOT
            / "evaluation"
            / "command_sets"
            / "checkpoint_response_5.json"
        )
        commands = load_command_set(path)

        self.assertEqual(len(commands), 8)
        self.assertTrue(all(command.episodes == 5 for command in commands))
        self.assertTrue(
            all(command.initial_state_mode == "official_reset" for command in commands)
        )
        self.assertEqual(
            [(command.vx, command.vy, command.wz) for command in commands],
            [
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 0.0),
                (0.3, 0.0, 0.0),
                (-0.3, 0.0, 0.0),
                (0.0, 0.2, 0.0),
                (0.0, -0.2, 0.0),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, -0.5),
            ],
        )

    def test_identity_quaternion_has_zero_euler_angles(self) -> None:
        roll, pitch, yaw = quaternion_to_euler((1.0, 0.0, 0.0, 0.0))

        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_wrapped_angle_delta_crosses_pi_continuously(self) -> None:
        delta = wrapped_angle_delta(math.radians(-179.0), math.radians(179.0))

        self.assertAlmostEqual(math.degrees(delta), 2.0)

    def test_official_reset_sampling_is_seeded_and_in_range(self) -> None:
        first = sample_initial_state("official_reset", 42)
        repeated = sample_initial_state("official_reset", 42)

        self.assertEqual(first, repeated)
        self.assertGreaterEqual(first["x"], -0.5)
        self.assertLessEqual(first["x"], 0.5)
        self.assertGreaterEqual(first["y"], -0.5)
        self.assertLessEqual(first["y"], 0.5)
        self.assertGreaterEqual(first["z"], 0.12)
        self.assertLessEqual(first["z"], 0.13)
        self.assertGreaterEqual(first["yaw"], -math.pi)
        self.assertLessEqual(first["yaw"], math.pi)


if __name__ == "__main__":
    unittest.main()
