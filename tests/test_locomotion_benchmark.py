from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.locomotion_benchmark import DEFAULT_COMMAND_SET, load_command_set


class CommandSetTests(unittest.TestCase):
    def test_straight_only_contract(self) -> None:
        commands = load_command_set(DEFAULT_COMMAND_SET)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].name, "straight_030")
        self.assertEqual((commands[0].vx, commands[0].vy, commands[0].wz), (0.3, 0.0, 0.0))
        self.assertEqual(commands[0].warmup_s, 1.0)
        self.assertEqual(commands[0].duration_s, 10.0)
        self.assertEqual(commands[0].episodes, 1)

    def test_duplicate_command_names_are_rejected(self) -> None:
        document = {
            "schema_version": 1,
            "defaults": {"warmup_s": 1.0, "duration_s": 10.0, "episodes": 1},
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


if __name__ == "__main__":
    unittest.main()
