from __future__ import annotations

import math
import unittest

from evaluation.plot_checkpoint_gate import (
    checkpoint_from_policy_path,
    command_key,
    describe,
)


class CheckpointGatePlotTests(unittest.TestCase):
    def test_checkpoint_is_inferred_from_policy_stem(self) -> None:
        self.assertEqual(
            checkpoint_from_policy_path("artifacts/models/model_1999.onnx"),
            1999,
        )

    def test_checkpoint_rejects_an_unversioned_policy_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot infer checkpoint"):
            checkpoint_from_policy_path("artifacts/models/policy.onnx")

    def test_gate_commands_are_classified_by_values(self) -> None:
        self.assertEqual(command_key({"vx": 0.3, "vy": 0.0, "wz": 0.0}), "forward")
        self.assertEqual(command_key({"vx": 0.0, "vy": 0.0, "wz": 0.5}), "yaw_pos")
        self.assertEqual(command_key({"vx": 0.0, "vy": 0.0, "wz": -0.5}), "yaw_neg")

    def test_describe_uses_population_standard_deviation(self) -> None:
        result = describe([1.0, 2.0, 3.0])

        self.assertEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], math.sqrt(2.0 / 3.0))
        self.assertEqual(result["values"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
