from __future__ import annotations

import unittest

from evaluation.analyze_action_symmetry import (
    JOINT_PERM,
    JOINT_SIGN,
    OBS_PERM,
    OBS_SIGN,
    mirror_vector,
)


class ActionSymmetryTests(unittest.TestCase):
    def test_action_mirror_is_an_involution(self) -> None:
        values = [float(index + 1) for index in range(14)]

        mirrored = mirror_vector(values, JOINT_PERM, JOINT_SIGN)
        restored = mirror_vector(mirrored, JOINT_PERM, JOINT_SIGN)

        self.assertEqual(restored, values)

    def test_observation_mirror_is_an_involution(self) -> None:
        values = [float(index + 1) for index in range(61)]

        mirrored = mirror_vector(values, OBS_PERM, OBS_SIGN)
        restored = mirror_vector(mirrored, OBS_PERM, OBS_SIGN)

        self.assertEqual(restored, values)

    def test_twist_mirror_keeps_vx_and_negates_vy_wz(self) -> None:
        values = [0.0] * 61
        values[48:51] = [0.3, 0.2, 0.5]

        mirrored = mirror_vector(values, OBS_PERM, OBS_SIGN)

        self.assertEqual(mirrored[48:51], [0.3, -0.2, -0.5])


if __name__ == "__main__":
    unittest.main()
