import unittest

from evaluation.pointgoal_stable_arrival_rescore import rescore_rows


def _rows(*, steps=120, vx=0.05, wz=0.02, phase="navigate"):
    return [
        {
            "phase": phase,
            "time_s": str((index + 1) * 0.02),
            "distance_to_goal_m": "0.1",
            "actual_vx": str(vx),
            "actual_vy": "0.0",
            "actual_wz": str(wz),
        }
        for index in range(steps)
    ]


class StableArrivalRescoreTests(unittest.TestCase):
    def _score(self, rows, k=5):
        return rescore_rows(
            rows,
            k=k,
            radius_m=0.2,
            hold_s=0.5,
            max_planar_speed_mps=0.1,
            max_abs_yaw_rate_radps=0.2,
        )

    def test_stationary_in_circle_succeeds_after_window_and_hold(self):
        result = self._score(_rows())
        self.assertTrue(result["observed_stable_arrival"])
        self.assertEqual(result["first_hit"]["phase"], "navigate")
        self.assertEqual(result["window_steps"], 5)

    def test_fast_motion_and_persistent_rotation_do_not_succeed(self):
        self.assertFalse(self._score(_rows(vx=0.15))["observed_stable_arrival"])
        self.assertFalse(self._score(_rows(wz=0.25))["observed_stable_arrival"])

    def test_post_arrival_hit_is_identified_separately(self):
        rows = _rows(steps=40, vx=0.15)
        rows += _rows(steps=80, phase="post_arrival")
        result = self._score(rows)
        self.assertTrue(result["observed_stable_arrival"])
        self.assertEqual(result["first_hit"]["phase"], "post_arrival")


if __name__ == "__main__":
    unittest.main()
