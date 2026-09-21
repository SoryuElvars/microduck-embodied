from __future__ import annotations

import math
import unittest
from pathlib import Path

from evaluation.pointgoal_protocol import (
    SMOKE_DIRECTIONS_RAD,
    generate_week04_classical_v1,
    load_pointgoal_protocol,
)
from navigation.classical_navigator import wrap_angle


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "pointgoal_protocols"
    / "week04_classical_v1.json"
)


class PointGoalProtocolTests(unittest.TestCase):
    def test_manifest_and_controller_protocol_are_frozen(self) -> None:
        protocol = load_pointgoal_protocol(PROTOCOL_PATH)

        self.assertEqual(protocol.protocol_id, "week04_classical_v1")
        self.assertEqual(protocol.protocol_status, "frozen")
        self.assertEqual(protocol.episode_manifest_status, "frozen")
        self.assertEqual(len(protocol.episodes), 200)
        self.assertEqual(len(protocol.quick_episode_ids), 10)

    def test_generation_is_exactly_reproducible(self) -> None:
        expected = generate_week04_classical_v1()
        protocol = load_pointgoal_protocol(PROTOCOL_PATH)

        self.assertEqual(
            protocol.episode_manifest_sha256,
            expected["episode_manifest_sha256"],
        )
        self.assertEqual(
            [episode.episode_id for episode in protocol.episodes],
            [episode["episode_id"] for episode in expected["episodes"]],
        )

    def test_episode_specs_have_unique_ids_and_reset_seeds(self) -> None:
        protocol = load_pointgoal_protocol(PROTOCOL_PATH)

        self.assertEqual(
            len({episode.episode_id for episode in protocol.episodes}), 200
        )
        self.assertEqual(
            len({episode.simulation_reset_seed for episode in protocol.episodes}),
            200,
        )
        self.assertTrue(
            all(episode.post_arrival_observation_time == 2.0 for episode in protocol.episodes)
        )

    def test_smoke_scenarios_cover_four_directions_from_frozen_manifest(self) -> None:
        protocol = load_pointgoal_protocol(PROTOCOL_PATH)

        for name, target in SMOKE_DIRECTIONS_RAD.items():
            episode = protocol.episode_map[protocol.smoke_episode_ids[name]]
            relative_heading = wrap_angle(
                math.atan2(
                    episode.goal_y - episode.start_y,
                    episode.goal_x - episode.start_x,
                )
                - episode.start_yaw
            )
            self.assertLess(abs(wrap_angle(relative_heading - target)), math.radians(2.0))

    def test_controllers_share_speed_yaw_and_goal_limits(self) -> None:
        protocol = load_pointgoal_protocol(PROTOCOL_PATH)

        self.assertEqual(
            protocol.naive_p.max_forward_speed_mps,
            protocol.constrained.max_forward_speed_mps,
        )
        self.assertEqual(
            protocol.naive_p.max_yaw_rate_radps,
            protocol.constrained.max_yaw_rate_radps,
        )
        self.assertEqual(
            protocol.naive_p.goal_tolerance_m,
            protocol.constrained.goal_tolerance_m,
        )
        self.assertEqual(protocol.naive_p.goal_tolerance_m, 0.15)
        self.assertTrue(
            all(
                protocol.naive_p.goal_tolerance_m < episode.success_radius
                for episode in protocol.episodes
            )
        )


if __name__ == "__main__":
    unittest.main()
