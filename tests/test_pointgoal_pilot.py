from __future__ import annotations

import math
import unittest
from dataclasses import replace

from evaluation.pointgoal_pilot import (
    DEFAULT_GOAL_SET,
    aggregate_episodes,
    goal_from_initial_body_frame,
    load_goal_set,
    select_run_goals,
)
from evaluation.plot_pointgoal_pilot import world_to_initial_body
from navigation.types import RobotState


def robot_state(*, x: float, y: float, yaw: float) -> RobotState:
    return RobotState(
        time_s=1.0,
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
    )


def episode(
    goal_name: str,
    *,
    success: bool,
    mirror_group: str | None = None,
    mirror_side: str | None = None,
) -> dict[str, object]:
    return {
        "goal_name": goal_name,
        "mirror_group": mirror_group,
        "mirror_side": mirror_side,
        "success": success,
        "fall": False,
        "timeout": not success,
        "final_distance_m": 0.1 if success else 0.8,
        "path_efficiency": 0.8 if success else None,
        "progress_efficiency": 0.8 if success else 0.4,
        "duration_s": 8.0 if success else 20.0,
    }


class PointGoalProtocolTests(unittest.TestCase):
    def test_default_goal_set_is_a_frozen_25_episode_protocol(self) -> None:
        protocol = load_goal_set(DEFAULT_GOAL_SET)

        self.assertEqual(protocol.protocol_status, "frozen")
        self.assertEqual(protocol.episode_count, 25)
        self.assertEqual([goal.name for goal in protocol.goals], [
            "front",
            "front_left",
            "front_right",
            "left",
            "right",
        ])
        self.assertTrue(all(goal.episodes == 5 for goal in protocol.goals))

    def test_goal_pairs_are_geometric_mirrors(self) -> None:
        protocol = load_goal_set(DEFAULT_GOAL_SET)
        goals = {goal.name: goal for goal in protocol.goals}

        for left_name, right_name in (("front_left", "front_right"), ("left", "right")):
            left = goals[left_name]
            right = goals[right_name]
            self.assertAlmostEqual(left.x_initial_body_m, right.x_initial_body_m)
            self.assertAlmostEqual(left.y_initial_body_m, -right.y_initial_body_m)
            self.assertAlmostEqual(
                math.hypot(left.x_initial_body_m, left.y_initial_body_m),
                math.hypot(right.x_initial_body_m, right.y_initial_body_m),
            )

    def test_initial_body_goal_is_rotated_into_world_frame(self) -> None:
        goal = goal_from_initial_body_frame(
            robot_state(x=2.0, y=3.0, yaw=math.pi / 2),
            1.5,
            0.0,
        )

        self.assertAlmostEqual(goal.x_world_m, 2.0)
        self.assertAlmostEqual(goal.y_world_m, 4.5)

    def test_world_point_round_trips_to_initial_body_frame(self) -> None:
        goal = goal_from_initial_body_frame(
            robot_state(x=2.0, y=3.0, yaw=math.pi / 3),
            1.2,
            -0.7,
        )

        local = world_to_initial_body(
            goal.x_world_m,
            goal.y_world_m,
            2.0,
            3.0,
            math.pi / 3,
        )

        self.assertAlmostEqual(local[0], 1.2)
        self.assertAlmostEqual(local[1], -0.7)

    def test_smoke_selects_one_episode_from_frozen_protocol(self) -> None:
        protocol = load_goal_set(DEFAULT_GOAL_SET)

        selected = select_run_goals(protocol, smoke=True, smoke_goal="front_left")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "front_left")
        self.assertEqual(selected[0].episodes, 1)

    def test_formal_run_selects_all_goals_from_frozen_protocol(self) -> None:
        protocol = load_goal_set(DEFAULT_GOAL_SET)

        selected = select_run_goals(protocol, smoke=False, smoke_goal="front")

        self.assertEqual(selected, protocol.goals)
        self.assertEqual(sum(goal.episodes for goal in selected), 25)

    def test_formal_run_is_rejected_while_protocol_is_draft(self) -> None:
        protocol = replace(load_goal_set(DEFAULT_GOAL_SET), protocol_status="draft")

        with self.assertRaisesRegex(ValueError, "requires protocol_status='frozen'"):
            select_run_goals(protocol, smoke=False, smoke_goal="front")

    def test_aggregate_keeps_goal_and_mirror_results_separate(self) -> None:
        episodes = [
            episode(
                "front_left",
                success=True,
                mirror_group="front_diagonal",
                mirror_side="left",
            ),
            episode(
                "front_right",
                success=False,
                mirror_group="front_diagonal",
                mirror_side="right",
            ),
        ]

        aggregate = aggregate_episodes(episodes)

        self.assertEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(aggregate["goals"]["front_left"]["success_rate"], 1.0)
        self.assertEqual(aggregate["goals"]["front_right"]["success_rate"], 0.0)
        self.assertEqual(
            aggregate["mirror_groups"]["front_diagonal"][
                "left_minus_right_success_rate"
            ],
            1.0,
        )

    def test_single_side_smoke_marks_mirror_summary_partial(self) -> None:
        aggregate = aggregate_episodes([
            episode(
                "front_left",
                success=False,
                mirror_group="front_diagonal",
                mirror_side="left",
            )
        ])

        self.assertEqual(
            aggregate["mirror_groups"]["front_diagonal"],
            {"complete": False, "present_sides": ["left"]},
        )


if __name__ == "__main__":
    unittest.main()
