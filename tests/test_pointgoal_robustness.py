from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.pointgoal_pilot import load_goal_set
from evaluation.pointgoal_robustness import (
    DEFAULT_CONFIG,
    RobustnessCondition,
    evaluate_condition_gate,
    load_robustness_protocol,
    protocol_for_mode,
    select_conditions,
)


class PointGoalRobustnessTests(unittest.TestCase):
    def test_default_protocol_has_frozen_quick_and_full_plans(self) -> None:
        protocol = load_robustness_protocol(DEFAULT_CONFIG)
        goals = load_goal_set(protocol.goal_set_path)

        self.assertEqual(protocol.protocol_status, "frozen")
        self.assertEqual(len(protocol.quick_condition_ids), 5)
        self.assertEqual(len(protocol.full_condition_ids), 14)
        self.assertEqual(
            len(protocol.quick_condition_ids)
            * len(goals.goals)
            * protocol.quick_episodes_per_goal,
            25,
        )
        self.assertEqual(
            len(protocol.full_condition_ids)
            * len(goals.goals)
            * protocol.full_episodes_per_goal,
            350,
        )

    def test_every_non_nominal_condition_changes_one_factor(self) -> None:
        protocol = load_robustness_protocol(DEFAULT_CONFIG)

        self.assertEqual(protocol.conditions[0].factor, "nominal")
        self.assertEqual(len(protocol.conditions), 14)
        for condition in protocol.conditions:
            condition.validate()

    def test_quick_subset_preserves_frozen_order(self) -> None:
        protocol = load_robustness_protocol(DEFAULT_CONFIG)

        selected = select_conditions(
            protocol,
            "quick",
            ["backlash_2deg", "nominal"],
        )

        self.assertEqual(
            [condition.condition_id for condition in selected],
            ["nominal", "backlash_2deg"],
        )

    def test_quick_rejects_a_full_only_condition(self) -> None:
        protocol = load_robustness_protocol(DEFAULT_CONFIG)

        with self.assertRaisesRegex(ValueError, "outside the quick plan"):
            select_conditions(protocol, "quick", ["delay_20ms"])

    def test_runtime_perturbation_resolves_official_backlash_scene(self) -> None:
        condition = RobustnessCondition(
            condition_id="backlash",
            factor="backlash",
            scene_xml_path="src/robot/scene_backlash.xml",
        )

        perturbation = condition.to_runtime_perturbation(Path("/official"))

        self.assertEqual(
            perturbation.scene_xml_path,
            Path("/official/src/robot/scene_backlash.xml"),
        )

    def test_mode_protocol_keeps_goals_and_pairs_episode_counts(self) -> None:
        protocol = load_robustness_protocol(DEFAULT_CONFIG)
        goals = load_goal_set(protocol.goal_set_path)

        quick_goals = protocol_for_mode(goals, 1)
        full_goals = protocol_for_mode(goals, 5)

        self.assertTrue(all(goal.episodes == 1 for goal in quick_goals.goals))
        self.assertTrue(all(goal.episodes == 5 for goal in full_goals.goals))
        self.assertEqual(goals.base_seed, quick_goals.base_seed)

    def test_full_gate_reports_each_criterion(self) -> None:
        aggregate = {
            "success_rate": 0.8,
            "fall_rate": 0.0,
            "timeout_rate": 0.2,
            "invalid_state_rate": 0.0,
            "goals": {
                "front": {"success_rate": 1.0},
                "left": {"success_rate": 0.6},
            },
            "mirror_groups": {
                "side": {
                    "complete": True,
                    "left_minus_right_success_rate": -0.4,
                }
            },
        }
        gate = {
            "minimum_overall_success_rate": 0.8,
            "minimum_per_goal_success_rate": 0.6,
            "maximum_fall_rate": 0.0,
            "maximum_timeout_rate": 0.2,
            "maximum_invalid_state_rate": 0.0,
            "maximum_absolute_mirror_success_rate_gap": 0.4,
        }

        result = evaluate_condition_gate(aggregate, gate)

        self.assertTrue(result["passed"])
        self.assertEqual(set(result["criteria"]), set(gate))


if __name__ == "__main__":
    unittest.main()
