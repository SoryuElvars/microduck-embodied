from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from evaluation.classical_pointgoal_ood import (
    DEFAULT_CONFIG,
    OodCondition,
    load_ood_protocol,
    select_specs,
)


class ClassicalPointGoalOodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_ood_protocol(Path(DEFAULT_CONFIG))

    def test_protocol_has_three_one_factor_conditions(self) -> None:
        self.assertEqual(
            tuple(condition.factor for condition in self.protocol.conditions),
            ("foot_friction", "actuator_delay", "motor_strength_proxy"),
        )
        for condition in self.protocol.conditions:
            condition.validate()

    def test_quick_and_full_selections_are_disjoint(self) -> None:
        quick = select_specs(self.protocol, "quick")
        frozen = replace(self.protocol, protocol_status="frozen")
        full = select_specs(frozen, "full")

        self.assertEqual(len(quick), 10)
        self.assertEqual(len(full), 25)
        self.assertFalse(
            {episode.episode_id for episode in quick}
            & {episode.episode_id for episode in full}
        )

    def test_full_is_blocked_while_protocol_is_draft(self) -> None:
        draft = replace(self.protocol, protocol_status="draft")
        with self.assertRaisesRegex(ValueError, "protocol_status='frozen'"):
            select_specs(draft, "full")

    def test_condition_rejects_multiple_active_factors(self) -> None:
        condition = OodCondition(
            condition_id="invalid",
            factor="foot_friction",
            foot_friction=0.3,
            actuator_delay_ms=40,
        )

        with self.assertRaisesRegex(ValueError, "must change only"):
            condition.validate()


if __name__ == "__main__":
    unittest.main()
