from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.rl_pointgoal_high_delay_ood import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    load_protocol,
    select_specs,
)


class HighLevelDelayOodProtocolTests(unittest.TestCase):
    def test_frozen_protocol_has_the_declared_delay_sweep(self) -> None:
        protocol = load_protocol(Path(DEFAULT_CONFIG))
        self.assertEqual(protocol.protocol_status, "frozen")
        self.assertEqual([item.command_delay_ms for item in protocol.conditions], [0, 20, 40])
        self.assertEqual(len(select_specs(protocol, "quick")), 10)
        self.assertEqual(len(select_specs(protocol, "full")), 200)

    def test_frozen_stress_protocol_is_a_separate_60_80ms_tier(self) -> None:
        protocol = load_protocol(
            PROJECT_ROOT
            / "evaluation"
            / "pointgoal_protocols"
            / "week05_rl_high_delay_ood_v2_stress.json"
        )
        self.assertEqual(protocol.protocol_status, "frozen")
        self.assertEqual([item.command_delay_ms for item in protocol.conditions], [60, 80])
        self.assertEqual(len(select_specs(protocol, "quick")), 10)

    def test_frozen_continuation_tier_is_100_120ms(self) -> None:
        protocol = load_protocol(
            PROJECT_ROOT
            / "evaluation"
            / "pointgoal_protocols"
            / "week05_rl_high_delay_ood_v3_stress_100_120.json"
        )
        self.assertEqual([item.command_delay_ms for item in protocol.conditions], [100, 120])

    def test_frozen_continuation_tier_is_140_160ms(self) -> None:
        protocol = load_protocol(
            PROJECT_ROOT
            / "evaluation"
            / "pointgoal_protocols"
            / "week05_rl_high_delay_ood_v4_stress_140_160.json"
        )
        self.assertEqual([item.command_delay_ms for item in protocol.conditions], [140, 160])

    def test_frozen_decision_delay_uses_whole_200ms_periods(self) -> None:
        protocol = load_protocol(
            PROJECT_ROOT
            / "evaluation"
            / "pointgoal_protocols"
            / "week05_rl_decision_delay_ood_v1.json"
        )
        self.assertEqual(protocol.delay_quantum_ms, 200)
        self.assertEqual([item.command_delay_ms for item in protocol.conditions], [0, 200, 400])

    def test_frozen_decision_delay_continuation_is_600_800ms(self) -> None:
        protocol = load_protocol(
            PROJECT_ROOT
            / "evaluation"
            / "pointgoal_protocols"
            / "week05_rl_decision_delay_ood_v2_600_800.json"
        )
        self.assertEqual(protocol.delay_quantum_ms, 200)
        self.assertEqual([item.command_delay_ms for item in protocol.conditions], [600, 800])


if __name__ == "__main__":
    unittest.main()
