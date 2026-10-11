from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from navigation.command_delay import DelayedVelocityCommandBackend
from navigation.types import VelocityCommand


@dataclass
class _FakeBackend:
    control_dt_s: float = 0.02

    def __post_init__(self) -> None:
        self.time_s = 0.0
        self.applied: list[VelocityCommand] = []

    def validate(self) -> None:
        pass

    def reset(self, episode_seed: int, initial_state_mode: str) -> SimpleNamespace:
        self.time_s = 0.0
        self.applied = []
        return self.observe()

    def observe(self) -> SimpleNamespace:
        return SimpleNamespace(time_s=self.time_s)

    def step(self, command: VelocityCommand) -> SimpleNamespace:
        self.applied.append(command)
        self.time_s += self.control_dt_s
        return SimpleNamespace()


class CommandDelayTests(unittest.TestCase):
    def test_two_control_steps_are_delayed_by_40ms(self) -> None:
        inner = _FakeBackend()
        backend = DelayedVelocityCommandBackend(inner, delay_ms=40)
        backend.reset(0, "zero")
        first = VelocityCommand(0.1, 0.0, 0.2)
        second = VelocityCommand(0.2, 0.0, 0.3)

        backend.step(first)
        backend.step(second)
        backend.step(second)

        self.assertEqual(inner.applied[:2], [VelocityCommand(0.0, 0.0, 0.0)] * 2)
        self.assertEqual(inner.applied[2], first)
        self.assertEqual(backend.readback()["observed_delay_ms"], [40.0])
        self.assertTrue(backend.readback()["readback_passed"])

    def test_zero_delay_applies_the_new_command_immediately(self) -> None:
        inner = _FakeBackend()
        backend = DelayedVelocityCommandBackend(inner, delay_ms=0)
        backend.reset(0, "zero")
        command = VelocityCommand(0.1, 0.0, -0.2)

        backend.step(command)

        self.assertEqual(inner.applied, [command])
        self.assertEqual(backend.readback()["observed_delay_ms"], [0.0])

    def test_delay_must_align_to_control_period(self) -> None:
        backend = DelayedVelocityCommandBackend(_FakeBackend(), delay_ms=10)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            backend.validate()


if __name__ == "__main__":
    unittest.main()
