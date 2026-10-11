"""FIFO transport delay between a Navigator and the frozen locomotion policy.

This deliberately operates above :class:`MujocoBackend`: it delays a velocity
command after the Navigator produced it, rather than changing the locomotion
actor's joint-action delay.  The two perturbations must not be conflated.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
import json

from navigation.types import VelocityCommand


@dataclass(frozen=True)
class CommandPacket:
    sequence: int
    generated_at_s: float
    command: VelocityCommand
    synthetic_initial: bool = False


@dataclass(frozen=True)
class CommandDelivery:
    configured_delay_ms: int
    configured_delay_steps: int
    generated_sequence: int
    generated_at_s: float
    applied_at_s: float
    observed_delay_s: float
    generated_command: VelocityCommand
    applied_command: VelocityCommand
    synthetic_initial: bool

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["generated_command"] = asdict(self.generated_command)
        record["applied_command"] = asdict(self.applied_command)
        return record


class DelayedVelocityCommandBackend:
    """Expose a MujocoBackend-compatible interface with a fixed FIFO delay."""

    def __init__(self, backend: Any, *, delay_ms: int) -> None:
        if delay_ms < 0:
            raise ValueError("delay_ms must be >= 0")
        self._backend = backend
        self.delay_ms = delay_ms
        self._queue: deque[CommandPacket] | None = None
        self._next_sequence = 0
        self.deliveries: list[CommandDelivery] = []

    @property
    def control_dt_s(self) -> float:
        return float(self._backend.control_dt_s)

    @property
    def delay_steps(self) -> int:
        steps = self.delay_ms / (1000.0 * self.control_dt_s)
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError(
                "high-level command delay must be an integer multiple of the "
                f"backend control period ({1000.0 * self.control_dt_s:g} ms)"
            )
        return round(steps)

    def validate(self) -> None:
        self._backend.validate()
        # MujocoBackend exposes its control period only after reset.  MicroDuck's
        # frozen command contract is 20 ms, so catch malformed static configs
        # here and verify the actual runtime period again in ``reset``.
        if self.delay_ms % 20:
            raise ValueError(
                "high-level command delay must be an integer multiple of the "
                "frozen 20 ms MicroDuck control period"
            )

    def reset(self, episode_seed: int, initial_state_mode: str) -> Any:
        state = self._backend.reset(episode_seed, initial_state_mode)
        delay_steps = self.delay_steps
        zero = VelocityCommand(0.0, 0.0, 0.0)
        self._queue = deque(
            CommandPacket(
                sequence=-delay_steps + index,
                generated_at_s=(-delay_steps + index) * self.control_dt_s,
                command=zero,
                synthetic_initial=True,
            )
            for index in range(delay_steps)
        )
        self._next_sequence = 0
        self.deliveries = []
        return state

    def observe(self) -> Any:
        return self._backend.observe()

    def step(self, command: VelocityCommand) -> Any:
        if self._queue is None:
            raise RuntimeError("backend must be reset before stepping")
        applied_at_s = float(self.observe().time_s)
        generated = CommandPacket(
            sequence=self._next_sequence,
            generated_at_s=applied_at_s,
            command=command,
        )
        self._next_sequence += 1
        if self.delay_steps:
            self._queue.append(generated)
            applied = self._queue.popleft()
        else:
            applied = generated
        result = self._backend.step(applied.command)
        self.deliveries.append(
            CommandDelivery(
                configured_delay_ms=self.delay_ms,
                configured_delay_steps=self.delay_steps,
                generated_sequence=applied.sequence,
                generated_at_s=applied.generated_at_s,
                applied_at_s=applied_at_s,
                observed_delay_s=applied_at_s - applied.generated_at_s,
                # This is the command carried by the packet whose generation
                # sequence/time is recorded above, not the newer packet that
                # has just entered the FIFO.
                generated_command=applied.command,
                applied_command=applied.command,
                synthetic_initial=applied.synthetic_initial,
            )
        )
        return result

    def open_viewer(self) -> Any:
        return self._backend.open_viewer()

    def readback(self) -> dict[str, Any]:
        actual = [item for item in self.deliveries if not item.synthetic_initial]
        observed_steps = sorted(
            {
                round(item.observed_delay_s / self.control_dt_s)
                for item in actual
            }
        )
        return {
            "configured_delay_ms": self.delay_ms,
            "configured_delay_steps": self.delay_steps,
            "control_dt_s": self.control_dt_s,
            "delivery_count": len(self.deliveries),
            "non_synthetic_delivery_count": len(actual),
            "observed_delay_steps": observed_steps,
            "observed_delay_ms": [
                1000.0 * steps * self.control_dt_s for steps in observed_steps
            ],
            "readback_passed": bool(actual) and observed_steps == [self.delay_steps],
        }

    def write_delivery_trace(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "readback": self.readback(),
            "deliveries": [item.as_record() for item in self.deliveries],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
