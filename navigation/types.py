"""Backend-independent state and command contracts for PointGoal navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RobotState:
    """Robot state exposed by a backend to a navigator.

    World-frame position and yaw are allowed in the simulator-backed pilot, but
    their source stays behind the backend boundary.  A real deployment can
    populate the same contract from motion capture, vision, or odometry.
    """

    time_s: float
    x_world_m: float
    y_world_m: float
    z_world_m: float
    yaw_rad: float
    roll_rad: float
    pitch_rad: float
    vx_body_mps: float
    vy_body_mps: float
    wz_body_radps: float
    tilt_rad: float
    fallen: bool
    finite: bool = True


@dataclass(frozen=True)
class GoalState:
    """A positional goal in the world frame; final orientation is not required."""

    x_world_m: float
    y_world_m: float


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity command consumed by the frozen locomotion policy."""

    vx_mps: float
    vy_mps: float
    wz_radps: float


class Navigator(Protocol):
    """Minimal interface shared by classical and future RL navigators."""

    def reset(self) -> None:
        """Reset controller memory before a new episode."""

    def compute_command(
        self,
        robot: RobotState,
        goal: GoalState,
        dt_s: float,
    ) -> VelocityCommand:
        """Produce one body-frame velocity command."""
