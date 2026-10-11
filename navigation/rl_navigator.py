"""RL PointGoal navigator with the same runtime contract as the classical baseline.

The policy only decides the high-level body velocity command.  The frozen
locomotion policy remains responsible for converting ``[vx, 0, wz]`` into the
14 joint actions used by MicroDuck.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from navigation.types import GoalState, RobotState, VelocityCommand


class NavigationPolicy(Protocol):
    """Minimal inference interface accepted by :class:`RLNavigator`."""

    def __call__(self, observation: Sequence[float]) -> Sequence[float]:
        """Return the two normalized actions ``[forward, yaw]``."""


@dataclass(frozen=True)
class RLNavigatorConfig:
    """Frozen command limits shared by training, deployment and evaluation."""

    max_forward_speed_mps: float = 0.25
    max_yaw_rate_radps: float = 0.5
    max_forward_accel_mps2: float = 0.5
    max_yaw_accel_radps2: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")


def pointgoal_observation(
    robot: RobotState, goal: GoalState
) -> tuple[float, float, float, float, float, float, float]:
    """Build the frozen seven-dimensional high-level observation.

    Layout::

      [dx_body, dy_body, vx_body, vy_body, wz_body, previous_cmd_vx,
       previous_cmd_wz]

    The last two entries are filled by :class:`RLNavigator`; this helper leaves
    them at zero so the geometric and measured-state part can also be tested in
    isolation.  Distance and heading error are intentionally not duplicated:
    both can be derived from ``dx_body`` and ``dy_body``.
    """

    if not robot.finite:
        raise ValueError("cannot navigate from a non-finite RobotState")

    dx_world = goal.x_world_m - robot.x_world_m
    dy_world = goal.y_world_m - robot.y_world_m
    cos_yaw = math.cos(robot.yaw_rad)
    sin_yaw = math.sin(robot.yaw_rad)
    dx_body = cos_yaw * dx_world + sin_yaw * dy_world
    dy_body = -sin_yaw * dx_world + cos_yaw * dy_world
    observation = (
        dx_body,
        dy_body,
        robot.vx_body_mps,
        robot.vy_body_mps,
        robot.wz_body_radps,
        0.0,
        0.0,
    )
    if not all(math.isfinite(value) for value in observation):
        raise ValueError("RL Navigator observation must be finite")
    return observation


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rate_limit(current: float, target: float, max_delta: float) -> float:
    return current + _clamp(target - current, -max_delta, max_delta)


class RLNavigator:
    """Policy-backed navigator with bounded and rate-limited output commands.

    The navigator deliberately contains no near-goal braking rule or heading
    heuristic.  Slowing, turning and stopping remain behavior learned by the
    high-level policy; only the shared actuator-facing safety envelope is
    enforced here.
    """

    def __init__(self, policy: NavigationPolicy, config: RLNavigatorConfig) -> None:
        config.validate()
        self.policy = policy
        self.config = config
        self._previous = VelocityCommand(0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._previous = VelocityCommand(0.0, 0.0, 0.0)

    def compute_command(
        self,
        robot: RobotState,
        goal: GoalState,
        dt_s: float,
    ) -> VelocityCommand:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and > 0")

        observation_base = pointgoal_observation(robot, goal)
        observation = (
            *observation_base[:5],
            self._previous.vx_mps,
            self._previous.wz_radps,
        )

        raw_action = tuple(float(value) for value in self.policy(observation))
        if len(raw_action) != 2:
            raise ValueError(
                f"RL Navigator policy must return 2 actions, got {len(raw_action)}"
            )
        if not all(math.isfinite(value) for value in raw_action):
            raise ValueError("RL Navigator policy action must be finite")

        normalized_vx = _clamp(float(raw_action[0]), -1.0, 1.0)
        normalized_wz = _clamp(float(raw_action[1]), -1.0, 1.0)
        target_vx = 0.5 * (normalized_vx + 1.0) * self.config.max_forward_speed_mps
        target_wz = normalized_wz * self.config.max_yaw_rate_radps

        command = VelocityCommand(
            vx_mps=_rate_limit(
                self._previous.vx_mps,
                target_vx,
                self.config.max_forward_accel_mps2 * dt_s,
            ),
            vy_mps=0.0,
            wz_radps=_rate_limit(
                self._previous.wz_radps,
                target_wz,
                self.config.max_yaw_accel_radps2 * dt_s,
            ),
        )
        self._previous = command
        return command
