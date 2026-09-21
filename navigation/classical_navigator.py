"""Classical go-to-goal controllers sharing the PointGoal Navigator contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from navigation.types import GoalState, RobotState, VelocityCommand


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def goal_error(robot: RobotState, goal: GoalState) -> tuple[float, float]:
    """Return planar distance and signed heading error to a positional goal."""

    dx = goal.x_world_m - robot.x_world_m
    dy = goal.y_world_m - robot.y_world_m
    distance = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    return distance, wrap_angle(heading - robot.yaw_rad)


@dataclass(frozen=True)
class ConstrainedGoToGoalConfig:
    """Smoke-stage controller parameters that must be frozen before the pilot."""

    distance_gain: float = 0.8
    heading_gain: float = 1.2
    max_forward_speed_mps: float = 0.25
    min_turn_speed_mps: float = 0.05
    max_yaw_rate_radps: float = 0.5
    max_forward_accel_mps2: float = 0.5
    max_yaw_accel_radps2: float = 1.0
    heading_slowdown_floor: float = 0.2
    goal_tolerance_m: float = 0.2
    arrival_hold_time_s: float = 0.5

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("controller parameters must be finite")
        positive = (
            "distance_gain",
            "heading_gain",
            "max_forward_speed_mps",
            "max_yaw_rate_radps",
            "max_forward_accel_mps2",
            "max_yaw_accel_radps2",
            "goal_tolerance_m",
            "arrival_hold_time_s",
        )
        for name in positive:
            if values[name] <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 <= self.min_turn_speed_mps <= self.max_forward_speed_mps:
            raise ValueError(
                "min_turn_speed_mps must be between 0 and max_forward_speed_mps"
            )
        if not 0 <= self.heading_slowdown_floor <= 1:
            raise ValueError("heading_slowdown_floor must be between 0 and 1")


@dataclass(frozen=True)
class NaivePConfig:
    """Plain proportional PointGoal baseline without smoothing or heuristics."""

    distance_gain: float = 0.8
    heading_gain: float = 1.2
    max_forward_speed_mps: float = 0.25
    max_yaw_rate_radps: float = 0.5
    goal_tolerance_m: float = 0.2

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("controller parameters must be finite")
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rate_limit(current: float, target: float, max_delta: float) -> float:
    return current + _clamp(target - current, -max_delta, max_delta)


class NaivePNavigator:
    """Distance/heading P controller with only command bounds and goal tolerance."""

    def __init__(self, config: NaivePConfig) -> None:
        config.validate()
        self.config = config

    def reset(self) -> None:
        """The memoryless baseline has no state to clear."""

    def compute_command(
        self,
        robot: RobotState,
        goal: GoalState,
        dt_s: float,
    ) -> VelocityCommand:
        if not robot.finite:
            raise ValueError("cannot navigate from a non-finite RobotState")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and > 0")

        distance, heading_error = goal_error(robot, goal)
        if distance <= self.config.goal_tolerance_m:
            return VelocityCommand(0.0, 0.0, 0.0)
        return VelocityCommand(
            vx_mps=_clamp(
                self.config.distance_gain * distance,
                0.0,
                self.config.max_forward_speed_mps,
            ),
            vy_mps=0.0,
            wz_radps=_clamp(
                self.config.heading_gain * heading_error,
                -self.config.max_yaw_rate_radps,
                self.config.max_yaw_rate_radps,
            ),
        )


class ConstrainedGoToGoalNavigator:
    """P controller with command bounds, curvature and slew-rate constraints."""

    def __init__(self, config: ConstrainedGoToGoalConfig) -> None:
        config.validate()
        self.config = config
        self._previous = VelocityCommand(0.0, 0.0, 0.0)
        self._goal_reached = False
        self._inside_goal_time_s = 0.0

    def reset(self) -> None:
        self._previous = VelocityCommand(0.0, 0.0, 0.0)
        self._goal_reached = False
        self._inside_goal_time_s = 0.0

    def compute_command(
        self,
        robot: RobotState,
        goal: GoalState,
        dt_s: float,
    ) -> VelocityCommand:
        if not robot.finite:
            raise ValueError("cannot navigate from a non-finite RobotState")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and > 0")

        distance, heading_error = goal_error(robot, goal)
        inside_goal = distance <= self.config.goal_tolerance_m
        self._inside_goal_time_s = (
            self._inside_goal_time_s + dt_s if inside_goal else 0.0
        )
        if self._inside_goal_time_s + 1e-12 >= self.config.arrival_hold_time_s:
            self._goal_reached = True
        if inside_goal or self._goal_reached:
            target_vx = 0.0
            target_wz = 0.0
        else:
            unconstrained_vx = min(
                self.config.max_forward_speed_mps,
                self.config.distance_gain * distance,
            )
            heading_scale = max(
                self.config.heading_slowdown_floor,
                max(0.0, math.cos(heading_error)),
            )
            target_vx = unconstrained_vx * heading_scale
            if abs(heading_error) >= math.pi / 3:
                target_vx = max(target_vx, self.config.min_turn_speed_mps)
            target_wz = _clamp(
                self.config.heading_gain * heading_error,
                -self.config.max_yaw_rate_radps,
                self.config.max_yaw_rate_radps,
            )

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
