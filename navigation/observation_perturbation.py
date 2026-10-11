"""Observation-side OOD perturbations for the high-level Navigator.

These wrappers deliberately modify only what the Navigator receives.  The
MuJoCo state, the frozen locomotion policy, and all benchmark truth metrics
remain untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import random
from typing import Any

from navigation.types import GoalState, Navigator, RobotState, VelocityCommand


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class ObservationPerturbation:
    """One factor applied to observations at Navigator decision time."""

    condition_id: str
    factor: str
    yaw_bias_rad: float = 0.0
    robot_x_bias_m: float = 0.0
    robot_y_bias_m: float = 0.0
    goal_x_bias_m: float = 0.0
    goal_y_bias_m: float = 0.0
    dropout_probability: float = 0.0
    dropout_seed: int = 0

    def validate(self) -> None:
        if not self.condition_id:
            raise ValueError("observation perturbation id must not be empty")
        supported = {
            "robot_yaw_bias",
            "robot_position_bias",
            "goal_position_bias",
            "robot_state_dropout",
        }
        if self.factor not in supported:
            raise ValueError(f"unsupported observation perturbation: {self.factor}")
        numerical = asdict(self)
        for name, value in numerical.items():
            if name in {"condition_id", "factor", "dropout_seed"}:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        active = {
            "robot_yaw_bias": bool(self.yaw_bias_rad),
            "robot_position_bias": bool(self.robot_x_bias_m or self.robot_y_bias_m),
            "goal_position_bias": bool(self.goal_x_bias_m or self.goal_y_bias_m),
            "robot_state_dropout": bool(self.dropout_probability),
        }
        if any(
            value
            for key, value in active.items()
            if key != self.factor
        ):
            raise ValueError("each OOD condition must alter exactly one factor")


class ObservationPerturbedNavigator:
    """Apply a declared observation perturbation before a Navigator call."""

    def __init__(
        self,
        navigator: Navigator,
        perturbation: ObservationPerturbation,
        *,
        episode_seed: int,
    ) -> None:
        perturbation.validate()
        self._navigator = navigator
        self._perturbation = perturbation
        self._episode_seed = episode_seed
        self._random: random.Random | None = None
        self._last_robot: RobotState | None = None
        self._last_input_robot: RobotState | None = None
        self._last_output_robot: RobotState | None = None
        self._last_input_goal: GoalState | None = None
        self._last_output_goal: GoalState | None = None
        self._decision_count = 0
        self._dropout_count = 0

    def reset(self) -> None:
        self._navigator.reset()
        self._random = random.Random(
            f"{self._perturbation.dropout_seed}:{self._episode_seed}"
        )
        self._last_robot = None
        self._last_input_robot = None
        self._last_output_robot = None
        self._last_input_goal = None
        self._last_output_goal = None
        self._decision_count = 0
        self._dropout_count = 0

    def compute_command(
        self,
        robot: RobotState,
        goal: GoalState,
        dt_s: float,
    ) -> VelocityCommand:
        observed_robot = self._observed_robot(robot)
        observed_goal = self._observed_goal(goal)
        self._last_input_robot = robot
        self._last_output_robot = observed_robot
        self._last_input_goal = goal
        self._last_output_goal = observed_goal
        return self._navigator.compute_command(observed_robot, observed_goal, dt_s)

    def _observed_robot(self, robot: RobotState) -> RobotState:
        self._decision_count += 1
        if self._perturbation.factor == "robot_state_dropout":
            if self._last_robot is not None:
                assert self._random is not None
                if self._random.random() < self._perturbation.dropout_probability:
                    self._dropout_count += 1
                    return self._last_robot
            self._last_robot = robot
            return robot
        if self._perturbation.factor == "robot_yaw_bias":
            return replace(
                robot,
                yaw_rad=_wrap_angle(robot.yaw_rad + self._perturbation.yaw_bias_rad),
            )
        if self._perturbation.factor == "robot_position_bias":
            return replace(
                robot,
                x_world_m=robot.x_world_m + self._perturbation.robot_x_bias_m,
                y_world_m=robot.y_world_m + self._perturbation.robot_y_bias_m,
            )
        return robot

    def _observed_goal(self, goal: GoalState) -> GoalState:
        if self._perturbation.factor != "goal_position_bias":
            return goal
        return replace(
            goal,
            x_world_m=goal.x_world_m + self._perturbation.goal_x_bias_m,
            y_world_m=goal.y_world_m + self._perturbation.goal_y_bias_m,
        )

    def readback(self) -> dict[str, Any]:
        if self._last_input_robot is None or self._last_output_robot is None:
            return {"condition": asdict(self._perturbation), "readback_passed": False}
        factor = self._perturbation.factor
        if factor == "robot_yaw_bias":
            applied = math.isclose(
                _wrap_angle(self._last_output_robot.yaw_rad - self._last_input_robot.yaw_rad),
                self._perturbation.yaw_bias_rad,
                abs_tol=1e-12,
            )
        elif factor == "robot_position_bias":
            applied = math.isclose(
                self._last_output_robot.x_world_m - self._last_input_robot.x_world_m,
                self._perturbation.robot_x_bias_m,
                abs_tol=1e-12,
            ) and math.isclose(
                self._last_output_robot.y_world_m - self._last_input_robot.y_world_m,
                self._perturbation.robot_y_bias_m,
                abs_tol=1e-12,
            )
        elif factor == "goal_position_bias":
            assert self._last_input_goal is not None and self._last_output_goal is not None
            applied = math.isclose(
                self._last_output_goal.x_world_m - self._last_input_goal.x_world_m,
                self._perturbation.goal_x_bias_m,
                abs_tol=1e-12,
            ) and math.isclose(
                self._last_output_goal.y_world_m - self._last_input_goal.y_world_m,
                self._perturbation.goal_y_bias_m,
                abs_tol=1e-12,
            )
        else:
            applied = (
                self._perturbation.dropout_probability == 0.0
                or self._dropout_count > 0
            )
        return {
            "condition": asdict(self._perturbation),
            "navigator_decision_count": self._decision_count,
            "dropout_count": self._dropout_count,
            "dropout_fraction": (
                self._dropout_count / max(1, self._decision_count - 1)
                if self._perturbation.factor == "robot_state_dropout"
                else 0.0
            ),
            "readback_passed": self._decision_count > 0 and applied,
        }
