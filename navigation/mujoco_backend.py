"""MuJoCo/BAM/ONNX backend for the PointGoal pilot.

The backend is intentionally the only navigation module that touches the
simulator-backed runtime.  Navigators consume ``RobotState`` only.
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.locomotion_benchmark import (
    FALL_TILT_DEG,
    BenchmarkConfig,
    _advance_control_step,
    _base_state,
    create_runtime,
    validate_artifacts,
)
from navigation.types import RobotState, VelocityCommand


@dataclass(frozen=True)
class BackendStep:
    state: RobotState
    observation: tuple[float, ...]
    action: tuple[float, ...]


class MujocoBackend:
    """Fresh-episode deployment rehearsal using the official inference stack."""

    def __init__(
        self,
        microduck_rl_root: Path,
        policy_path: Path,
        metadata_path: Path,
        output_dir: Path,
    ) -> None:
        self._benchmark_config = BenchmarkConfig(
            microduck_rl_root=microduck_rl_root.resolve(),
            policy_path=policy_path.resolve(),
            command_set_path=metadata_path.resolve(),
            output_dir=output_dir.resolve(),
            seed=0,
        )
        self._runtime: Any | None = None
        self._elapsed_s = 0.0
        self._last_command: VelocityCommand | None = None

    @property
    def control_dt_s(self) -> float:
        self._require_runtime()
        return float(self._runtime.control_dt)

    def validate(self) -> None:
        validate_artifacts(self._benchmark_config)

    def reset(self, episode_seed: int, initial_state_mode: str) -> RobotState:
        self.validate()
        self._runtime, _ = create_runtime(
            self._benchmark_config,
            episode_seed,
            initial_state_mode,
        )
        self._elapsed_s = 0.0
        self._last_command = VelocityCommand(0.0, 0.0, 0.0)
        return self.observe()

    def observe(self) -> RobotState:
        self._require_runtime()
        return self._to_robot_state(_base_state(self._runtime))

    def step(self, command: VelocityCommand) -> BackendStep:
        self._require_runtime()
        if command != self._last_command:
            with contextlib.redirect_stdout(io.StringIO()):
                self._runtime.policy.set_vel_cmd(
                    command.vx_mps,
                    command.vy_mps,
                    command.wz_radps,
                )
            self._last_command = command
        observation, action, base_state = _advance_control_step(self._runtime)
        self._elapsed_s += self._runtime.control_dt
        return BackendStep(
            state=self._to_robot_state(base_state),
            observation=tuple(float(value) for value in observation),
            action=tuple(float(value) for value in action),
        )

    def _to_robot_state(self, state: dict[str, Any]) -> RobotState:
        self._require_runtime()
        np = self._runtime.official.np
        finite = bool(
            np.isfinite(self._runtime.data.qpos).all()
            and np.isfinite(self._runtime.data.qvel).all()
            and np.isfinite(state["position"]).all()
            and np.isfinite(state["body_velocity"]).all()
        )
        return RobotState(
            time_s=self._elapsed_s,
            x_world_m=float(state["position"][0]),
            y_world_m=float(state["position"][1]),
            z_world_m=float(state["position"][2]),
            yaw_rad=float(state["yaw"]),
            roll_rad=float(state["roll"]),
            pitch_rad=float(state["pitch"]),
            vx_body_mps=float(state["body_velocity"][0]),
            vy_body_mps=float(state["body_velocity"][1]),
            wz_body_radps=float(state["yaw_rate"]),
            tilt_rad=float(state["tilt"]),
            fallen=(not finite) or state["tilt"] >= math.radians(FALL_TILT_DEG),
            finite=finite,
        )

    def _require_runtime(self) -> None:
        if self._runtime is None:
            raise RuntimeError("backend must be reset before use")
