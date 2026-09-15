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
from navigation.types import GoalState, RobotState, VelocityCommand


@dataclass(frozen=True)
class BackendStep:
    state: RobotState
    observation: tuple[float, ...]
    action: tuple[float, ...]


class ViewerSession:
    """Native MuJoCo viewer with PointGoal markers and a tracking camera."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._viewer_context: Any | None = None
        self._viewer: Any | None = None

    def __enter__(self) -> "ViewerSession":
        self._viewer_context = self._runtime.official.mujoco.viewer.launch_passive(
            self._runtime.model,
            self._runtime.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self._viewer = self._viewer_context.__enter__()
        mujoco = self._runtime.official.mujoco
        trunk_id = mujoco.mj_name2id(
            self._runtime.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "trunk_base",
        )
        with self._viewer.lock():
            if trunk_id >= 0:
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._viewer.cam.trackbodyid = trunk_id
            self._viewer.cam.distance = 2.5
            self._viewer.cam.azimuth = 135.0
            self._viewer.cam.elevation = -35.0
        self.sync()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._viewer_context is not None:
            self._viewer_context.__exit__(exc_type, exc_value, traceback)
        self._viewer = None
        self._viewer_context = None

    def is_running(self) -> bool:
        return self._viewer is not None and bool(self._viewer.is_running())

    def sync(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def set_goal(self, goal: GoalState, success_radius_m: float) -> None:
        """Draw a target beacon and a translucent success zone."""

        if self._viewer is None:
            raise RuntimeError("viewer session is not open")
        np = self._runtime.official.np
        mujoco = self._runtime.official.mujoco
        position = np.asarray([goal.x_world_m, goal.y_world_m, 0.025], dtype=float)
        identity = np.eye(3, dtype=float).reshape(-1)
        with self._viewer.lock():
            scene = self._viewer.user_scn
            scene.ngeom = 2
            mujoco.mjv_initGeom(
                scene.geoms[0],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.asarray([success_radius_m, 0.012, 0.0], dtype=float),
                position,
                identity,
                np.asarray([0.1, 0.8, 0.25, 0.28], dtype=float),
            )
            mujoco.mjv_initGeom(
                scene.geoms[1],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray([0.055, 0.0, 0.0], dtype=float),
                position + np.asarray([0.0, 0.0, 0.09], dtype=float),
                identity,
                np.asarray([1.0, 0.25, 0.05, 1.0], dtype=float),
            )
        self.sync()


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

    def open_viewer(self) -> ViewerSession:
        """Create a viewer for the current episode without exposing simulator state."""

        self._require_runtime()
        return ViewerSession(self._runtime)

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
