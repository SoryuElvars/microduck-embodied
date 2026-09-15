#!/usr/bin/env python3
"""Run the lightweight Classical PointGoal pilot with a frozen ONNX policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.locomotion_benchmark import (
    BAM_VIN,
    BAM_VIN_DROP_GAIN,
    DECIMATION,
    EXPECTED_ACTION_SIZE,
    EXPECTED_OBSERVATION_SIZE,
    PHYSICS_TIMESTEP_S,
    PROJECT_ROOT,
    path_for_record,
)
from navigation.classical_navigator import (
    ConstrainedGoToGoalConfig,
    ConstrainedGoToGoalNavigator,
    goal_error,
)
from navigation.mujoco_backend import BackendStep, MujocoBackend, ViewerSession
from navigation.types import GoalState, RobotState, VelocityCommand


DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "artifacts"
    / "week03"
    / "02_angular_std025_candidate"
    / "models"
    / "model_2500.onnx"
)
DEFAULT_GOAL_SET = Path(__file__).parent / "goal_sets" / "pointgoal_pilot_5.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "week03"
    / "04_classical_pointgoal_pilot"
    / "model_2500"
)
PROTOCOL_STATUSES = {"draft", "frozen"}
MIRROR_SIDES = {"left", "right"}


@dataclass(frozen=True)
class GoalCase:
    name: str
    x_initial_body_m: float
    y_initial_body_m: float
    episodes: int
    mirror_group: str | None
    mirror_side: str | None


@dataclass(frozen=True)
class PilotProtocol:
    source_path: Path
    protocol_status: str
    description: str
    base_seed: int
    initial_state_mode: str
    warmup_s: float
    timeout_s: float
    smoke_timeout_s: float
    success_radius_m: float
    success_hold_s: float
    navigator_frequency_hz: float
    controller: ConstrainedGoToGoalConfig
    goals: tuple[GoalCase, ...]

    @property
    def episode_count(self) -> int:
        return sum(goal.episodes for goal in self.goals)


def _finite_positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    number = float(value)
    minimum_ok = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not minimum_ok:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {relation}")
    return number


def load_goal_set(path: Path) -> PilotProtocol:
    """Load and validate the versioned PointGoal pilot protocol."""

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError("goal set schema_version must be 1")

    status = str(document.get("protocol_status", ""))
    if status not in PROTOCOL_STATUSES:
        raise ValueError(f"protocol_status must be one of: {sorted(PROTOCOL_STATUSES)}")
    defaults = document.get("defaults")
    controller_raw = document.get("controller")
    raw_goals = document.get("goals")
    if not isinstance(defaults, dict):
        raise ValueError("goal set defaults must be an object")
    if not isinstance(controller_raw, dict):
        raise ValueError("goal set controller must be an object")
    if not isinstance(raw_goals, list) or not raw_goals:
        raise ValueError("goal set goals must be a non-empty list")

    success_radius = _finite_positive(
        "success_radius_m", defaults.get("success_radius_m")
    )
    controller = ConstrainedGoToGoalConfig(
        **controller_raw,
        goal_tolerance_m=success_radius,
    )
    controller.validate()

    default_episodes = int(defaults.get("episodes", 0))
    if default_episodes < 1:
        raise ValueError("defaults.episodes must be >= 1")
    goals: list[GoalCase] = []
    for raw in raw_goals:
        episodes = int(raw.get("episodes", default_episodes))
        if episodes < 1:
            raise ValueError("goal episodes must be >= 1")
        mirror_group = raw.get("mirror_group")
        mirror_side = raw.get("mirror_side")
        if (mirror_group is None) != (mirror_side is None):
            raise ValueError("mirror_group and mirror_side must both be set or both be null")
        if mirror_side is not None and mirror_side not in MIRROR_SIDES:
            raise ValueError(f"mirror_side must be one of: {sorted(MIRROR_SIDES)}")
        goal = GoalCase(
            name=str(raw.get("name", "")),
            x_initial_body_m=float(raw.get("x_initial_body_m")),
            y_initial_body_m=float(raw.get("y_initial_body_m")),
            episodes=episodes,
            mirror_group=None if mirror_group is None else str(mirror_group),
            mirror_side=None if mirror_side is None else str(mirror_side),
        )
        if not goal.name.strip():
            raise ValueError("goal names must not be empty")
        if not all(
            math.isfinite(value)
            for value in (goal.x_initial_body_m, goal.y_initial_body_m)
        ):
            raise ValueError(f"goal {goal.name!r} contains a non-finite coordinate")
        if math.hypot(goal.x_initial_body_m, goal.y_initial_body_m) <= success_radius:
            raise ValueError(f"goal {goal.name!r} must begin outside the success radius")
        goals.append(goal)

    names = [goal.name for goal in goals]
    if len(names) != len(set(names)):
        raise ValueError("goal names must be unique")
    mirror_members: dict[str, set[str]] = {}
    for goal in goals:
        if goal.mirror_group is not None:
            mirror_members.setdefault(goal.mirror_group, set()).add(str(goal.mirror_side))
    if any(sides != MIRROR_SIDES for sides in mirror_members.values()):
        raise ValueError("each mirror_group must contain one left and one right goal")

    return PilotProtocol(
        source_path=path.resolve(),
        protocol_status=status,
        description=str(document.get("description", "")),
        base_seed=int(defaults.get("base_seed", 0)),
        initial_state_mode=str(defaults.get("initial_state_mode", "")),
        warmup_s=_finite_positive("warmup_s", defaults.get("warmup_s")),
        timeout_s=_finite_positive("timeout_s", defaults.get("timeout_s")),
        smoke_timeout_s=_finite_positive(
            "smoke_timeout_s", defaults.get("smoke_timeout_s")
        ),
        success_radius_m=success_radius,
        success_hold_s=_finite_positive(
            "success_hold_s", defaults.get("success_hold_s")
        ),
        navigator_frequency_hz=_finite_positive(
            "navigator_frequency_hz", defaults.get("navigator_frequency_hz")
        ),
        controller=controller,
        goals=tuple(goals),
    )


def goal_from_initial_body_frame(
    robot: RobotState,
    x_initial_body_m: float,
    y_initial_body_m: float,
) -> GoalState:
    """Transform one goal from the episode's initial body frame to world frame."""

    cosine = math.cos(robot.yaw_rad)
    sine = math.sin(robot.yaw_rad)
    return GoalState(
        x_world_m=robot.x_world_m + cosine * x_initial_body_m - sine * y_initial_body_m,
        y_world_m=robot.y_world_m + sine * x_initial_body_m + cosine * y_initial_body_m,
    )


def select_run_goals(
    protocol: PilotProtocol,
    *,
    smoke: bool,
    smoke_goal: str,
) -> tuple[GoalCase, ...]:
    """Choose the run plan and prevent accidental formal runs of draft protocols."""

    if not smoke:
        if protocol.protocol_status != "frozen":
            raise ValueError(
                "formal pilot requires protocol_status='frozen'; run --smoke first, "
                "review the trajectory, then freeze the goal-set parameters"
            )
        return protocol.goals
    for goal in protocol.goals:
        if goal.name == smoke_goal:
            return (replace(goal, episodes=1),)
    raise ValueError(f"unknown smoke goal: {smoke_goal!r}")


def select_named_goal(protocol: PilotProtocol, goal_name: str) -> GoalCase:
    """Return one configured goal for an interactive viewer run."""

    for goal in protocol.goals:
        if goal.name == goal_name:
            return replace(goal, episodes=1)
    raise ValueError(f"unknown goal: {goal_name!r}")


def _step_row(
    phase: str,
    episode_seed: int,
    goal_case: GoalCase,
    goal: GoalState | None,
    command: VelocityCommand,
    step: BackendStep,
) -> dict[str, Any]:
    distance = ""
    heading_error = ""
    if goal is not None:
        distance, heading_error = goal_error(step.state, goal)
    row: dict[str, Any] = {
        "phase": phase,
        "time_s": step.state.time_s,
        "seed": episode_seed,
        "goal_name": goal_case.name,
        "goal_world_x_m": "" if goal is None else goal.x_world_m,
        "goal_world_y_m": "" if goal is None else goal.y_world_m,
        "distance_to_goal_m": distance,
        "heading_error_rad": heading_error,
        "cmd_vx": command.vx_mps,
        "cmd_vy": command.vy_mps,
        "cmd_wz": command.wz_radps,
        "actual_vx": step.state.vx_body_mps,
        "actual_vy": step.state.vy_body_mps,
        "actual_wz": step.state.wz_body_radps,
        "world_x": step.state.x_world_m,
        "world_y": step.state.y_world_m,
        "trunk_z": step.state.z_world_m,
        "roll_rad": step.state.roll_rad,
        "pitch_rad": step.state.pitch_rad,
        "yaw_rad": step.state.yaw_rad,
        "tilt_rad": step.state.tilt_rad,
        "fallen": step.state.fallen,
        "finite": step.state.finite,
    }
    row.update(
        {f"obs_{index:02d}": value for index, value in enumerate(step.observation)}
    )
    row.update({f"action_{index:02d}": value for index, value in enumerate(step.action)})
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _step_episode(
    backend: MujocoBackend,
    command: VelocityCommand,
    viewer: ViewerSession | None,
    *,
    realtime: bool,
) -> BackendStep:
    """Advance one control step and optionally pace/sync an interactive viewer."""

    started_at = time.perf_counter()
    step = backend.step(command)
    if viewer is not None:
        viewer.sync()
    if realtime:
        remaining_s = backend.control_dt_s - (time.perf_counter() - started_at)
        if remaining_s > 0:
            time.sleep(remaining_s)
    return step


def run_episode(
    backend: MujocoBackend,
    protocol: PilotProtocol,
    goal_case: GoalCase,
    episode_index: int,
    output_dir: Path,
    *,
    smoke: bool,
    episode_seed: int | None = None,
    reset_backend: bool = True,
    viewer: ViewerSession | None = None,
    realtime: bool = False,
) -> dict[str, Any]:
    """Run one closed-loop PointGoal episode and write its raw trajectory."""

    episode_seed = (
        protocol.base_seed + episode_index if episode_seed is None else episode_seed
    )
    if reset_backend:
        backend.reset(episode_seed, protocol.initial_state_mode)
    navigator = ConstrainedGoToGoalNavigator(protocol.controller)
    zero_command = VelocityCommand(0.0, 0.0, 0.0)
    rows: list[dict[str, Any]] = []

    warmup_steps = round(protocol.warmup_s / backend.control_dt_s)
    viewer_closed = False
    for _ in range(warmup_steps):
        if viewer is not None and not viewer.is_running():
            viewer_closed = True
            break
        step = _step_episode(backend, zero_command, viewer, realtime=realtime)
        rows.append(
            _step_row("warmup", episode_seed, goal_case, None, zero_command, step)
        )
        if step.state.fallen:
            break

    start = backend.observe()
    goal = goal_from_initial_body_frame(
        start,
        goal_case.x_initial_body_m,
        goal_case.y_initial_body_m,
    )
    if viewer is not None and viewer.is_running():
        viewer.set_goal(goal, protocol.success_radius_m)
    initial_distance, _ = goal_error(start, goal)
    previous_x = start.x_world_m
    previous_y = start.y_world_m
    path_length = 0.0
    minimum_distance = initial_distance
    success_steps = 0
    required_success_steps = max(
        1, math.ceil(protocol.success_hold_s / backend.control_dt_s)
    )
    navigator_interval_steps = max(
        1,
        round(1.0 / (protocol.navigator_frequency_hz * backend.control_dt_s)),
    )
    navigator_dt_s = navigator_interval_steps * backend.control_dt_s
    command = zero_command
    termination_reason = (
        "fall" if start.fallen else "viewer_closed" if viewer_closed else "timeout"
    )
    timeout_s = protocol.smoke_timeout_s if smoke else protocol.timeout_s
    test_steps = round(timeout_s / backend.control_dt_s)
    test_rows: list[dict[str, Any]] = []

    navigator.reset()
    if not start.fallen and not viewer_closed:
        for control_step in range(test_steps):
            if viewer is not None and not viewer.is_running():
                termination_reason = "viewer_closed"
                break
            if control_step % navigator_interval_steps == 0:
                command = navigator.compute_command(backend.observe(), goal, navigator_dt_s)
            step = _step_episode(backend, command, viewer, realtime=realtime)
            test_rows.append(
                _step_row("test", episode_seed, goal_case, goal, command, step)
            )
            path_length += math.hypot(
                step.state.x_world_m - previous_x,
                step.state.y_world_m - previous_y,
            )
            previous_x = step.state.x_world_m
            previous_y = step.state.y_world_m
            distance, _ = goal_error(step.state, goal)
            minimum_distance = min(minimum_distance, distance)
            if step.state.fallen:
                termination_reason = "invalid_state" if not step.state.finite else "fall"
                break
            success_steps = success_steps + 1 if distance <= protocol.success_radius_m else 0
            if success_steps >= required_success_steps:
                termination_reason = "success"
                break

    rows.extend(test_rows)
    raw_path = (
        output_dir
        / "raw"
        / f"{goal_case.name}_seed{episode_seed}_episode{episode_index:03d}_steps.csv"
    )
    _write_rows(raw_path, rows)

    final = backend.observe()
    final_distance, final_heading_error = goal_error(final, goal)
    completed_s = len(test_rows) * backend.control_dt_s
    success = termination_reason == "success"
    max_vx = protocol.controller.max_forward_speed_mps
    max_wz = protocol.controller.max_yaw_rate_radps
    vx_saturated = sum(abs(float(row["cmd_vx"])) >= max_vx - 1e-9 for row in test_rows)
    wz_saturated = sum(abs(float(row["cmd_wz"])) >= max_wz - 1e-9 for row in test_rows)
    denominator = max(1, len(test_rows))
    path_efficiency = (
        initial_distance / max(path_length, initial_distance) if success else None
    )
    progress_efficiency = (
        max(0.0, initial_distance - minimum_distance) / path_length
        if path_length > 0
        else 0.0
    )
    return {
        "goal_name": goal_case.name,
        "mirror_group": goal_case.mirror_group,
        "mirror_side": goal_case.mirror_side,
        "episode": episode_index,
        "seed": episode_seed,
        "success": success,
        "termination_reason": termination_reason,
        "duration_s": completed_s,
        "initial_goal_distance_m": initial_distance,
        "final_distance_m": final_distance,
        "minimum_distance_m": minimum_distance,
        "final_heading_error_rad": final_heading_error,
        "path_length_m": path_length,
        "path_efficiency": path_efficiency,
        "progress_efficiency": progress_efficiency,
        "fall": termination_reason == "fall",
        "invalid_state": termination_reason == "invalid_state",
        "timeout": termination_reason == "timeout",
        "viewer_closed": termination_reason == "viewer_closed",
        "vx_saturation_fraction": vx_saturated / denominator,
        "wz_saturation_fraction": wz_saturated / denominator,
        "start_state": asdict(start),
        "goal_world": asdict(goal),
        "final_state": asdict(final),
        "raw_steps_csv": str(raw_path.relative_to(PROJECT_ROOT)),
    }


def _median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pilot outcomes without merging mirrored target categories."""

    by_goal: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_goal.setdefault(str(episode["goal_name"]), []).append(episode)
    goal_summaries = {
        name: {
            "episodes": len(items),
            "success_rate": fmean(float(item["success"]) for item in items),
            "fall_rate": fmean(float(item["fall"]) for item in items),
            "timeout_rate": fmean(float(item["timeout"]) for item in items),
            "median_final_distance_m": median(item["final_distance_m"] for item in items),
            "median_path_efficiency_successes": _median_or_none(
                [item["path_efficiency"] for item in items if item["success"]]
            ),
            "median_progress_efficiency": median(
                item["progress_efficiency"] for item in items
            ),
            "median_completion_time_s_successes": _median_or_none(
                [item["duration_s"] for item in items if item["success"]]
            ),
        }
        for name, items in by_goal.items()
    }

    mirror_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for episode in episodes:
        group = episode.get("mirror_group")
        side = episode.get("mirror_side")
        if group is not None and side is not None:
            mirror_groups.setdefault(str(group), {}).setdefault(str(side), []).append(episode)
    mirror_summary: dict[str, Any] = {}
    for group, sides in mirror_groups.items():
        left = sides.get("left", [])
        right = sides.get("right", [])
        if not left or not right:
            mirror_summary[group] = {
                "complete": False,
                "present_sides": sorted(sides),
            }
            continue
        left_success = fmean(float(item["success"]) for item in left)
        right_success = fmean(float(item["success"]) for item in right)
        mirror_summary[group] = {
            "complete": True,
            "left_success_rate": left_success,
            "right_success_rate": right_success,
            "left_minus_right_success_rate": left_success - right_success,
            "left_median_final_distance_m": median(
                item["final_distance_m"] for item in left
            ),
            "right_median_final_distance_m": median(
                item["final_distance_m"] for item in right
            ),
        }

    return {
        "episode_count": len(episodes),
        "success_rate": fmean(float(item["success"]) for item in episodes),
        "fall_rate": fmean(float(item["fall"]) for item in episodes),
        "timeout_rate": fmean(float(item["timeout"]) for item in episodes),
        "median_final_distance_m": median(
            item["final_distance_m"] for item in episodes
        ),
        "median_path_efficiency_successes": _median_or_none(
            [item["path_efficiency"] for item in episodes if item["success"]]
        ),
        "median_progress_efficiency": median(
            item["progress_efficiency"] for item in episodes
        ),
        "goals": goal_summaries,
        "mirror_groups": mirror_summary,
    }


def write_summary(
    protocol: PilotProtocol,
    policy_path: Path,
    microduck_rl_root: Path,
    output_dir: Path,
    episodes: list[dict[str, Any]],
    *,
    smoke: bool,
) -> Path:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke else "pilot"
    run_label = f"{suffix}_{episodes[0]['goal_name']}" if smoke else suffix
    path = summary_dir / f"pointgoal_{run_label}_{policy_path.stem}.json"
    official_commit = subprocess.run(
        ["git", "-C", str(microduck_rl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "classical_pointgoal_onnx_bam_pilot",
        "run_mode": suffix,
        "protocol_status": protocol.protocol_status,
        "pose_source": "simulator_ground_truth_via_backend",
        "reward_available": False,
        "policy_path": path_for_record(policy_path, microduck_rl_root, PROJECT_ROOT),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "microduck_rl_commit": official_commit,
        "goal_set_path": str(protocol.source_path.relative_to(PROJECT_ROOT)),
        "runtime": {
            "observation_size": EXPECTED_OBSERVATION_SIZE,
            "action_size": EXPECTED_ACTION_SIZE,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "decimation": DECIMATION,
            "locomotion_frequency_hz": 1.0 / (PHYSICS_TIMESTEP_S * DECIMATION),
            "navigator_frequency_hz": protocol.navigator_frequency_hz,
            "actuator": "BAM M6 XL330",
            "bam_vin": BAM_VIN,
            "bam_vin_drop_gain": BAM_VIN_DROP_GAIN,
            "state_latency_ms": 0,
            "command_latency_ms": 0,
        },
        "protocol": {
            "initial_state_mode": protocol.initial_state_mode,
            "base_seed": protocol.base_seed,
            "warmup_s": protocol.warmup_s,
            "timeout_s": protocol.smoke_timeout_s if smoke else protocol.timeout_s,
            "success_radius_m": protocol.success_radius_m,
            "success_hold_s": protocol.success_hold_s,
            "controller": asdict(protocol.controller),
        },
        "aggregate": aggregate_episodes(episodes),
        "episodes": episodes,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, smoke-test, or run the Classical PointGoal pilot."
    )
    parser.add_argument(
        "--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--goal-set", type=Path, default=DEFAULT_GOAL_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run one short episode even while the protocol is still draft",
    )
    run_mode.add_argument(
        "--view",
        action="store_true",
        help="Run one real-time episode in the native MuJoCo viewer",
    )
    parser.add_argument("--smoke-goal", default="front")
    parser.add_argument("--view-goal", default="front")
    parser.add_argument("--view-seed", type=int, default=42)
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    policy_path = _resolved(args.policy)
    goal_set_path = _resolved(args.goal_set)
    output_dir = _resolved(args.output_dir)
    microduck_rl_root = _resolved(args.microduck_rl_root)
    protocol = load_goal_set(goal_set_path)
    backend = MujocoBackend(
        microduck_rl_root=microduck_rl_root,
        policy_path=policy_path,
        metadata_path=goal_set_path,
        output_dir=output_dir,
    )
    backend.validate()

    print("MicroDuck Classical PointGoal pilot contract is valid")
    print(f"  protocol status:   {protocol.protocol_status}")
    print(f"  pose source:       simulator ground truth via MujocoBackend")
    print(f"  policy:            {policy_path}")
    print(f"  goal set:          {goal_set_path}")
    print(f"  planned episodes:  {protocol.episode_count}")
    print(f"  expected I/O:      {EXPECTED_OBSERVATION_SIZE} obs -> {EXPECTED_ACTION_SIZE} actions")
    if args.validate_only:
        return 0

    if args.view:
        try:
            goal_case = select_named_goal(protocol, args.view_goal)
        except ValueError as error:
            parser.error(str(error))
        backend.reset(args.view_seed, protocol.initial_state_mode)
        print(
            f"viewer: goal={goal_case.name}, seed={args.view_seed}; "
            "orange marker=goal, green disk=success radius"
        )
        try:
            with backend.open_viewer() as viewer:
                result = run_episode(
                    backend,
                    protocol,
                    goal_case,
                    0,
                    output_dir / "viewer",
                    smoke=False,
                    episode_seed=args.view_seed,
                    reset_backend=False,
                    viewer=viewer,
                    realtime=True,
                )
                print(
                    f"viewer result: {result['termination_reason']}, "
                    f"final_distance={result['final_distance_m']:.3f} m"
                )
                print("Episode finished; close the MuJoCo viewer window to exit")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
        except KeyboardInterrupt:
            print("viewer interrupted")
        return 0

    try:
        selected_goals = select_run_goals(
            protocol,
            smoke=args.smoke,
            smoke_goal=args.smoke_goal,
        )
    except ValueError as error:
        parser.error(str(error))
    episode_results: list[dict[str, Any]] = []
    for goal_case in selected_goals:
        for episode_index in range(goal_case.episodes):
            result = run_episode(
                backend,
                protocol,
                goal_case,
                episode_index,
                output_dir,
                smoke=args.smoke,
            )
            episode_results.append(result)
            print(
                f"{goal_case.name} seed={result['seed']}: "
                f"{result['termination_reason']}, "
                f"final_distance={result['final_distance_m']:.3f} m, "
                f"progress_efficiency={result['progress_efficiency']:.3f}"
            )
    summary_path = write_summary(
        protocol,
        policy_path,
        microduck_rl_root,
        output_dir,
        episode_results,
        smoke=args.smoke,
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
