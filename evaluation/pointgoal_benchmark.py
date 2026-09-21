#!/usr/bin/env python3
"""Paired Naive-P versus Constrained Classical PointGoal benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
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
from evaluation.pointgoal_protocol import (
    CONTROLLER_IDS,
    ClassicalPointGoalProtocol,
    EpisodeSpec,
    load_pointgoal_protocol,
)
from navigation.classical_navigator import (
    ConstrainedGoToGoalNavigator,
    NaivePNavigator,
    goal_error,
    wrap_angle,
)
from navigation.mujoco_backend import BackendStep, MujocoBackend, ViewerSession
from navigation.types import GoalState, Navigator, RobotState, VelocityCommand


DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "artifacts"
    / "week03"
    / "06_yaw_only_tracking_candidate"
    / "models"
    / "model_1500.onnx"
)
DEFAULT_PROTOCOL = (
    Path(__file__).parent
    / "pointgoal_protocols"
    / "week04_classical_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "week04"
    / "01_classical_pointgoal_benchmark"
    / "model_1500"
)


def build_navigator(
    protocol: ClassicalPointGoalProtocol,
    controller_id: str,
) -> Navigator:
    """Construct one controller through the shared Navigator interface."""

    if controller_id == "naive_p":
        return NaivePNavigator(protocol.naive_p)
    if controller_id == "constrained":
        return ConstrainedGoToGoalNavigator(protocol.constrained)
    raise ValueError(f"unknown controller: {controller_id!r}")


def select_episode_specs(
    protocol: ClassicalPointGoalProtocol,
    mode: str,
) -> tuple[EpisodeSpec, ...]:
    """Select a frozen subset without generating scenes at run time."""

    mapping = protocol.episode_map
    if mode == "smoke":
        return tuple(mapping[value] for value in protocol.smoke_episode_ids.values())
    if mode == "quick":
        return tuple(mapping[value] for value in protocol.quick_episode_ids)
    if mode == "full":
        if protocol.protocol_status != "frozen":
            raise ValueError(
                "full benchmark requires protocol_status='frozen'; review smoke and "
                "quick results before freezing controller parameters"
            )
        return protocol.episodes
    raise ValueError(f"unsupported run mode: {mode!r}")


def _controller_record(
    protocol: ClassicalPointGoalProtocol,
    controller_id: str,
) -> dict[str, Any]:
    config = protocol.naive_p if controller_id == "naive_p" else protocol.constrained
    return {"controller_id": controller_id, "config": asdict(config)}


def _step_row(
    *,
    phase: str,
    controller_id: str,
    spec: EpisodeSpec,
    goal: GoalState,
    command: VelocityCommand,
    step: BackendStep,
) -> dict[str, Any]:
    distance, heading_error = goal_error(step.state, goal)
    row: dict[str, Any] = {
        "phase": phase,
        "controller_id": controller_id,
        "episode_id": spec.episode_id,
        "simulation_reset_seed": spec.simulation_reset_seed,
        "time_s": step.state.time_s,
        "goal_world_x_m": goal.x_world_m,
        "goal_world_y_m": goal.y_world_m,
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
    row.update(
        {f"action_{index:02d}": value for index, value in enumerate(step.action)}
    )
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty trajectory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _step_backend(
    backend: MujocoBackend,
    command: VelocityCommand,
    viewer: ViewerSession | None,
    *,
    realtime: bool,
) -> BackendStep:
    started_at = time.perf_counter()
    step = backend.step(command)
    if viewer is not None:
        viewer.sync()
    if realtime:
        remaining_s = backend.control_dt_s - (time.perf_counter() - started_at)
        if remaining_s > 0:
            time.sleep(remaining_s)
    return step


def _assert_reset_pose(actual: RobotState, spec: EpisodeSpec) -> None:
    tolerance = 2e-8
    mismatches = {
        "x": actual.x_world_m - spec.start_x,
        "y": actual.y_world_m - spec.start_y,
        "yaw": wrap_angle(actual.yaw_rad - spec.start_yaw),
    }
    if any(abs(value) > tolerance for value in mismatches.values()):
        raise ValueError(
            f"backend reset does not match EpisodeSpec {spec.episode_id}: {mismatches}"
        )


def run_episode(
    *,
    backend: MujocoBackend,
    protocol: ClassicalPointGoalProtocol,
    spec: EpisodeSpec,
    controller_id: str,
    output_dir: Path,
    navigation_timeout_s: float | None = None,
    viewer: ViewerSession | None = None,
    realtime: bool = False,
    reset_backend: bool = True,
) -> dict[str, Any]:
    """Run one frozen scenario, including the post-arrival stability phase."""

    reset_state = (
        backend.reset(spec.simulation_reset_seed, protocol.initial_state_mode)
        if reset_backend
        else backend.observe()
    )
    _assert_reset_pose(reset_state, spec)
    goal = GoalState(spec.goal_x, spec.goal_y)
    if viewer is not None and viewer.is_running():
        viewer.set_goal(goal, spec.success_radius)
    navigator = build_navigator(protocol, controller_id)
    zero = VelocityCommand(0.0, 0.0, 0.0)
    rows: list[dict[str, Any]] = []
    viewer_closed = False
    warmup_steps = round(protocol.warmup_s / backend.control_dt_s)
    for _ in range(warmup_steps):
        if viewer is not None and not viewer.is_running():
            viewer_closed = True
            break
        step = _step_backend(backend, zero, viewer, realtime=realtime)
        rows.append(
            _step_row(
                phase="warmup",
                controller_id=controller_id,
                spec=spec,
                goal=goal,
                command=zero,
                step=step,
            )
        )
        if step.state.fallen:
            break

    start = backend.observe()
    initial_distance, _ = goal_error(start, goal)
    previous_x = start.x_world_m
    previous_y = start.y_world_m
    path_length = 0.0
    path_length_at_arrival: float | None = None
    minimum_distance = initial_distance
    navigator_interval_steps = max(
        1,
        round(1.0 / (protocol.navigator_frequency_hz * backend.control_dt_s)),
    )
    navigator_dt_s = navigator_interval_steps * backend.control_dt_s
    timeout_s = spec.timeout if navigation_timeout_s is None else min(
        spec.timeout, navigation_timeout_s
    )
    navigation_limit_steps = max(1, round(timeout_s / backend.control_dt_s))
    hold_steps = max(1, math.ceil(spec.success_hold_time / backend.control_dt_s))
    post_steps_required = max(
        1,
        math.ceil(spec.post_arrival_observation_time / backend.control_dt_s),
    )
    inside_steps = 0
    navigation_steps = 0
    post_steps = 0
    controller_steps = 0
    reached_success_hold = False
    completion_time_s: float | None = None
    arrival_state: RobotState | None = None
    post_states: list[RobotState] = []
    redeparted_after_arrival = False
    command = zero
    termination_reason = (
        "invalid_state"
        if not start.finite
        else "fall"
        if start.fallen
        else "viewer_closed"
        if viewer_closed
        else "timeout"
    )
    test_rows: list[dict[str, Any]] = []

    navigator.reset()
    while not start.fallen and not viewer_closed:
        if not reached_success_hold and navigation_steps >= navigation_limit_steps:
            termination_reason = "timeout"
            break
        if reached_success_hold and post_steps >= post_steps_required:
            termination_reason = "success"
            break
        if viewer is not None and not viewer.is_running():
            termination_reason = "viewer_closed"
            break
        if controller_steps % navigator_interval_steps == 0:
            command = navigator.compute_command(backend.observe(), goal, navigator_dt_s)
        phase = "post_arrival" if reached_success_hold else "navigate"
        step = _step_backend(backend, command, viewer, realtime=realtime)
        test_rows.append(
            _step_row(
                phase=phase,
                controller_id=controller_id,
                spec=spec,
                goal=goal,
                command=command,
                step=step,
            )
        )
        controller_steps += 1
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
        if reached_success_hold:
            post_steps += 1
            post_states.append(step.state)
            if distance > spec.success_radius:
                redeparted_after_arrival = True
        else:
            navigation_steps += 1
            inside_steps = inside_steps + 1 if distance <= spec.success_radius else 0
            if inside_steps >= hold_steps:
                reached_success_hold = True
                completion_time_s = navigation_steps * backend.control_dt_s
                arrival_state = step.state
                path_length_at_arrival = path_length

    rows.extend(test_rows)
    raw_path = (
        output_dir
        / "raw"
        / f"{spec.episode_id}_{controller_id}_seed{spec.simulation_reset_seed}_steps.csv"
    )
    _write_rows(raw_path, rows)
    final = backend.observe()
    final_distance, final_heading_error = goal_error(final, goal)
    navigation_rows = [row for row in test_rows if row["phase"] == "navigate"]
    post_rows = [row for row in test_rows if row["phase"] == "post_arrival"]
    max_vx = protocol.naive_p.max_forward_speed_mps
    max_wz = protocol.naive_p.max_yaw_rate_radps
    denominator = max(1, len(test_rows))
    vx_saturated = sum(
        abs(float(row["cmd_vx"])) >= max_vx - 1e-9 for row in test_rows
    )
    wz_saturated = sum(
        abs(float(row["cmd_wz"])) >= max_wz - 1e-9 for row in test_rows
    )
    successful_path_length = (
        path_length_at_arrival if path_length_at_arrival is not None else path_length
    )
    path_efficiency = (
        initial_distance / max(successful_path_length, initial_distance)
        if reached_success_hold
        else None
    )
    progress_efficiency = (
        max(0.0, initial_distance - minimum_distance) / successful_path_length
        if successful_path_length > 0
        else 0.0
    )
    post_stop_drift = (
        math.hypot(
            final.x_world_m - arrival_state.x_world_m,
            final.y_world_m - arrival_state.y_world_m,
        )
        if arrival_state is not None
        else None
    )
    post_max_drift = (
        max(
            math.hypot(
                state.x_world_m - arrival_state.x_world_m,
                state.y_world_m - arrival_state.y_world_m,
            )
            for state in post_states
        )
        if arrival_state is not None and post_states
        else None
    )
    post_max_goal_distance = (
        max(goal_error(state, goal)[0] for state in post_states)
        if post_states
        else None
    )

    def rmse(rows_for_metric: list[dict[str, Any]], actual: str, target: str) -> float | None:
        if not rows_for_metric:
            return None
        return math.sqrt(
            fmean(
                (float(row[actual]) - float(row[target])) ** 2
                for row in rows_for_metric
            )
        )

    return {
        "controller_id": controller_id,
        "episode_id": spec.episode_id,
        "simulation_reset_seed": spec.simulation_reset_seed,
        "success": termination_reason == "success",
        "reached_success_hold": reached_success_hold,
        "termination_reason": termination_reason,
        "completion_time_s": completion_time_s,
        "navigation_duration_s": navigation_steps * backend.control_dt_s,
        "post_arrival_duration_s": post_steps * backend.control_dt_s,
        "initial_goal_distance_m": initial_distance,
        "final_distance_m": final_distance,
        "minimum_distance_m": minimum_distance,
        "final_heading_error_rad": final_heading_error,
        "path_length_to_arrival_m": path_length_at_arrival,
        "total_path_length_m": path_length,
        "path_efficiency": path_efficiency,
        "progress_efficiency": progress_efficiency,
        "tracking_rmse_vx_mps": rmse(navigation_rows, "actual_vx", "cmd_vx"),
        "tracking_rmse_wz_radps": rmse(navigation_rows, "actual_wz", "cmd_wz"),
        "vx_saturation_fraction": vx_saturated / denominator,
        "wz_saturation_fraction": wz_saturated / denominator,
        "post_arrival_stop_drift_m": post_stop_drift,
        "post_arrival_max_drift_m": post_max_drift,
        "post_arrival_max_goal_distance_m": post_max_goal_distance,
        "post_arrival_mean_planar_speed_mps": (
            fmean(
                math.hypot(float(row["actual_vx"]), float(row["actual_vy"]))
                for row in post_rows
            )
            if post_rows
            else None
        ),
        "redeparted_after_arrival": redeparted_after_arrival,
        "fall": termination_reason == "fall",
        "timeout": termination_reason == "timeout",
        "invalid_state": termination_reason == "invalid_state",
        "viewer_closed": termination_reason == "viewer_closed",
        "reset_state": asdict(reset_state),
        "navigation_start_state": asdict(start),
        "goal_world": asdict(goal),
        "arrival_state": None if arrival_state is None else asdict(arrival_state),
        "final_state": asdict(final),
        "raw_steps_csv": path_for_record(raw_path, PROJECT_ROOT),
    }


def _optional_median(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return median(values) if values else None


def aggregate_controller(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one controller without pooling it with its paired baseline."""

    arrivals = [item for item in episodes if item["reached_success_hold"]]
    return {
        "episode_count": len(episodes),
        "success_rate": fmean(float(item["success"]) for item in episodes),
        "reached_success_hold_rate": fmean(
            float(item["reached_success_hold"]) for item in episodes
        ),
        "fall_rate": fmean(float(item["fall"]) for item in episodes),
        "timeout_rate": fmean(float(item["timeout"]) for item in episodes),
        "invalid_state_rate": fmean(
            float(item["invalid_state"]) for item in episodes
        ),
        "median_final_distance_m": median(
            float(item["final_distance_m"]) for item in episodes
        ),
        "median_path_length_to_arrival_m": _optional_median(
            episodes, "path_length_to_arrival_m"
        ),
        "median_path_efficiency_arrivals": _optional_median(
            episodes, "path_efficiency"
        ),
        "median_completion_time_s_arrivals": _optional_median(
            episodes, "completion_time_s"
        ),
        "mean_vx_saturation_fraction": fmean(
            float(item["vx_saturation_fraction"]) for item in episodes
        ),
        "mean_wz_saturation_fraction": fmean(
            float(item["wz_saturation_fraction"]) for item in episodes
        ),
        "median_post_arrival_stop_drift_m": _optional_median(
            episodes, "post_arrival_stop_drift_m"
        ),
        "median_post_arrival_max_drift_m": _optional_median(
            episodes, "post_arrival_max_drift_m"
        ),
        "post_arrival_redeparture_rate": (
            fmean(float(item["redeparted_after_arrival"]) for item in arrivals)
            if arrivals
            else None
        ),
    }


def paired_comparison(
    controller_episodes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return per-scenario constrained-minus-naive paired differences."""

    naive = {item["episode_id"]: item for item in controller_episodes["naive_p"]}
    constrained = {
        item["episode_id"]: item for item in controller_episodes["constrained"]
    }
    if set(naive) != set(constrained):
        raise ValueError("paired comparison requires identical EpisodeSpec ids")
    pairs: list[dict[str, Any]] = []
    metric_keys = (
        "final_distance_m",
        "path_length_to_arrival_m",
        "path_efficiency",
        "completion_time_s",
        "vx_saturation_fraction",
        "wz_saturation_fraction",
        "post_arrival_stop_drift_m",
    )
    for episode_id in naive:
        left = naive[episode_id]
        right = constrained[episode_id]
        pair: dict[str, Any] = {
            "episode_id": episode_id,
            "success_delta_constrained_minus_naive": int(right["success"])
            - int(left["success"]),
            "redeparture_delta_constrained_minus_naive": int(
                right["redeparted_after_arrival"]
            )
            - int(left["redeparted_after_arrival"]),
        }
        for key in metric_keys:
            pair[f"{key}_delta_constrained_minus_naive"] = (
                None
                if left.get(key) is None or right.get(key) is None
                else float(right[key]) - float(left[key])
            )
        pairs.append(pair)

    success_deltas = [item["success_delta_constrained_minus_naive"] for item in pairs]
    aggregate: dict[str, Any] = {
        "pair_count": len(pairs),
        "constrained_success_wins": sum(value > 0 for value in success_deltas),
        "naive_success_wins": sum(value < 0 for value in success_deltas),
        "success_ties": sum(value == 0 for value in success_deltas),
    }
    for key in metric_keys:
        delta_key = f"{key}_delta_constrained_minus_naive"
        values = [float(item[delta_key]) for item in pairs if item[delta_key] is not None]
        aggregate[f"median_{delta_key}"] = median(values) if values else None
    return {"aggregate": aggregate, "episodes": pairs}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_signature(
    *,
    policy_path: Path,
    protocol: ClassicalPointGoalProtocol,
    microduck_rl_commit: str,
    mode: str,
) -> str:
    evaluator_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "evaluation" / "pointgoal_protocol.py",
        PROJECT_ROOT / "evaluation" / "locomotion_benchmark.py",
        PROJECT_ROOT / "navigation" / "mujoco_backend.py",
        PROJECT_ROOT / "navigation" / "classical_navigator.py",
        PROJECT_ROOT / "navigation" / "types.py",
    )
    payload = {
        "policy_sha256": _sha256(policy_path),
        "protocol_sha256": _sha256(protocol.source_path),
        "episode_manifest_sha256": protocol.episode_manifest_sha256,
        "microduck_rl_commit": microduck_rl_commit,
        "mode": mode,
        "evaluator_sha256": {
            path_for_record(path, PROJECT_ROOT): _sha256(path)
            for path in evaluator_paths
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _progress_path(
    output_dir: Path,
    mode: str,
    controller_id: str,
    episode_id: str,
) -> Path:
    return output_dir / mode / controller_id / "progress" / f"{episode_id}.json"


def load_progress(
    path: Path,
    *,
    signature: str,
    controller_id: str,
    episode_id: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("run_signature") != signature:
        raise ValueError(
            f"stale progress signature at {path}; use a new output directory"
        )
    if document.get("controller_id") != controller_id:
        raise ValueError(f"progress controller mismatch at {path}")
    if document.get("episode_id") != episode_id:
        raise ValueError(f"progress EpisodeSpec mismatch at {path}")
    return dict(document["episode"])


def write_progress(
    path: Path,
    *,
    signature: str,
    controller_id: str,
    spec: EpisodeSpec,
    episode: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "run_signature": signature,
        "controller_id": controller_id,
        "episode_id": spec.episode_id,
        "episode_spec": asdict(spec),
        "episode": episode,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluate_quick_gate(
    protocol_path: Path,
    aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document = json.loads(protocol_path.read_text(encoding="utf-8"))
    gate = document.get("quick_gate", {})
    criteria: dict[str, Any] = {}
    mapping = {
        "maximum_fall_rate": "fall_rate",
        "maximum_invalid_state_rate": "invalid_state_rate",
    }
    for controller_id, aggregate in aggregates.items():
        criteria[controller_id] = {}
        for criterion, metric in mapping.items():
            threshold = float(gate[criterion])
            observed = float(aggregate[metric])
            criteria[controller_id][criterion] = {
                "observed": observed,
                "threshold": threshold,
                "passed": observed <= threshold,
            }
    return {
        "passed": all(
            item["passed"]
            for controller in criteria.values()
            for item in controller.values()
        ),
        "criteria": criteria,
        "note": gate.get("note"),
    }


def write_summary(
    *,
    protocol: ClassicalPointGoalProtocol,
    policy_path: Path,
    microduck_rl_root: Path,
    output_dir: Path,
    mode: str,
    signature: str,
    controller_episodes: dict[str, list[dict[str, Any]]],
) -> Path:
    aggregates = {
        controller_id: aggregate_controller(episodes)
        for controller_id, episodes in controller_episodes.items()
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "classical_pointgoal_paired_controller_v1",
        "run_mode": mode,
        "protocol_id": protocol.protocol_id,
        "protocol_status": protocol.protocol_status,
        "episode_manifest_status": protocol.episode_manifest_status,
        "protocol_path": path_for_record(protocol.source_path, PROJECT_ROOT),
        "protocol_sha256": _sha256(protocol.source_path),
        "episode_manifest_sha256": protocol.episode_manifest_sha256,
        "run_signature": signature,
        "policy_path": path_for_record(policy_path, microduck_rl_root, PROJECT_ROOT),
        "policy_sha256": _sha256(policy_path),
        "microduck_rl_commit": _official_commit(microduck_rl_root),
        "pose_source": "simulator_ground_truth_via_MujocoBackend",
        "reward_available": False,
        "task_definition": {
            "goal_type": "position_pointgoal",
            "requires_final_yaw": False,
            "navigator_action_scope": "[vx, 0, wz]",
        },
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
        },
        "controllers": {
            controller_id: {
                **_controller_record(protocol, controller_id),
                "aggregate": aggregates[controller_id],
                "episodes": controller_episodes[controller_id],
            }
            for controller_id in protocol.controller_order
        },
        "paired_comparison": paired_comparison(controller_episodes),
    }
    if mode == "quick":
        document["quick_gate"] = _evaluate_quick_gate(
            protocol.source_path, aggregates
        )
    summary_dir = output_dir / mode / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"pointgoal_{mode}_{policy_path.stem}.json"
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or run the paired Naive-P/Constrained Classical PointGoal benchmark."
        )
    )
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--viewer", action="store_true")
    parser.add_argument("--controller", choices=CONTROLLER_IDS, default="constrained")
    parser.add_argument(
        "--episode-id",
        help="Viewer EpisodeSpec id; defaults to the frozen front smoke scenario.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    microduck_rl_root = _resolved(args.microduck_rl_root)
    policy_path = _resolved(args.policy)
    protocol_path = _resolved(args.protocol)
    output_dir = _resolved(args.output_dir)
    protocol = load_pointgoal_protocol(protocol_path)
    official_commit = _official_commit(microduck_rl_root)
    validation_backend = MujocoBackend(
        microduck_rl_root=microduck_rl_root,
        policy_path=policy_path,
        metadata_path=protocol_path,
        output_dir=output_dir,
    )
    validation_backend.validate()
    print("Week 4 Classical PointGoal benchmark contract is valid")
    print(f"  protocol:          {protocol.protocol_id} ({protocol.protocol_status})")
    print(f"  frozen manifest:   {protocol.episode_manifest_sha256}")
    print(f"  scenarios:         {len(protocol.episodes)} shared EpisodeSpecs")
    print(f"  controllers:       {', '.join(protocol.controller_order)}")
    print(f"  policy SHA-256:    {_sha256(policy_path)}")
    print(f"  microduck_rl HEAD: {official_commit}")
    print(f"  expected I/O:      {EXPECTED_OBSERVATION_SIZE} obs -> {EXPECTED_ACTION_SIZE} actions")
    if args.validate_only:
        return 0

    if args.viewer:
        episode_id = args.episode_id or protocol.smoke_episode_ids["front"]
        if episode_id not in protocol.episode_map:
            parser.error(f"unknown EpisodeSpec id: {episode_id!r}")
        spec = protocol.episode_map[episode_id]
        viewer_dir = output_dir / "viewer" / args.controller / episode_id
        backend = MujocoBackend(
            microduck_rl_root=microduck_rl_root,
            policy_path=policy_path,
            metadata_path=protocol_path,
            output_dir=viewer_dir,
        )
        backend.reset(spec.simulation_reset_seed, protocol.initial_state_mode)
        try:
            with backend.open_viewer() as viewer:
                result = run_episode(
                    backend=backend,
                    protocol=protocol,
                    spec=spec,
                    controller_id=args.controller,
                    output_dir=viewer_dir,
                    viewer=viewer,
                    realtime=True,
                    reset_backend=False,
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

    mode = "smoke" if args.smoke else "quick" if args.quick else "full"
    try:
        specs = select_episode_specs(protocol, mode)
    except ValueError as error:
        parser.error(str(error))
    signature = run_signature(
        policy_path=policy_path,
        protocol=protocol,
        microduck_rl_commit=official_commit,
        mode=mode,
    )
    controller_episodes: dict[str, list[dict[str, Any]]] = {
        controller_id: [] for controller_id in protocol.controller_order
    }
    for controller_id in protocol.controller_order:
        controller_dir = output_dir / mode / controller_id
        backend = MujocoBackend(
            microduck_rl_root=microduck_rl_root,
            policy_path=policy_path,
            metadata_path=protocol_path,
            output_dir=controller_dir,
        )
        backend.validate()
        for spec in specs:
            progress_path = _progress_path(
                output_dir, mode, controller_id, spec.episode_id
            )
            result = (
                None
                if args.no_resume
                else load_progress(
                    progress_path,
                    signature=signature,
                    controller_id=controller_id,
                    episode_id=spec.episode_id,
                )
            )
            if result is None:
                result = run_episode(
                    backend=backend,
                    protocol=protocol,
                    spec=spec,
                    controller_id=controller_id,
                    output_dir=controller_dir,
                    navigation_timeout_s=(
                        protocol.smoke_timeout_s if mode == "smoke" else None
                    ),
                )
                write_progress(
                    progress_path,
                    signature=signature,
                    controller_id=controller_id,
                    spec=spec,
                    episode=result,
                )
            controller_episodes[controller_id].append(result)
            print(
                f"{controller_id}/{spec.episode_id} seed={spec.simulation_reset_seed}: "
                f"{result['termination_reason']}, "
                f"final_distance={result['final_distance_m']:.3f} m"
            )
    summary_path = write_summary(
        protocol=protocol,
        policy_path=policy_path,
        microduck_rl_root=microduck_rl_root,
        output_dir=output_dir,
        mode=mode,
        signature=signature,
        controller_episodes=controller_episodes,
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
