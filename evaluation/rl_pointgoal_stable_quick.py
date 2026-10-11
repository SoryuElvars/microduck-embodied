#!/usr/bin/env python3
"""Live smoke/quick of old RL ONNX models with the new stop-aware arrival gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.pointgoal_benchmark import (
    DEFAULT_MICRODUCK_RL_ROOT,
    DEFAULT_POLICY,
    DEFAULT_PROTOCOL,
    _assert_reset_pose,
    _official_commit,
    _sha256,
    _step_row,
    _write_rows,
)
from evaluation.pointgoal_protocol import EpisodeSpec, load_pointgoal_protocol
from evaluation.pointgoal_stable_arrival_rescore import StableArrivalGate
from evaluation.rl_pointgoal_quick import OnnxNavigationPolicy
from navigation.classical_navigator import goal_error
from navigation.mujoco_backend import MujocoBackend
from navigation.rl_navigator import RLNavigator, RLNavigatorConfig
from navigation.types import GoalState, VelocityCommand


def run_episode(
    *,
    backend: MujocoBackend,
    protocol,
    spec: EpisodeSpec,
    k: int,
    navigator: RLNavigator,
    output_dir: Path,
    max_planar_speed_mps: float,
    max_abs_yaw_rate_radps: float,
    navigation_timeout_s: float | None = None,
) -> dict[str, object]:
    controller_id = f"rl_k{k}_stable"
    reset = backend.reset(spec.simulation_reset_seed, protocol.initial_state_mode)
    _assert_reset_pose(reset, spec)
    goal = GoalState(spec.goal_x, spec.goal_y)
    zero = VelocityCommand(0.0, 0.0, 0.0)
    rows = []
    for _ in range(round(protocol.warmup_s / backend.control_dt_s)):
        step = backend.step(zero)
        rows.append(_step_row(phase="warmup", controller_id=controller_id, spec=spec, goal=goal, command=zero, step=step))
        if step.state.fallen or not step.state.finite:
            break

    start = backend.observe()
    navigator.reset()
    gate = StableArrivalGate(
        radius_m=spec.success_radius,
        hold_s=spec.success_hold_time,
        dt_s=backend.control_dt_s * k,
        max_planar_speed_mps=max_planar_speed_mps,
        max_abs_yaw_rate_radps=max_abs_yaw_rate_radps,
    )
    timeout_s = spec.timeout if navigation_timeout_s is None else min(spec.timeout, navigation_timeout_s)
    limit_steps = max(1, round(timeout_s / backend.control_dt_s))
    post_steps_required = max(1, math.ceil(spec.post_arrival_observation_time / backend.control_dt_s))
    command = zero
    navigation_steps = 0
    post_steps = 0
    reached_stable_arrival = False
    completion_time_s = None
    redeparted_after_arrival = False
    reason = "invalid_state" if not start.finite else "fall" if start.fallen else "timeout"
    while start.finite and not start.fallen:
        if not reached_stable_arrival and navigation_steps >= limit_steps:
            break
        if reached_stable_arrival and post_steps >= post_steps_required:
            reason = "success"
            break
        step_index = navigation_steps + post_steps
        if step_index % k == 0:
            command = navigator.compute_command(backend.observe(), goal, backend.control_dt_s * k)
        phase = "post_arrival" if reached_stable_arrival else "navigate"
        step = backend.step(command)
        rows.append(_step_row(phase=phase, controller_id=controller_id, spec=spec, goal=goal, command=command, step=step))
        if not step.state.finite or step.state.fallen:
            reason = "invalid_state" if not step.state.finite else "fall"
            break
        distance, _ = goal_error(step.state, goal)
        if reached_stable_arrival:
            post_steps += 1
            if distance > spec.success_radius:
                redeparted_after_arrival = True
        else:
            navigation_steps += 1
            if navigation_steps % k == 0 and gate.update(
                distance,
                step.state.vx_body_mps,
                step.state.vy_body_mps,
                step.state.wz_body_radps,
            ):
                reached_stable_arrival = True
                completion_time_s = navigation_steps * backend.control_dt_s

    raw_path = output_dir / "raw" / f"{spec.episode_id}_{controller_id}_seed{spec.simulation_reset_seed}_steps.csv"
    _write_rows(raw_path, rows)
    final = backend.observe()
    final_distance, _ = goal_error(final, goal)
    return {
        "episode_id": spec.episode_id,
        "simulation_reset_seed": spec.simulation_reset_seed,
        "termination_reason": reason,
        "success": reason == "success",
        "reached_stable_arrival": reached_stable_arrival,
        "completion_time_s": completion_time_s,
        "navigation_duration_s": navigation_steps * backend.control_dt_s,
        "post_arrival_duration_s": post_steps * backend.control_dt_s,
        "final_distance_m": final_distance,
        "fall": reason == "fall",
        "timeout": reason == "timeout",
        "invalid_state": reason == "invalid_state",
        "redeparted_after_arrival": redeparted_after_arrival,
        "raw_steps_csv": str(raw_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-policy", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=(2, 5, 10), required=True)
    parser.add_argument("--low-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--max-planar-speed-mps", type=float, default=0.1)
    parser.add_argument("--max-abs-yaw-rate-radps", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.max_planar_speed_mps <= 0 or args.max_abs_yaw_rate_radps <= 0:
        parser.error("speed thresholds must be positive")

    high_path = args.high_policy.expanduser().resolve()
    low_path = args.low_policy.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    official_root = args.microduck_rl_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol = load_pointgoal_protocol(protocol_path)
    config = RLNavigatorConfig()
    config.validate()
    policy = OnnxNavigationPolicy(high_path, args.k)
    backend = MujocoBackend(official_root, low_path, protocol_path, output_dir)
    backend.validate()
    official_commit = _official_commit(official_root)
    file_hashes = {
        "high_policy": _sha256(high_path),
        "low_policy": _sha256(low_path),
        "protocol": _sha256(protocol_path),
        "live_evaluator": _sha256(Path(__file__)),
        "rescore_gate": _sha256(Path(__file__).with_name("pointgoal_stable_arrival_rescore.py")),
        "rl_navigator": _sha256(Path(__file__).parents[1] / "navigation" / "rl_navigator.py"),
    }
    signature = hashlib.sha256(json.dumps({"files": file_hashes, "k": args.k, "official_commit": official_commit,
        "thresholds": [args.max_planar_speed_mps, args.max_abs_yaw_rate_radps]}, sort_keys=True).encode()).hexdigest()
    print(f"validated K={args.k}, manifest={protocol.episode_manifest_sha256}, signature={signature}")
    if args.validate_only:
        return 0

    mode_name = "smoke" if args.smoke else "quick"
    specs = (
        (protocol.episode_map[protocol.smoke_episode_ids["front"]],)
        if args.smoke else tuple(protocol.episode_map[episode_id] for episode_id in protocol.quick_episode_ids)
    )
    navigator = RLNavigator(policy, config)
    episodes = []
    for spec in specs:
        progress_path = output_dir / mode_name / "progress" / f"{spec.episode_id}.json"
        if progress_path.exists():
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") != signature or saved.get("episode_id") != spec.episode_id:
                raise ValueError(f"stale or mismatched progress: {progress_path}")
            result = saved["result"]
        else:
            result = run_episode(
                backend=backend,
                protocol=protocol,
                spec=spec,
                k=args.k,
                navigator=navigator,
                output_dir=output_dir / mode_name,
                max_planar_speed_mps=args.max_planar_speed_mps,
                max_abs_yaw_rate_radps=args.max_abs_yaw_rate_radps,
                navigation_timeout_s=protocol.smoke_timeout_s if args.smoke else None,
            )
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({"signature": signature, "episode_id": spec.episode_id, "result": result}, indent=2) + "\n", encoding="utf-8")
        episodes.append(result)
        print(f"{spec.episode_id}: {result['termination_reason']}, final distance={result['final_distance_m']:.3f} m", flush=True)
    summary = {
        "kind": "week05_rl_stable_arrival_live_quick",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode_name,
        "k": args.k,
        "signature": signature,
        "files_sha256": file_hashes,
        "official_commit": official_commit,
        "episode_manifest_sha256": protocol.episode_manifest_sha256,
        "thresholds": {"radius_m": specs[0].success_radius, "hold_s": specs[0].success_hold_time,
            "max_planar_speed_mps": args.max_planar_speed_mps,
            "max_abs_yaw_rate_radps": args.max_abs_yaw_rate_radps},
        "aggregate": {"episodes": len(episodes), "successes": sum(item["success"] for item in episodes),
            "falls": sum(item["fall"] for item in episodes), "timeouts": sum(item["timeout"] for item in episodes),
            "invalid_states": sum(item["invalid_state"] for item in episodes),
            "median_final_distance_m": median(item["final_distance_m"] for item in episodes)},
        "episodes": episodes,
    }
    summary_path = output_dir / mode_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
