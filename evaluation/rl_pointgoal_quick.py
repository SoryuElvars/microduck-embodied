#!/usr/bin/env python3
"""Evaluate a Week-05 RL navigator on the frozen Week-04 EpisodeSpecs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import numpy as np
import onnxruntime as ort

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.pointgoal_benchmark import (
    DEFAULT_MICRODUCK_RL_ROOT,
    DEFAULT_POLICY,
    DEFAULT_PROTOCOL,
    _official_commit,
    _sha256,
    aggregate_controller,
    load_progress,
    run_episode,
    select_episode_specs,
    write_progress,
)
from evaluation.pointgoal_protocol import load_pointgoal_protocol
import evaluation.pointgoal_benchmark as frozen_benchmark
from navigation.mujoco_backend import MujocoBackend
from navigation.rl_navigator import RLNavigator, RLNavigatorConfig


PROJECT_ROOT = Path(__file__).parents[1]
EXPECTED_K = (2, 5, 10)


class OnnxNavigationPolicy:
    def __init__(self, path: Path, k: int) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("navigation ONNX must have one input and one output")
        if inputs[0].shape != [1, 7] or outputs[0].shape != [1, 2]:
            raise ValueError("navigation ONNX must have [1,7] -> [1,2] shape")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("policy_kind") != "pointgoal_navigation":
            raise ValueError("ONNX metadata is not a PointGoal navigation policy")
        if f"k{k}-" not in metadata.get("run_path", ""):
            raise ValueError(f"navigation ONNX run_path does not match K={k}")
        self.input_name = inputs[0].name

    def __call__(self, observation: Sequence[float]) -> tuple[float, float]:
        value = np.asarray(observation, dtype=np.float32).reshape(1, 7)
        output = self.session.run(None, {self.input_name: value})[0]
        return float(output[0, 0]), float(output[0, 1])


def signature(*, high_policy: Path, low_policy: Path, protocol: Path, k: int, mode: str, commit: str) -> str:
    paths = [high_policy, low_policy, protocol, Path(__file__), PROJECT_ROOT / "evaluation" / "pointgoal_benchmark.py", PROJECT_ROOT / "evaluation" / "pointgoal_protocol.py", PROJECT_ROOT / "evaluation" / "locomotion_benchmark.py", PROJECT_ROOT / "navigation" / "mujoco_backend.py", PROJECT_ROOT / "navigation" / "rl_navigator.py", PROJECT_ROOT / "navigation" / "types.py"]
    payload = {"files": {str(path): _sha256(path) for path in paths}, "k": k, "mode": mode, "official_commit": commit}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-policy", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=EXPECTED_K, required=True)
    parser.add_argument("--low-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--episode-id",
        help="Frozen EpisodeSpec to show; defaults to the front smoke scenario.",
    )
    args = parser.parse_args(argv)

    high_policy = args.high_policy.expanduser().resolve()
    low_policy = args.low_policy.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    official_root = args.microduck_rl_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol = load_pointgoal_protocol(protocol_path)
    config = RLNavigatorConfig()
    config.validate()
    if (config.max_forward_speed_mps, config.max_yaw_rate_radps) != (
        protocol.constrained.max_forward_speed_mps,
        protocol.constrained.max_yaw_rate_radps,
    ):
        raise ValueError("RL command limits differ from frozen Week-04 limits")
    if (config.max_forward_accel_mps2, config.max_yaw_accel_radps2) != (
        protocol.constrained.max_forward_accel_mps2,
        protocol.constrained.max_yaw_accel_radps2,
    ):
        raise ValueError("RL acceleration limits differ from frozen Week-04 limits")
    policy = OnnxNavigationPolicy(high_policy, args.k)
    backend = MujocoBackend(official_root, low_policy, protocol_path, output_dir)
    backend.validate()
    commit = _official_commit(official_root)
    print(f"validated K={args.k}, high={_sha256(high_policy)}, low={_sha256(low_policy)}, manifest={protocol.episode_manifest_sha256}")
    if args.validate_only:
        return 0

    runtime_protocol = replace(protocol, navigator_frequency_hz=50.0 / args.k)
    controller_id = f"rl_k{args.k}"
    if args.viewer:
        episode_id = args.episode_id or protocol.smoke_episode_ids["front"]
        if episode_id not in protocol.episode_map:
            parser.error(f"unknown EpisodeSpec id: {episode_id!r}")
        spec = protocol.episode_map[episode_id]
        viewer_dir = output_dir / "viewer" / controller_id / episode_id
        backend.reset(spec.simulation_reset_seed, protocol.initial_state_mode)
        try:
            with patch.object(frozen_benchmark, "build_navigator", return_value=RLNavigator(policy, config)):
                with backend.open_viewer() as viewer:
                    result = run_episode(
                        backend=backend,
                        protocol=runtime_protocol,
                        spec=spec,
                        controller_id=controller_id,
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

    if args.smoke:
        mode_name = "smoke"
        specs = (protocol.episode_map[protocol.smoke_episode_ids["front"]],)
    else:
        mode_name = "full" if args.full else "quick"
        specs = select_episode_specs(protocol, mode_name)
    run_signature = signature(high_policy=high_policy, low_policy=low_policy, protocol=protocol_path, k=args.k, mode=mode_name, commit=commit)
    episodes = []
    for spec in specs:
        progress_path = output_dir / mode_name / "progress" / f"{spec.episode_id}.json"
        result = load_progress(progress_path, signature=run_signature, controller_id=controller_id, episode_id=spec.episode_id)
        if result is None:
            # The Week-04 evaluator stays unchanged. This adapter supplies the
            # RL controller and its K-step cadence to the same metric loop.
            with patch.object(frozen_benchmark, "build_navigator", return_value=RLNavigator(policy, config)):
                result = run_episode(
                    backend=backend,
                    protocol=runtime_protocol,
                    spec=spec,
                    controller_id=controller_id,
                    output_dir=output_dir / mode_name,
                    navigation_timeout_s=protocol.smoke_timeout_s if args.smoke else None,
                )
            write_progress(progress_path, signature=run_signature, controller_id=controller_id, spec=spec, episode=result)
        episodes.append(result)
        print(f"{controller_id}/{spec.episode_id}: {result['termination_reason']}, distance={result['final_distance_m']:.3f} m")
    aggregate = aggregate_controller(episodes)
    quick_gate = None
    if mode_name == "quick":
        gate_config = json.loads(protocol_path.read_text(encoding="utf-8"))["quick_gate"]
        gate = {key: float(gate_config[key]) for key in ("maximum_fall_rate", "maximum_invalid_state_rate")}
        quick_gate = {
            key: {
                "threshold": threshold,
                "observed": aggregate["fall_rate" if "fall" in key else "invalid_state_rate"],
                "passed": aggregate["fall_rate" if "fall" in key else "invalid_state_rate"] <= threshold,
            }
            for key, threshold in gate.items()
        }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "week05_rl_pointgoal_benchmark_v1",
        "run_mode": mode_name,
        "controller_id": controller_id,
        "k": args.k,
        "navigator_frequency_hz": 50.0 / args.k,
        "run_signature": run_signature,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": _sha256(protocol_path),
        "episode_manifest_sha256": protocol.episode_manifest_sha256,
        "official_commit": commit,
        "high_policy_path": str(high_policy),
        "high_policy_sha256": _sha256(high_policy),
        "low_policy_path": str(low_policy),
        "low_policy_sha256": _sha256(low_policy),
        "reward_available": False,
        "evaluation_distribution_note": protocol.generation.get(
            "distribution_note",
            "Week-04 manifest: 1.0-1.8 m, all headings; training Stage 1 uses mostly 0.4-0.8 m/front, Stage 2 expands to 1.3 m/front half-plane, and Stage 3 uses all headings to 1.8 m.",
        ),
        "aggregate": aggregate,
        "episodes": episodes,
    }
    if quick_gate is not None:
        summary["quick_gate"] = quick_gate
    summary_path = output_dir / mode_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
