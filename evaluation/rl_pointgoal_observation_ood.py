#!/usr/bin/env python3
"""Evaluate the frozen Week-05 RL Navigator under one observation-side OOD factor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.pointgoal_benchmark import (
    DEFAULT_MICRODUCK_RL_ROOT,
    _official_commit,
    _sha256,
    aggregate_controller,
    run_episode,
)
from evaluation.pointgoal_protocol import EpisodeSpec, load_pointgoal_protocol
import evaluation.pointgoal_benchmark as frozen_benchmark
from evaluation.rl_pointgoal_quick import OnnxNavigationPolicy
from navigation.mujoco_backend import MujocoBackend
from navigation.observation_perturbation import (
    ObservationPerturbation,
    ObservationPerturbedNavigator,
)
from navigation.rl_navigator import RLNavigator, RLNavigatorConfig


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "evaluation" / "pointgoal_protocols" / "week05_rl_yaw_bias_ood_v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "week05" / "17_high_level_yaw_bias_ood_k10"
PAIRED_METRICS = {
    "final_distance_m": "lower",
    "path_length_to_arrival_m": "lower",
    "path_efficiency": "higher",
    "completion_time_s": "lower",
    "post_arrival_stop_drift_m": "lower",
    "post_arrival_max_drift_m": "lower",
}


@dataclass(frozen=True)
class ObservationOodProtocol:
    source_path: Path
    protocol_id: str
    protocol_status: str
    factor: str
    base_protocol_path: Path
    base_protocol: Any
    selected_model_path: Path
    selected_model: dict[str, Any]
    conditions: tuple[ObservationPerturbation, ...]
    smoke_episode_id: str
    quick_episode_ids: tuple[str, ...]
    quick_gate: dict[str, Any]
    full_gate: dict[str, Any]


def _resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _parse_condition(raw: Mapping[str, Any], factor: str) -> ObservationPerturbation:
    condition_factor = str(raw.get("factor", factor))
    return ObservationPerturbation(
        condition_id=str(raw["id"]),
        factor=condition_factor,
        yaw_bias_rad=float(raw.get("yaw_bias_deg", 0.0)) * 3.141592653589793 / 180.0,
        robot_x_bias_m=float(raw.get("robot_x_bias_m", 0.0)),
        robot_y_bias_m=float(raw.get("robot_y_bias_m", 0.0)),
        goal_x_bias_m=float(raw.get("goal_x_bias_m", 0.0)),
        goal_y_bias_m=float(raw.get("goal_y_bias_m", 0.0)),
        dropout_probability=float(raw.get("dropout_probability", 0.0)),
        dropout_seed=int(raw.get("dropout_seed", 0)),
    )


def load_protocol(path: Path) -> ObservationOodProtocol:
    source_path = path.expanduser().resolve()
    raw = _load_object(source_path)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported observation OOD schema_version")
    if raw.get("protocol_status") not in {"draft", "frozen"}:
        raise ValueError("OOD protocol_status must be 'draft' or 'frozen'")
    factor = str(raw["factor"])
    base_path = _resolve_project_path(raw["base_protocol_path"])
    if _sha256(base_path) != raw.get("base_protocol_sha256"):
        raise ValueError("base PointGoal protocol SHA-256 mismatch")
    base_protocol = load_pointgoal_protocol(base_path)
    if base_protocol.protocol_status != "frozen":
        raise ValueError("base PointGoal protocol must be frozen")
    selected_path = _resolve_project_path(raw["selected_model_path"])
    if _sha256(selected_path) != raw.get("selected_model_sha256"):
        raise ValueError("selected-model manifest SHA-256 mismatch")
    selected_model = _load_object(selected_path)
    if selected_model.get("selection_status") != "frozen":
        raise ValueError("selected model must be frozen")
    policy = selected_model.get("high_level_policy", {})
    if policy.get("k") != 10 or policy.get("navigator_frequency_hz") != 5.0:
        raise ValueError("OOD protocol expects the frozen K=10, 5 Hz Navigator")
    conditions = tuple(_parse_condition(item, factor) for item in raw.get("conditions", ()))
    if not conditions:
        raise ValueError("observation OOD protocol must declare at least one condition")
    if len({item.condition_id for item in conditions}) != len(conditions):
        raise ValueError("OOD condition ids must be unique")
    for condition in conditions:
        condition.validate()
        if condition.factor != factor:
            raise ValueError("all conditions in one protocol must share its declared factor")
    smoke_id = str(raw["smoke_episode_id"])
    quick_ids = tuple(str(item) for item in raw["quick_episode_ids"])
    known_ids = set(base_protocol.episode_map)
    if smoke_id not in known_ids or len(quick_ids) != 10 or not set(quick_ids) <= known_ids:
        raise ValueError("OOD selections must reference ten frozen Week-04 EpisodeSpecs")
    if len(set(quick_ids)) != len(quick_ids):
        raise ValueError("quick selection must not contain duplicate EpisodeSpecs")
    return ObservationOodProtocol(
        source_path=source_path,
        protocol_id=str(raw["protocol_id"]),
        protocol_status=str(raw["protocol_status"]),
        factor=factor,
        base_protocol_path=base_path,
        base_protocol=base_protocol,
        selected_model_path=selected_path,
        selected_model=selected_model,
        conditions=conditions,
        smoke_episode_id=smoke_id,
        quick_episode_ids=quick_ids,
        quick_gate=dict(raw["quick_gate"]),
        full_gate=dict(raw["full_gate"]),
    )


def select_specs(protocol: ObservationOodProtocol, mode: str) -> tuple[EpisodeSpec, ...]:
    if mode == "smoke":
        return (protocol.base_protocol.episode_map[protocol.smoke_episode_id],)
    if mode == "quick":
        return tuple(protocol.base_protocol.episode_map[item] for item in protocol.quick_episode_ids)
    if mode == "full":
        if protocol.protocol_status != "frozen":
            raise ValueError("full OOD requires protocol_status='frozen'")
        return protocol.base_protocol.episodes
    raise ValueError(f"unsupported mode: {mode}")


def _validate_nominal_source(protocol: ObservationOodProtocol, commit: str) -> dict[str, Any]:
    model = protocol.selected_model
    evidence = model["formal_evidence"]
    summary = _load_object(_resolve_project_path(evidence["position_summary"]))
    expected = {
        "run_mode": "full",
        "high_policy_sha256": model["high_level_policy"]["sha256"],
        "low_policy_sha256": model["low_level_policy"]["sha256"],
        "protocol_sha256": evidence["protocol_sha256"],
        "episode_manifest_sha256": evidence["episode_manifest_sha256"],
        "official_commit": commit,
        "k": 10,
    }
    mismatch = {key: {"expected": value, "actual": summary.get(key)} for key, value in expected.items() if summary.get(key) != value}
    if mismatch:
        raise ValueError(f"frozen nominal summary identity mismatch: {mismatch}")
    if len(summary.get("episodes", ())) != 200:
        raise ValueError("frozen nominal source must contain 200 episodes")
    return summary


def _run_signature(protocol: ObservationOodProtocol, commit: str, mode: str) -> str:
    model = protocol.selected_model
    paths = (
        protocol.source_path,
        protocol.base_protocol_path,
        protocol.selected_model_path,
        _resolve_project_path(model["high_level_policy"]["path"]),
        _resolve_project_path(model["low_level_policy"]["path"]),
        Path(__file__),
        PROJECT_ROOT / "evaluation" / "pointgoal_benchmark.py",
        PROJECT_ROOT / "evaluation" / "rl_pointgoal_quick.py",
        PROJECT_ROOT / "navigation" / "mujoco_backend.py",
        PROJECT_ROOT / "navigation" / "observation_perturbation.py",
        PROJECT_ROOT / "navigation" / "rl_navigator.py",
    )
    payload = {
        "files": {str(path): _sha256(path) for path in paths},
        "official_commit": commit,
        "mode": mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _progress_path(output_dir: Path, mode: str, condition_id: str, episode_id: str) -> Path:
    return output_dir / mode / condition_id / "progress" / f"{episode_id}.json"


def _load_progress(path: Path, *, signature: str, condition_id: str, episode_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    saved = _load_object(path)
    if saved.get("run_signature") != signature or saved.get("condition_id") != condition_id or saved.get("episode_id") != episode_id:
        raise ValueError(f"stale or mismatched progress: {path}")
    return dict(saved["episode"])


def _write_progress(path: Path, *, signature: str, condition: ObservationPerturbation, spec: EpisodeSpec, episode: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_signature": signature,
                "condition_id": condition.condition_id,
                "condition": asdict(condition),
                "episode_id": spec.episode_id,
                "episode_spec": asdict(spec),
                "episode": episode,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _paired_vs_nominal(nominal: Sequence[Mapping[str, Any]], ood: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nominal_map = {str(item["episode_id"]): item for item in nominal}
    ood_map = {str(item["episode_id"]): item for item in ood}
    if set(nominal_map) != set(ood_map):
        raise ValueError("nominal and OOD EpisodeSpec ids do not match")
    metrics = {}
    for key, preferred in PAIRED_METRICS.items():
        deltas = [float(ood_map[item][key]) - float(nominal_map[item][key]) for item in nominal_map if ood_map[item].get(key) is not None and nominal_map[item].get(key) is not None]
        metrics[key] = {"preferred": preferred, "pair_count": len(deltas), "mean_delta_ood_minus_nominal": (sum(deltas) / len(deltas) if deltas else None)}
    lost_successes = [item for item in nominal_map if nominal_map[item]["success"] and not ood_map[item]["success"]]
    return {"pair_count": len(nominal_map), "nominal_success_lost": lost_successes, "metrics": metrics}


def _evaluate_gate(aggregate: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_success_rate": ("success_rate", ">="),
        "maximum_fall_rate": ("fall_rate", "<="),
        "maximum_invalid_state_rate": ("invalid_state_rate", "<="),
        "maximum_post_arrival_redeparture_rate": ("post_arrival_redeparture_rate", "<="),
    }
    criteria = {}
    for name, threshold in gate.items():
        if name == "note":
            continue
        metric, operator = checks[name]
        observed = float(aggregate[metric])
        criteria[name] = {"metric": metric, "operator": operator, "threshold": float(threshold), "observed": observed, "passed": observed >= float(threshold) if operator == ">=" else observed <= float(threshold)}
    return {"passed": all(item["passed"] for item in criteria.values()), "criteria": criteria, "note": gate.get("note")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--condition", action="append", dest="condition_ids")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    protocol = load_protocol(args.config)
    root = args.microduck_rl_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model = protocol.selected_model
    high_path = _resolve_project_path(model["high_level_policy"]["path"])
    low_path = _resolve_project_path(model["low_level_policy"]["path"])
    commit = _official_commit(root)
    nominal_summary = _validate_nominal_source(protocol, commit)
    config = RLNavigatorConfig()
    config.validate()
    if config.max_forward_speed_mps != 0.25 or config.max_yaw_rate_radps != 0.5:
        raise ValueError("RL command limits differ from frozen selected-model contract")
    MujocoBackend(root, low_path, protocol.base_protocol_path, output_dir).validate()
    print(f"validated {protocol.protocol_id}: factor={protocol.factor}, K=10, high={_sha256(high_path)}, low={_sha256(low_path)}")
    if args.validate_only:
        return 0
    mode = "smoke" if args.smoke else "quick" if args.quick else "full"
    specs = select_specs(protocol, mode)
    selected_ids = tuple(args.condition_ids or [item.condition_id for item in protocol.conditions])
    condition_map = {item.condition_id: item for item in protocol.conditions}
    if set(selected_ids) - set(condition_map):
        parser.error(f"unknown OOD condition(s): {sorted(set(selected_ids) - set(condition_map))}")
    signature = _run_signature(protocol, commit, mode)
    runtime_protocol = replace(protocol.base_protocol, navigator_frequency_hz=5.0)
    policy = OnnxNavigationPolicy(high_path, 10)
    results = {}
    for condition_id in selected_ids:
        condition = condition_map[condition_id]
        condition_dir = output_dir / mode / condition.condition_id
        episodes = []
        for spec in specs:
            progress = _progress_path(output_dir, mode, condition.condition_id, spec.episode_id)
            result = None if args.no_resume else _load_progress(progress, signature=signature, condition_id=condition.condition_id, episode_id=spec.episode_id)
            if result is None:
                backend = MujocoBackend(root, low_path, protocol.base_protocol_path, condition_dir)
                navigator = ObservationPerturbedNavigator(
                    RLNavigator(policy, config), condition, episode_seed=spec.simulation_reset_seed
                )
                with patch.object(frozen_benchmark, "build_navigator", return_value=navigator):
                    result = run_episode(
                        backend=backend,
                        protocol=runtime_protocol,
                        spec=spec,
                        controller_id="rl_k10",
                        output_dir=condition_dir,
                        navigation_timeout_s=protocol.base_protocol.smoke_timeout_s if mode == "smoke" else None,
                    )
                result.update({
                    "condition_id": condition.condition_id,
                    "factor": condition.factor,
                    "observation_perturbation_readback": navigator.readback(),
                })
                if not result["observation_perturbation_readback"]["readback_passed"]:
                    raise RuntimeError(f"observation readback failed for {condition.condition_id}/{spec.episode_id}")
                _write_progress(progress, signature=signature, condition=condition, spec=spec, episode=result)
            episodes.append(result)
            print(f"{condition.condition_id}/{spec.episode_id}: {result['termination_reason']}, distance={result['final_distance_m']:.3f} m")
        selected = {item.episode_id for item in specs}
        nominal = [item for item in nominal_summary["episodes"] if item["episode_id"] in selected]
        aggregate = aggregate_controller(episodes)
        results[condition.condition_id] = {
            "condition": asdict(condition),
            "aggregate": aggregate,
            "paired_vs_nominal": _paired_vs_nominal(nominal, episodes),
            "episodes": episodes,
        }
        if mode in {"quick", "full"}:
            results[condition.condition_id]["gate"] = _evaluate_gate(aggregate, protocol.quick_gate if mode == "quick" else protocol.full_gate)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "week05_rl_observation_ood",
        "run_mode": mode,
        "protocol_id": protocol.protocol_id,
        "protocol_status": protocol.protocol_status,
        "ood_protocol_sha256": _sha256(protocol.source_path),
        "base_protocol_sha256": _sha256(protocol.base_protocol_path),
        "episode_manifest_sha256": protocol.base_protocol.episode_manifest_sha256,
        "selected_model_manifest_sha256": _sha256(protocol.selected_model_path),
        "high_policy_path": str(high_path),
        "high_policy_sha256": _sha256(high_path),
        "low_policy_path": str(low_path),
        "low_policy_sha256": _sha256(low_path),
        "official_commit": commit,
        "k": 10,
        "navigator_frequency_hz": 5.0,
        "reward_available": False,
        "truth_metric_note": "Robot and goal truth metrics remain unperturbed MuJoCo state; only inputs to the high-level Navigator are perturbed.",
        "conditions": results,
    }
    if mode in {"quick", "full"}:
        summary["all_conditions_passed"] = all(item["gate"]["passed"] for item in results.values())
    summary_path = output_dir / mode / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
