#!/usr/bin/env python3
"""Run lightweight OOD checks for the frozen winning Classical controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.locomotion_benchmark import (
    EXPECTED_ACTION_SIZE,
    EXPECTED_OBSERVATION_SIZE,
    RuntimePerturbation,
    path_for_record,
)
from evaluation.pointgoal_benchmark import aggregate_controller, run_episode
from evaluation.pointgoal_protocol import (
    ClassicalPointGoalProtocol,
    EpisodeSpec,
    load_pointgoal_protocol,
)
from navigation.mujoco_backend import MujocoBackend


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "artifacts"
    / "week03"
    / "06_yaw_only_tracking_candidate"
    / "models"
    / "model_1500.onnx"
)
DEFAULT_CONFIG = (
    Path(__file__).parent
    / "pointgoal_protocols"
    / "week04_classical_ood_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "week04"
    / "02_classical_pointgoal_ood"
    / "model_1500_constrained"
)
CONDITION_FACTORS = {
    "foot_friction",
    "actuator_delay",
    "motor_strength_proxy",
}
PAIRED_METRICS = {
    "final_distance_m": "lower",
    "path_length_to_arrival_m": "lower",
    "path_efficiency": "higher",
    "completion_time_s": "lower",
    "vx_saturation_fraction": "lower",
    "wz_saturation_fraction": "lower",
    "post_arrival_stop_drift_m": "lower",
    "post_arrival_max_drift_m": "lower",
    "post_arrival_max_goal_distance_m": "lower",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


@dataclass(frozen=True)
class OodCondition:
    condition_id: str
    factor: str
    foot_friction: float | None = None
    actuator_delay_ms: int = 0
    bam_voltage_scale: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OodCondition":
        condition = cls(
            condition_id=str(raw.get("id", "")),
            factor=str(raw.get("factor", "")),
            foot_friction=(
                None
                if raw.get("foot_friction") is None
                else float(raw["foot_friction"])
            ),
            actuator_delay_ms=int(raw.get("actuator_delay_ms", 0)),
            bam_voltage_scale=float(raw.get("bam_voltage_scale", 1.0)),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if not self.condition_id:
            raise ValueError("OOD condition id must not be empty")
        if self.factor not in CONDITION_FACTORS:
            raise ValueError(f"unknown OOD factor: {self.factor!r}")
        active = {
            "foot_friction": self.foot_friction is not None,
            "actuator_delay": self.actuator_delay_ms != 0,
            "motor_strength_proxy": self.bam_voltage_scale != 1.0,
        }
        actual = {name for name, enabled in active.items() if enabled}
        if actual != {self.factor}:
            raise ValueError(
                f"condition {self.condition_id!r} must change only {self.factor!r}; "
                f"active perturbations: {sorted(actual)}"
            )
        self.to_runtime_perturbation().validate()

    def to_runtime_perturbation(self) -> RuntimePerturbation:
        return RuntimePerturbation(
            foot_friction=self.foot_friction,
            actuator_delay_ms=self.actuator_delay_ms,
            bam_voltage_scale=self.bam_voltage_scale,
        )


@dataclass(frozen=True)
class ClassicalOodProtocol:
    source_path: Path
    protocol_id: str
    protocol_status: str
    description: str
    base_protocol: ClassicalPointGoalProtocol
    nominal_summary_path: Path
    nominal_summary_sha256: str
    controller_id: str
    conditions: tuple[OodCondition, ...]
    selections: dict[str, tuple[str, ...]]
    smoke_timeout_s: float
    quick_gate: dict[str, Any]
    full_gate: dict[str, Any]

    @property
    def condition_map(self) -> dict[str, OodCondition]:
        return {condition.condition_id: condition for condition in self.conditions}


def _resolve_record_path(raw: Any, source_path: Path) -> Path:
    value = Path(str(raw)).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def load_ood_protocol(path: Path) -> ClassicalOodProtocol:
    source_path = path.expanduser().resolve()
    document = _load_json(source_path)
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported OOD protocol schema_version")
    status = str(document.get("protocol_status", ""))
    if status not in {"draft", "frozen"}:
        raise ValueError("OOD protocol_status must be 'draft' or 'frozen'")

    base_path = _resolve_record_path(document.get("base_protocol_path"), source_path)
    if _sha256(base_path) != str(document.get("base_protocol_sha256", "")):
        raise ValueError("base PointGoal protocol SHA-256 mismatch")
    base_protocol = load_pointgoal_protocol(base_path)
    if base_protocol.protocol_status != "frozen":
        raise ValueError("base PointGoal protocol must be frozen")

    nominal_path = _resolve_record_path(
        document.get("nominal_full_summary_path"), source_path
    )
    nominal_sha = str(document.get("nominal_full_summary_sha256", ""))
    if _sha256(nominal_path) != nominal_sha:
        raise ValueError("nominal full summary SHA-256 mismatch")

    controller_id = str(document.get("controller_id", ""))
    if controller_id != "constrained":
        raise ValueError("Week 4 OOD protocol must use the winning constrained controller")

    conditions = tuple(
        OodCondition.from_mapping(raw) for raw in document.get("conditions", ())
    )
    if tuple(condition.factor for condition in conditions) != (
        "foot_friction",
        "actuator_delay",
        "motor_strength_proxy",
    ):
        raise ValueError("OOD conditions must keep the frozen one-factor order")
    condition_ids = [condition.condition_id for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("OOD condition ids must be unique")

    raw_selections = document.get("selections", {})
    selections = {
        mode: tuple(str(value) for value in raw_selections.get(mode, ()))
        for mode in ("smoke", "quick", "full")
    }
    expected_sizes = {"smoke": 1, "quick": 10, "full": 25}
    known_ids = set(base_protocol.episode_map)
    for mode, expected_size in expected_sizes.items():
        ids = selections[mode]
        if len(ids) != expected_size or len(set(ids)) != expected_size:
            raise ValueError(f"{mode} selection must contain {expected_size} unique ids")
        if not set(ids) <= known_ids:
            raise ValueError(f"{mode} selection contains unknown EpisodeSpec ids")
    if set(selections["quick"]) & set(selections["full"]):
        raise ValueError("quick and full OOD EpisodeSpecs must be disjoint")

    smoke_timeout_s = float(document.get("smoke_timeout_s", 0.0))
    if not math.isfinite(smoke_timeout_s) or smoke_timeout_s <= 0:
        raise ValueError("smoke_timeout_s must be finite and > 0")

    return ClassicalOodProtocol(
        source_path=source_path,
        protocol_id=str(document.get("protocol_id", "")),
        protocol_status=status,
        description=str(document.get("description", "")),
        base_protocol=base_protocol,
        nominal_summary_path=nominal_path,
        nominal_summary_sha256=nominal_sha,
        controller_id=controller_id,
        conditions=conditions,
        selections=selections,
        smoke_timeout_s=smoke_timeout_s,
        quick_gate=dict(document.get("quick_gate", {})),
        full_gate=dict(document.get("full_gate", {})),
    )


def select_specs(protocol: ClassicalOodProtocol, mode: str) -> tuple[EpisodeSpec, ...]:
    if mode == "full" and protocol.protocol_status != "frozen":
        raise ValueError("full OOD requires protocol_status='frozen'")
    try:
        ids = protocol.selections[mode]
    except KeyError as error:
        raise ValueError(f"unsupported OOD mode: {mode!r}") from error
    return tuple(protocol.base_protocol.episode_map[episode_id] for episode_id in ids)


def _validate_nominal_source(
    protocol: ClassicalOodProtocol,
    policy_path: Path,
    official_commit: str,
) -> dict[str, Any]:
    document = _load_json(protocol.nominal_summary_path)
    expected = {
        "run_mode": "full",
        "protocol_id": protocol.base_protocol.protocol_id,
        "protocol_status": "frozen",
        "protocol_sha256": _sha256(protocol.base_protocol.source_path),
        "episode_manifest_sha256": protocol.base_protocol.episode_manifest_sha256,
        "policy_sha256": _sha256(policy_path),
        "microduck_rl_commit": official_commit,
    }
    mismatches = {
        key: {"expected": value, "actual": document.get(key)}
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        raise ValueError(f"nominal full summary identity mismatch: {mismatches}")
    episodes = document.get("controllers", {}).get(protocol.controller_id, {}).get(
        "episodes", []
    )
    if len(episodes) != 200:
        raise ValueError("nominal source must contain 200 constrained episodes")
    return document


def _run_signature(
    *,
    policy_path: Path,
    protocol: ClassicalOodProtocol,
    official_commit: str,
    mode: str,
) -> str:
    evaluator_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "evaluation" / "pointgoal_benchmark.py",
        PROJECT_ROOT / "evaluation" / "pointgoal_protocol.py",
        PROJECT_ROOT / "evaluation" / "locomotion_benchmark.py",
        PROJECT_ROOT / "navigation" / "mujoco_backend.py",
        PROJECT_ROOT / "navigation" / "classical_navigator.py",
        PROJECT_ROOT / "navigation" / "types.py",
    )
    payload = {
        "policy_sha256": _sha256(policy_path),
        "ood_protocol_sha256": _sha256(protocol.source_path),
        "base_protocol_sha256": _sha256(protocol.base_protocol.source_path),
        "nominal_summary_sha256": protocol.nominal_summary_sha256,
        "microduck_rl_commit": official_commit,
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
    output_dir: Path, mode: str, condition_id: str, episode_id: str
) -> Path:
    return output_dir / mode / condition_id / "progress" / f"{episode_id}.json"


def _load_progress(
    path: Path, *, signature: str, condition_id: str, episode_id: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = _load_json(path)
    if document.get("run_signature") != signature:
        raise ValueError(f"stale progress signature at {path}; use a new output directory")
    if document.get("condition_id") != condition_id:
        raise ValueError(f"progress condition mismatch at {path}")
    if document.get("episode_id") != episode_id:
        raise ValueError(f"progress EpisodeSpec mismatch at {path}")
    return dict(document["episode"])


def _write_progress(
    path: Path,
    *,
    signature: str,
    condition: OodCondition,
    spec: EpisodeSpec,
    episode: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "run_signature": signature,
        "condition_id": condition.condition_id,
        "condition": asdict(condition),
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


def _paired_vs_nominal(
    nominal: Sequence[Mapping[str, Any]],
    ood: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    nominal_map = {str(item["episode_id"]): item for item in nominal}
    ood_map = {str(item["episode_id"]): item for item in ood}
    if set(nominal_map) != set(ood_map):
        raise ValueError("nominal and OOD EpisodeSpec ids do not match")
    ood_success_wins = 0
    nominal_success_wins = 0
    metrics: dict[str, Any] = {}
    for episode_id in nominal_map:
        nominal_success = bool(nominal_map[episode_id]["success"])
        ood_success = bool(ood_map[episode_id]["success"])
        ood_success_wins += ood_success and not nominal_success
        nominal_success_wins += nominal_success and not ood_success
    for key, preference in PAIRED_METRICS.items():
        deltas = []
        ood_better = nominal_better = ties = 0
        for episode_id in nominal_map:
            nominal_value = nominal_map[episode_id].get(key)
            ood_value = ood_map[episode_id].get(key)
            if nominal_value is None or ood_value is None:
                continue
            delta = float(ood_value) - float(nominal_value)
            deltas.append(delta)
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                ties += 1
            elif (delta < 0.0) == (preference == "lower"):
                ood_better += 1
            else:
                nominal_better += 1
        metrics[key] = {
            "preferred": preference,
            "pair_count": len(deltas),
            "median_delta_ood_minus_nominal": (
                median(deltas) if deltas else None
            ),
            "ood_better_count": ood_better,
            "nominal_better_count": nominal_better,
            "tie_count": ties,
        }
    return {
        "pair_count": len(nominal_map),
        "success": {
            "ood_wins": ood_success_wins,
            "nominal_wins": nominal_success_wins,
            "ties": len(nominal_map) - ood_success_wins - nominal_success_wins,
        },
        "metrics": metrics,
    }


def _evaluate_gate(aggregate: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_success_rate": ("success_rate", ">="),
        "maximum_fall_rate": ("fall_rate", "<="),
        "maximum_timeout_rate": ("timeout_rate", "<="),
        "maximum_invalid_state_rate": ("invalid_state_rate", "<="),
        "maximum_post_arrival_redeparture_rate": (
            "post_arrival_redeparture_rate",
            "<=",
        ),
    }
    criteria = {}
    for criterion, (metric, operator) in checks.items():
        threshold = float(gate[criterion])
        observed = float(aggregate[metric])
        passed = observed >= threshold if operator == ">=" else observed <= threshold
        criteria[criterion] = {
            "metric": metric,
            "observed": observed,
            "threshold": threshold,
            "operator": operator,
            "passed": passed,
        }
    return {
        "passed": all(item["passed"] for item in criteria.values()),
        "criteria": criteria,
        "note": gate.get("note"),
    }


def _write_summary(
    *,
    protocol: ClassicalOodProtocol,
    policy_path: Path,
    microduck_rl_root: Path,
    output_dir: Path,
    mode: str,
    signature: str,
    specs: Sequence[EpisodeSpec],
    nominal_summary: Mapping[str, Any],
    conditions: Sequence[tuple[OodCondition, dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
) -> Path:
    selected_ids = {spec.episode_id for spec in specs}
    nominal_all = nominal_summary["controllers"][protocol.controller_id]["episodes"]
    nominal_episodes = [
        episode for episode in nominal_all if episode["episode_id"] in selected_ids
    ]
    if len(nominal_episodes) != len(specs):
        raise ValueError("nominal source is missing selected EpisodeSpecs")
    nominal_aggregate = aggregate_controller(nominal_episodes)
    gate_config = protocol.quick_gate if mode == "quick" else protocol.full_gate
    condition_documents = {}
    for condition, aggregate, episodes, runtime_record in conditions:
        item = {
            "condition": asdict(condition),
            "runtime_perturbation": runtime_record,
            "aggregate": aggregate,
            "paired_vs_nominal": _paired_vs_nominal(nominal_episodes, episodes),
            "episodes": episodes,
        }
        if mode in {"quick", "full"}:
            item["gate"] = _evaluate_gate(aggregate, gate_config)
        condition_documents[condition.condition_id] = item
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "classical_pointgoal_winner_ood_v1",
        "run_mode": mode,
        "protocol_id": protocol.protocol_id,
        "protocol_status": protocol.protocol_status,
        "ood_protocol_path": path_for_record(protocol.source_path, PROJECT_ROOT),
        "ood_protocol_sha256": _sha256(protocol.source_path),
        "base_protocol_path": path_for_record(
            protocol.base_protocol.source_path, PROJECT_ROOT
        ),
        "base_protocol_sha256": _sha256(protocol.base_protocol.source_path),
        "episode_manifest_sha256": protocol.base_protocol.episode_manifest_sha256,
        "nominal_full_summary_path": path_for_record(
            protocol.nominal_summary_path, PROJECT_ROOT
        ),
        "nominal_full_summary_sha256": protocol.nominal_summary_sha256,
        "run_signature": signature,
        "policy_path": path_for_record(policy_path, microduck_rl_root, PROJECT_ROOT),
        "policy_sha256": _sha256(policy_path),
        "microduck_rl_commit": _official_commit(microduck_rl_root),
        "controller_id": protocol.controller_id,
        "controller_config": asdict(protocol.base_protocol.constrained),
        "pose_source": "simulator_ground_truth_via_MujocoBackend",
        "reward_available": False,
        "runtime_contract": {
            "observation_size": EXPECTED_OBSERVATION_SIZE,
            "action_size": EXPECTED_ACTION_SIZE,
            "navigator_action_scope": "[vx, 0, wz]",
        },
        "episode_ids": [spec.episode_id for spec in specs],
        "nominal_baseline": {
            "source_run_signature": nominal_summary["run_signature"],
            "aggregate": nominal_aggregate,
            "episodes": nominal_episodes,
        },
        "conditions": condition_documents,
    }
    if mode in {"quick", "full"}:
        document["all_conditions_passed"] = all(
            item["gate"]["passed"] for item in condition_documents.values()
        )
    summary_dir = output_dir / mode / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"classical_pointgoal_ood_{mode}_{policy_path.stem}.json"
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--condition",
        action="append",
        dest="condition_ids",
        help="Run only this planned condition; repeat to select several.",
    )
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
    config_path = _resolved(args.config)
    output_dir = _resolved(args.output_dir)
    protocol = load_ood_protocol(config_path)
    official_commit = _official_commit(microduck_rl_root)
    nominal_summary = _validate_nominal_source(protocol, policy_path, official_commit)
    validation_backend = MujocoBackend(
        microduck_rl_root=microduck_rl_root,
        policy_path=policy_path,
        metadata_path=config_path,
        output_dir=output_dir,
    )
    validation_backend.validate()
    print("Week 4 Classical winner OOD contract is valid")
    print(f"  protocol:          {protocol.protocol_id} ({protocol.protocol_status})")
    print(f"  controller:        {protocol.controller_id}")
    print(f"  conditions:        {', '.join(protocol.condition_map)}")
    print("  quick/full:        10/25 disjoint EpisodeSpecs per condition")
    print(f"  policy SHA-256:    {_sha256(policy_path)}")
    print(f"  nominal SHA-256:   {protocol.nominal_summary_sha256}")
    print(f"  microduck_rl HEAD: {official_commit}")
    if args.validate_only:
        return 0

    if args.viewer:
        selected_ids = tuple(args.condition_ids or ())
        if len(selected_ids) != 1:
            parser.error("--viewer requires exactly one --condition")
        condition_id = selected_ids[0]
        if condition_id not in protocol.condition_map:
            parser.error(f"unknown OOD condition: {condition_id!r}")
        condition = protocol.condition_map[condition_id]

        episode_id = (
            args.episode_id
            or protocol.base_protocol.smoke_episode_ids["front"]
        )
        if episode_id not in protocol.base_protocol.episode_map:
            parser.error(f"unknown EpisodeSpec id: {episode_id!r}")
        spec = protocol.base_protocol.episode_map[episode_id]
        viewer_dir = (
            output_dir
            / "viewer"
            / condition.condition_id
            / protocol.controller_id
            / episode_id
        )
        backend = MujocoBackend(
            microduck_rl_root=microduck_rl_root,
            policy_path=policy_path,
            metadata_path=config_path,
            output_dir=viewer_dir,
            perturbation=condition.to_runtime_perturbation(),
        )
        backend.validate()
        backend.reset(
            spec.simulation_reset_seed,
            protocol.base_protocol.initial_state_mode,
        )
        print(
            "viewer runtime perturbation: "
            f"{json.dumps(backend.perturbation_record, ensure_ascii=False)}"
        )
        try:
            with backend.open_viewer() as viewer:
                result = run_episode(
                    backend=backend,
                    protocol=protocol.base_protocol,
                    spec=spec,
                    controller_id=protocol.controller_id,
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
        specs = select_specs(protocol, mode)
    except ValueError as error:
        parser.error(str(error))
    selected_ids = tuple(args.condition_ids or protocol.condition_map)
    unknown = set(selected_ids) - set(protocol.condition_map)
    if unknown:
        parser.error(f"unknown OOD conditions: {sorted(unknown)}")
    conditions = tuple(
        condition
        for condition in protocol.conditions
        if condition.condition_id in selected_ids
    )
    signature = _run_signature(
        policy_path=policy_path,
        protocol=protocol,
        official_commit=official_commit,
        mode=mode,
    )
    condition_results = []
    for condition in conditions:
        condition_dir = output_dir / mode / condition.condition_id
        backend = MujocoBackend(
            microduck_rl_root=microduck_rl_root,
            policy_path=policy_path,
            metadata_path=config_path,
            output_dir=condition_dir,
            perturbation=condition.to_runtime_perturbation(),
        )
        backend.validate()
        episodes = []
        for spec in specs:
            progress_path = _progress_path(
                output_dir, mode, condition.condition_id, spec.episode_id
            )
            result = (
                None
                if args.no_resume
                else _load_progress(
                    progress_path,
                    signature=signature,
                    condition_id=condition.condition_id,
                    episode_id=spec.episode_id,
                )
            )
            if result is None:
                result = run_episode(
                    backend=backend,
                    protocol=protocol.base_protocol,
                    spec=spec,
                    controller_id=protocol.controller_id,
                    output_dir=condition_dir,
                    navigation_timeout_s=(
                        protocol.smoke_timeout_s if mode == "smoke" else None
                    ),
                )
                result.update(
                    {
                        "condition_id": condition.condition_id,
                        "factor": condition.factor,
                    }
                )
                _write_progress(
                    progress_path,
                    signature=signature,
                    condition=condition,
                    spec=spec,
                    episode=result,
                )
            episodes.append(result)
            print(
                f"{condition.condition_id}/{spec.episode_id} "
                f"seed={spec.simulation_reset_seed}: {result['termination_reason']}, "
                f"final_distance={result['final_distance_m']:.3f} m"
            )
        condition_results.append(
            (
                condition,
                aggregate_controller(episodes),
                episodes,
                backend.perturbation_record,
            )
        )
    summary_path = _write_summary(
        protocol=protocol,
        policy_path=policy_path,
        microduck_rl_root=microduck_rl_root,
        output_dir=output_dir,
        mode=mode,
        signature=signature,
        specs=specs,
        nominal_summary=nominal_summary,
        conditions=condition_results,
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
