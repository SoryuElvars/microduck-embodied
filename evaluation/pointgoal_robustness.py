#!/usr/bin/env python3
"""Run a paired, one-factor-at-a-time PointGoal robustness protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.locomotion_benchmark import (
    EXPECTED_ACTION_SIZE,
    EXPECTED_OBSERVATION_SIZE,
    PROJECT_ROOT,
    RuntimePerturbation,
    path_for_record,
)
from evaluation.pointgoal_pilot import (
    PilotProtocol,
    aggregate_episodes,
    load_goal_set,
    run_episode,
)
from navigation.mujoco_backend import MujocoBackend


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
    Path(__file__).parent / "robustness_configs" / "pointgoal_ood_v1.json"
)
DEFAULT_SENSOR_MASS_CONFIG = (
    Path(__file__).parent
    / "robustness_configs"
    / "pointgoal_sensor_mass_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "week03"
    / "08_pointgoal_robustness"
    / "model_1500"
)
FACTORS = {
    "nominal",
    "foot_friction",
    "actuator_delay",
    "motor_strength_proxy",
    "backlash",
    "mass_inertia",
    "imu_observation_noise",
    "joint_encoder_noise",
}


@dataclass(frozen=True)
class RobustnessCondition:
    condition_id: str
    factor: str
    foot_friction: float | None = None
    actuator_delay_ms: int = 0
    bam_voltage_scale: float = 1.0
    scene_xml_path: str | None = None
    trunk_mass_inertia_scale: float = 1.0
    imu_ang_vel_noise_uniform_radps: float = 0.0
    imu_gravity_noise_uniform: float = 0.0
    joint_pos_noise_uniform_rad: float = 0.0
    joint_vel_noise_uniform_radps: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "RobustnessCondition":
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
            scene_xml_path=(
                None
                if raw.get("scene_xml_path") is None
                else str(raw["scene_xml_path"])
            ),
            trunk_mass_inertia_scale=float(
                raw.get("trunk_mass_inertia_scale", 1.0)
            ),
            imu_ang_vel_noise_uniform_radps=float(
                raw.get("imu_ang_vel_noise_uniform_radps", 0.0)
            ),
            imu_gravity_noise_uniform=float(
                raw.get("imu_gravity_noise_uniform", 0.0)
            ),
            joint_pos_noise_uniform_rad=float(
                raw.get("joint_pos_noise_uniform_rad", 0.0)
            ),
            joint_vel_noise_uniform_radps=float(
                raw.get("joint_vel_noise_uniform_radps", 0.0)
            ),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if not self.condition_id.strip():
            raise ValueError("robustness condition id must not be empty")
        if self.factor not in FACTORS:
            raise ValueError(f"unknown robustness factor: {self.factor!r}")
        active = {
            "foot_friction": self.foot_friction is not None,
            "actuator_delay": self.actuator_delay_ms != 0,
            "motor_strength_proxy": self.bam_voltage_scale != 1.0,
            "backlash": self.scene_xml_path is not None,
            "mass_inertia": self.trunk_mass_inertia_scale != 1.0,
            "imu_observation_noise": (
                self.imu_ang_vel_noise_uniform_radps != 0.0
                or self.imu_gravity_noise_uniform != 0.0
            ),
            "joint_encoder_noise": (
                self.joint_pos_noise_uniform_rad != 0.0
                or self.joint_vel_noise_uniform_radps != 0.0
            ),
        }
        expected = set() if self.factor == "nominal" else {self.factor}
        actual = {name for name, enabled in active.items() if enabled}
        if actual != expected:
            raise ValueError(
                f"condition {self.condition_id!r} must change only {self.factor!r}; "
                f"active perturbations: {sorted(actual)}"
            )
        if self.factor == "imu_observation_noise" and not (
            self.imu_ang_vel_noise_uniform_radps > 0
            and self.imu_gravity_noise_uniform > 0
        ):
            raise ValueError(
                "imu_observation_noise must define positive angular-velocity "
                "and projected-gravity half-ranges"
            )
        if self.factor == "joint_encoder_noise" and not (
            self.joint_pos_noise_uniform_rad > 0
            and self.joint_vel_noise_uniform_radps > 0
        ):
            raise ValueError(
                "joint_encoder_noise must define positive joint-position and "
                "joint-velocity half-ranges"
            )
        RuntimePerturbation(
            foot_friction=self.foot_friction,
            actuator_delay_ms=self.actuator_delay_ms,
            bam_voltage_scale=self.bam_voltage_scale,
            trunk_mass_inertia_scale=self.trunk_mass_inertia_scale,
            imu_ang_vel_noise_uniform_radps=(
                self.imu_ang_vel_noise_uniform_radps
            ),
            imu_gravity_noise_uniform=self.imu_gravity_noise_uniform,
            joint_pos_noise_uniform_rad=self.joint_pos_noise_uniform_rad,
            joint_vel_noise_uniform_radps=self.joint_vel_noise_uniform_radps,
        ).validate()

    def to_runtime_perturbation(
        self, microduck_rl_root: Path
    ) -> RuntimePerturbation:
        scene = (
            None
            if self.scene_xml_path is None
            else (microduck_rl_root / self.scene_xml_path).resolve()
        )
        perturbation = RuntimePerturbation(
            foot_friction=self.foot_friction,
            actuator_delay_ms=self.actuator_delay_ms,
            bam_voltage_scale=self.bam_voltage_scale,
            scene_xml_path=scene,
            trunk_mass_inertia_scale=self.trunk_mass_inertia_scale,
            imu_ang_vel_noise_uniform_radps=(
                self.imu_ang_vel_noise_uniform_radps
            ),
            imu_gravity_noise_uniform=self.imu_gravity_noise_uniform,
            joint_pos_noise_uniform_rad=self.joint_pos_noise_uniform_rad,
            joint_vel_noise_uniform_radps=self.joint_vel_noise_uniform_radps,
        )
        perturbation.validate()
        return perturbation


@dataclass(frozen=True)
class RobustnessProtocol:
    source_path: Path
    protocol_status: str
    description: str
    goal_set_path: Path
    output_subdir: str | None
    quick_episodes_per_goal: int
    quick_condition_ids: tuple[str, ...]
    quick_safety_gate: dict[str, Any]
    full_episodes_per_goal: int
    full_condition_ids: tuple[str, ...]
    full_acceptance_gate: dict[str, Any]
    conditions: tuple[RobustnessCondition, ...]

    @property
    def condition_map(self) -> dict[str, RobustnessCondition]:
        return {condition.condition_id: condition for condition in self.conditions}


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_robustness_protocol(path: Path) -> RobustnessProtocol:
    """Load and validate the frozen OOD factor matrix."""

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError("robustness config schema_version must be 1")
    status = str(document.get("protocol_status", ""))
    if status not in {"draft", "frozen"}:
        raise ValueError("protocol_status must be 'draft' or 'frozen'")
    quick = document.get("quick")
    full = document.get("full")
    raw_conditions = document.get("conditions")
    if not isinstance(quick, dict) or not isinstance(full, dict):
        raise ValueError("quick and full robustness plans must be objects")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions must be a non-empty list")
    conditions = tuple(
        RobustnessCondition.from_mapping(raw) for raw in raw_conditions
    )
    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("robustness condition ids must be unique")
    known = set(ids)

    def plan_ids(plan: dict[str, Any], name: str) -> tuple[str, ...]:
        values = tuple(str(value) for value in plan.get("condition_ids", []))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name}.condition_ids must be non-empty and unique")
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"{name} references unknown conditions: {sorted(unknown)}")
        return values

    quick_episodes = int(quick.get("episodes_per_goal", 0))
    full_episodes = int(full.get("episodes_per_goal", 0))
    if quick_episodes < 1 or full_episodes < 1:
        raise ValueError("episodes_per_goal must be >= 1")
    goal_set_path = _resolve_project_path(str(document.get("goal_set_path", "")))
    load_goal_set(goal_set_path)
    raw_output_subdir = document.get("output_subdir")
    output_subdir = None if raw_output_subdir is None else str(raw_output_subdir)
    if output_subdir is not None:
        output_path = Path(output_subdir)
        if (
            not output_subdir.strip()
            or output_path.is_absolute()
            or ".." in output_path.parts
        ):
            raise ValueError("output_subdir must be a safe relative path")
    return RobustnessProtocol(
        source_path=path.resolve(),
        protocol_status=status,
        description=str(document.get("description", "")),
        goal_set_path=goal_set_path,
        output_subdir=output_subdir,
        quick_episodes_per_goal=quick_episodes,
        quick_condition_ids=plan_ids(quick, "quick"),
        quick_safety_gate=dict(quick.get("safety_gate", {})),
        full_episodes_per_goal=full_episodes,
        full_condition_ids=plan_ids(full, "full"),
        full_acceptance_gate=dict(full.get("acceptance_gate", {})),
        conditions=conditions,
    )


def select_conditions(
    protocol: RobustnessProtocol,
    mode: str,
    requested_ids: Sequence[str] | None = None,
) -> tuple[RobustnessCondition, ...]:
    """Select an ordered protocol subset without silently adding conditions."""

    planned_ids = (
        protocol.quick_condition_ids if mode == "quick" else protocol.full_condition_ids
    )
    requested = tuple(requested_ids or ())
    if requested:
        unknown = set(requested) - set(planned_ids)
        if unknown:
            raise ValueError(
                f"requested conditions are outside the {mode} plan: {sorted(unknown)}"
            )
        selected_ids = tuple(value for value in planned_ids if value in requested)
    else:
        selected_ids = planned_ids
    mapping = protocol.condition_map
    return tuple(mapping[value] for value in selected_ids)


def protocol_for_mode(
    goal_protocol: PilotProtocol,
    episodes_per_goal: int,
) -> PilotProtocol:
    return replace(
        goal_protocol,
        goals=tuple(
            replace(goal, episodes=episodes_per_goal) for goal in goal_protocol.goals
        ),
    )


def evaluate_condition_gate(
    aggregate: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate predeclared gates and retain every criterion separately."""

    observed_mirror_gap = max(
        (
            abs(float(item["left_minus_right_success_rate"]))
            for item in aggregate["mirror_groups"].values()
            if item.get("complete")
        ),
        default=0.0,
    )
    observed_per_goal = min(
        float(item["success_rate"]) for item in aggregate["goals"].values()
    )
    comparisons = {
        "minimum_overall_success_rate": float(aggregate["success_rate"]),
        "minimum_per_goal_success_rate": observed_per_goal,
        "maximum_fall_rate": float(aggregate["fall_rate"]),
        "maximum_timeout_rate": float(aggregate["timeout_rate"]),
        "maximum_invalid_state_rate": float(aggregate["invalid_state_rate"]),
        "maximum_absolute_mirror_success_rate_gap": observed_mirror_gap,
    }
    criteria: dict[str, Any] = {}
    for name, threshold in gate.items():
        if name == "note":
            continue
        if name not in comparisons:
            raise ValueError(f"unsupported gate criterion: {name}")
        observed = comparisons[name]
        is_minimum = name.startswith("minimum_")
        passed = observed >= float(threshold) if is_minimum else observed <= float(threshold)
        criteria[name] = {
            "observed": observed,
            "threshold": float(threshold),
            "passed": passed,
        }
    return {
        "passed": all(item["passed"] for item in criteria.values()),
        "criteria": criteria,
        "note": gate.get("note"),
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


def _run_signature(
    policy_path: Path,
    robustness_protocol: RobustnessProtocol,
    goal_protocol: PilotProtocol,
    microduck_rl_commit: str,
    mode: str,
) -> str:
    evaluator_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "evaluation" / "pointgoal_pilot.py",
        PROJECT_ROOT / "evaluation" / "locomotion_benchmark.py",
        PROJECT_ROOT / "navigation" / "mujoco_backend.py",
        PROJECT_ROOT / "navigation" / "classical_navigator.py",
    )
    payload = {
        "policy_sha256": _sha256(policy_path),
        "robustness_config_sha256": _sha256(robustness_protocol.source_path),
        "goal_set_sha256": _sha256(goal_protocol.source_path),
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
    condition_id: str,
    goal_name: str,
    episode_index: int,
    seed: int,
) -> Path:
    return (
        output_dir
        / mode
        / condition_id
        / "progress"
        / f"{goal_name}_seed{seed}_episode{episode_index:03d}.json"
    )


def _load_progress(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("run_signature") != signature:
        raise ValueError(
            f"stale progress signature at {path}; use a new output directory"
        )
    return dict(document["episode"])


def _write_progress(
    path: Path,
    signature: str,
    condition: RobustnessCondition,
    episode: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "run_signature": signature,
        "condition": asdict(condition),
        "episode": episode,
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_condition(
    *,
    condition: RobustnessCondition,
    goal_protocol: PilotProtocol,
    microduck_rl_root: Path,
    policy_path: Path,
    output_dir: Path,
    mode: str,
    signature: str,
    resume: bool,
    smoke_goal: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run or resume one condition using paired goal/reset seeds."""

    perturbation = condition.to_runtime_perturbation(microduck_rl_root)
    condition_dir = output_dir / mode / condition.condition_id
    backend = MujocoBackend(
        microduck_rl_root=microduck_rl_root,
        policy_path=policy_path,
        metadata_path=goal_protocol.source_path,
        output_dir=condition_dir,
        perturbation=perturbation,
    )
    backend.validate()
    goals = tuple(
        goal for goal in goal_protocol.goals if smoke_goal is None or goal.name == smoke_goal
    )
    if not goals:
        raise ValueError(f"unknown smoke goal: {smoke_goal!r}")
    episodes: list[dict[str, Any]] = []
    for goal in goals:
        episode_count = 1 if smoke_goal is not None else goal.episodes
        for episode_index in range(episode_count):
            seed = goal_protocol.base_seed + episode_index
            progress_path = _progress_path(
                output_dir,
                mode,
                condition.condition_id,
                goal.name,
                episode_index,
                seed,
            )
            result = _load_progress(progress_path, signature) if resume else None
            if result is None:
                result = run_episode(
                    backend,
                    goal_protocol,
                    replace(goal, episodes=1),
                    episode_index,
                    condition_dir,
                    smoke=smoke_goal is not None,
                    episode_seed=seed,
                )
                result.update(
                    {
                        "condition_id": condition.condition_id,
                        "factor": condition.factor,
                    }
                )
                _write_progress(progress_path, signature, condition, result)
            episodes.append(result)
            print(
                f"{condition.condition_id}/{goal.name} seed={seed}: "
                f"{result['termination_reason']}, "
                f"final_distance={result['final_distance_m']:.3f} m"
            )
    return aggregate_episodes(episodes), {"episodes": episodes, "runtime": backend.perturbation_record}


def write_suite_summary(
    *,
    robustness_protocol: RobustnessProtocol,
    goal_protocol: PilotProtocol,
    policy_path: Path,
    microduck_rl_root: Path,
    output_dir: Path,
    mode: str,
    signature: str,
    condition_documents: list[dict[str, Any]],
) -> Path:
    summary_dir = output_dir / mode / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    gate = (
        robustness_protocol.quick_safety_gate
        if mode == "quick"
        else robustness_protocol.full_acceptance_gate
        if mode == "full"
        else None
    )
    if gate is not None:
        for condition in condition_documents:
            condition["gate"] = evaluate_condition_gate(condition["aggregate"], gate)
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "classical_pointgoal_onnx_bam_ood",
        "run_mode": mode,
        "protocol_status": robustness_protocol.protocol_status,
        "one_factor_at_a_time": True,
        "paired_episode_seeds_across_conditions": True,
        "reward_available": False,
        "pose_source": "simulator_ground_truth_via_backend",
        "policy_path": path_for_record(policy_path, microduck_rl_root, PROJECT_ROOT),
        "policy_sha256": _sha256(policy_path),
        "microduck_rl_commit": _official_commit(microduck_rl_root),
        "robustness_config_path": path_for_record(
            robustness_protocol.source_path, PROJECT_ROOT
        ),
        "goal_set_path": path_for_record(goal_protocol.source_path, PROJECT_ROOT),
        "run_signature": signature,
        "expected_io": {
            "observation_size": EXPECTED_OBSERVATION_SIZE,
            "action_size": EXPECTED_ACTION_SIZE,
        },
        "navigator_action_scope": "[vx, 0, wz]",
        "conditions": condition_documents,
    }
    selected_ids = [
        item["condition"]["condition_id"] for item in condition_documents
    ]
    planned_ids = (
        list(robustness_protocol.quick_condition_ids)
        if mode == "quick"
        else list(robustness_protocol.full_condition_ids)
        if mode == "full"
        else []
    )
    if mode == "smoke":
        run_label = f"smoke_{selected_ids[0]}"
    elif selected_ids == planned_ids:
        run_label = mode
    else:
        short_selection = "_".join(selected_ids)
        run_label = f"{mode}_partial_{short_selection}"
    path = summary_dir / f"pointgoal_robustness_{run_label}_{policy_path.stem}.json"
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, smoke-test, quick-screen, or run the full PointGoal OOD matrix."
    )
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the protocol-specific output directory.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--smoke-condition", default="nominal")
    parser.add_argument("--smoke-goal", default="front")
    parser.add_argument(
        "--condition",
        action="append",
        dest="condition_ids",
        help="Run only this planned condition; repeat to select several.",
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
    robustness_protocol = load_robustness_protocol(config_path)
    default_output_dir = (
        DEFAULT_OUTPUT_DIR
        if robustness_protocol.output_subdir is None
        else DEFAULT_OUTPUT_DIR / robustness_protocol.output_subdir
    )
    output_dir = _resolved(args.output_dir or default_output_dir)
    goal_protocol = load_goal_set(robustness_protocol.goal_set_path)
    if not policy_path.is_file():
        parser.error(f"ONNX policy not found: {policy_path}")

    print("MicroDuck PointGoal robustness contract is valid")
    print(f"  protocol status:  {robustness_protocol.protocol_status}")
    print(f"  policy:           {policy_path}")
    print(f"  goal set:         {goal_protocol.source_path}")
    print(
        f"  quick/full:       "
        f"{len(robustness_protocol.quick_condition_ids) * len(goal_protocol.goals) * robustness_protocol.quick_episodes_per_goal}/"
        f"{len(robustness_protocol.full_condition_ids) * len(goal_protocol.goals) * robustness_protocol.full_episodes_per_goal} episodes"
    )
    if args.validate_only or not (args.smoke or args.quick or args.full):
        return 0
    if (args.quick or args.full) and robustness_protocol.protocol_status != "frozen":
        parser.error("quick/full runs require protocol_status='frozen'")

    official_commit = _official_commit(microduck_rl_root)
    if args.smoke:
        mode = "smoke"
        try:
            condition = robustness_protocol.condition_map[args.smoke_condition]
        except KeyError:
            parser.error(f"unknown smoke condition: {args.smoke_condition!r}")
        selected = (condition,)
        run_protocol = protocol_for_mode(goal_protocol, 1)
        smoke_goal = args.smoke_goal
    else:
        mode = "quick" if args.quick else "full"
        try:
            selected = select_conditions(
                robustness_protocol, mode, args.condition_ids
            )
        except ValueError as error:
            parser.error(str(error))
        episodes_per_goal = (
            robustness_protocol.quick_episodes_per_goal
            if mode == "quick"
            else robustness_protocol.full_episodes_per_goal
        )
        run_protocol = protocol_for_mode(goal_protocol, episodes_per_goal)
        smoke_goal = None

    signature = _run_signature(
        policy_path,
        robustness_protocol,
        goal_protocol,
        official_commit,
        mode,
    )
    condition_documents: list[dict[str, Any]] = []
    for condition in selected:
        aggregate, details = run_condition(
            condition=condition,
            goal_protocol=run_protocol,
            microduck_rl_root=microduck_rl_root,
            policy_path=policy_path,
            output_dir=output_dir,
            mode=mode,
            signature=signature,
            resume=not args.no_resume,
            smoke_goal=smoke_goal,
        )
        condition_documents.append(
            {
                "condition": asdict(condition),
                "runtime": details["runtime"],
                "aggregate": aggregate,
                "episodes": details["episodes"],
            }
        )
    summary = write_suite_summary(
        robustness_protocol=robustness_protocol,
        goal_protocol=run_protocol,
        policy_path=policy_path,
        microduck_rl_root=microduck_rl_root,
        output_dir=output_dir,
        mode=mode,
        signature=signature,
        condition_documents=condition_documents,
    )
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
