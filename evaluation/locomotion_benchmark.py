#!/usr/bin/env python3
"""Headless locomotion benchmark entry point.

Phase 0 of this module validates the benchmark contract and its external
artifacts.  The MuJoCo/BAM stepping loop will be added behind ``run_episode``
after this skeleton is verified against the known straight-line failure case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Sequence


EXPECTED_OBSERVATION_SIZE = 61
EXPECTED_ACTION_SIZE = 14
PHYSICS_TIMESTEP_S = 0.005
DECIMATION = 4
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
FALL_TILT_DEG = 70.0
INITIAL_STATE_MODES = {"fixed", "official_reset"}
OFFICIAL_RESET_RANGES = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (0.12, 0.13),
    "yaw": (-math.pi, math.pi),
}
PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_COMMAND_SET = Path(__file__).parent / "command_sets" / "straight_only.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "week02"


@dataclass(frozen=True)
class CommandCase:
    """One reproducible velocity-command evaluation case."""

    name: str
    vx: float
    vy: float
    wz: float
    warmup_s: float
    duration_s: float
    episodes: int
    initial_state_mode: str

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        defaults: dict[str, Any],
    ) -> "CommandCase":
        def value(key: str) -> Any:
            if key in raw:
                return raw[key]
            if key in defaults:
                return defaults[key]
            raise ValueError(f"command {raw.get('name', '<unnamed>')!r} is missing {key!r}")

        case = cls(
            name=str(value("name")),
            vx=float(value("vx")),
            vy=float(value("vy")),
            wz=float(value("wz")),
            warmup_s=float(value("warmup_s")),
            duration_s=float(value("duration_s")),
            episodes=int(value("episodes")),
            initial_state_mode=str(value("initial_state_mode")),
        )
        case.validate()
        return case

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("command name must not be empty")
        if not all(math.isfinite(v) for v in (self.vx, self.vy, self.wz)):
            raise ValueError(f"command {self.name!r} contains a non-finite velocity")
        if not math.isfinite(self.warmup_s) or self.warmup_s < 0:
            raise ValueError(f"command {self.name!r} warmup_s must be finite and >= 0")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError(f"command {self.name!r} duration_s must be finite and > 0")
        if self.episodes < 1:
            raise ValueError(f"command {self.name!r} episodes must be >= 1")
        if self.initial_state_mode not in INITIAL_STATE_MODES:
            allowed = ", ".join(sorted(INITIAL_STATE_MODES))
            raise ValueError(
                f"command {self.name!r} initial_state_mode must be one of: {allowed}"
            )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved paths and reproducibility controls for one benchmark run."""

    microduck_rl_root: Path
    policy_path: Path
    command_set_path: Path
    output_dir: Path
    seed: int


@dataclass
class Runtime:
    """Official deployment-rehearsal components for one fresh episode."""

    official: Any
    model: Any
    data: Any
    bam_ctrl: Any
    policy: Any
    qpos_adr: int
    qvel_adr: int
    control_dt: float


def load_command_set(path: Path) -> list[CommandCase]:
    """Load and validate a versioned JSON command set."""

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    if document.get("schema_version") != 1:
        raise ValueError("command set schema_version must be 1")

    defaults = document.get("defaults", {})
    raw_commands = document.get("commands")
    if not isinstance(defaults, dict):
        raise ValueError("command set defaults must be an object")
    if not isinstance(raw_commands, list) or not raw_commands:
        raise ValueError("command set commands must be a non-empty list")

    commands = [CommandCase.from_mapping(raw, defaults) for raw in raw_commands]
    names = [case.name for case in commands]
    if len(names) != len(set(names)):
        raise ValueError("command names must be unique")
    return commands


def validate_artifacts(config: BenchmarkConfig) -> None:
    """Fail early when the official checkout or exported policy is missing."""

    required_official_files = (
        config.microduck_rl_root / "scripts" / "infer_policy.py",
        config.microduck_rl_root
        / "src"
        / "mjlab_microduck"
        / "robot"
        / "microduck"
        / "scene.xml",
    )
    for path in required_official_files:
        if not path.is_file():
            raise FileNotFoundError(f"required official artifact not found: {path}")

    if not config.policy_path.is_file():
        raise FileNotFoundError(f"ONNX policy not found: {config.policy_path}")
    if config.policy_path.suffix.lower() != ".onnx":
        raise ValueError(f"policy must be an .onnx file: {config.policy_path}")


def load_official_inference_module(microduck_rl_root: Path) -> Any:
    """Load the upstream inference script without copying its implementation."""

    module_name = "_microduck_official_infer_policy"
    if module_name in sys.modules:
        return sys.modules[module_name]

    script_path = microduck_rl_root / "scripts" / "infer_policy.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official inference module: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sample_initial_state(mode: str, episode_seed: int) -> dict[str, float]:
    """Sample a documented reset pose without inventing joint perturbations."""

    if mode == "fixed":
        return {"x": 0.0, "y": 0.0, "z": 0.125, "yaw": 0.0}
    if mode != "official_reset":
        raise ValueError(f"unknown initial state mode: {mode}")

    rng = random.Random(episode_seed)
    return {
        name: rng.uniform(lower, upper)
        for name, (lower, upper) in OFFICIAL_RESET_RANGES.items()
    }


def create_runtime(
    config: BenchmarkConfig,
    episode_seed: int,
    initial_state_mode: str,
) -> tuple[Runtime, dict[str, float]]:
    """Create a fresh nominal CPU MuJoCo + BAM + ONNX runtime."""

    official = load_official_inference_module(config.microduck_rl_root)
    np = official.np
    mujoco = official.mujoco

    random.seed(episode_seed)
    np.random.seed(episode_seed)

    xml_path = config.microduck_rl_root / official.MICRODUCK_XML
    bam_model = official.load_bam_model(
        official.BAM_KP_FW,
        BAM_VIN,
        0.0,
    )
    model, data, bam_ctrl, _ = official.load_mujoco_with_bam(
        str(xml_path),
        bam_model,
        PHYSICS_TIMESTEP_S,
        BAM_VIN_DROP_GAIN,
        official.BAM_VIN_MIN,
    )
    policy = official.PolicyInference(
        model,
        data,
        walking_onnx_path=str(config.policy_path),
        action_scale=1.0,
        bam_ctrl=bam_ctrl,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )

    freejoint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "trunk_base_freejoint",
    )
    if freejoint_id < 0:
        raise ValueError("official scene has no trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    qvel_adr = int(model.jnt_dofadr[freejoint_id])

    initial_state = sample_initial_state(initial_state_mode, episode_seed)
    half_yaw = initial_state["yaw"] / 2.0
    data.qpos[qpos_adr : qpos_adr + 3] = (
        initial_state["x"],
        initial_state["y"],
        initial_state["z"],
    )
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    for index, joint_qpos_index in enumerate(policy.joint_qpos_indices):
        data.qpos[joint_qpos_index] = policy.default_pose[index]
    bam_ctrl.reset(data.qpos)
    policy.last_action[:] = 0.0
    policy.set_position_targets(policy.default_pose)
    policy.set_vel_cmd(0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    observation = policy.get_observations()
    if observation.size != EXPECTED_OBSERVATION_SIZE:
        raise ValueError(
            f"unexpected observation size: {observation.size}; "
            f"expected {EXPECTED_OBSERVATION_SIZE}"
        )
    output_shape = policy.walking_session.get_outputs()[0].shape
    if output_shape[-1] != EXPECTED_ACTION_SIZE:
        raise ValueError(
            f"unexpected action size: {output_shape}; expected [1, {EXPECTED_ACTION_SIZE}]"
        )

    return (
        Runtime(
            official=official,
            model=model,
            data=data,
            bam_ctrl=bam_ctrl,
            policy=policy,
            qpos_adr=qpos_adr,
            qvel_adr=qvel_adr,
            control_dt=DECIMATION * model.opt.timestep,
        ),
        initial_state,
    )


def quaternion_to_euler(quaternion: Sequence[float]) -> tuple[float, float, float]:
    """Convert a MuJoCo ``[w, x, y, z]`` quaternion to roll, pitch and yaw."""

    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sine)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrapped_angle_delta(current: float, previous: float) -> float:
    """Return the shortest signed angular displacement between two samples."""

    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def _base_state(runtime: Runtime) -> dict[str, Any]:
    np = runtime.official.np
    data = runtime.data
    policy = runtime.policy
    qpos_adr = runtime.qpos_adr
    qvel_adr = runtime.qvel_adr

    position = data.qpos[qpos_adr : qpos_adr + 3].copy()
    quaternion = data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
    roll, pitch, yaw = quaternion_to_euler(quaternion)
    world_velocity = np.asarray(data.qvel[qvel_adr : qvel_adr + 3], dtype=np.float32)
    body_velocity = policy.quat_rotate_inverse(
        np.asarray(quaternion, dtype=np.float32),
        world_velocity,
    )
    projected_gravity = policy.get_projected_gravity()
    tilt = math.acos(max(-1.0, min(1.0, -float(projected_gravity[2]))))
    return {
        "position": position,
        "quaternion": quaternion,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "tilt": tilt,
        "body_velocity": body_velocity,
        "yaw_rate": float(data.qvel[qvel_adr + 5]),
    }


def _advance_control_step(runtime: Runtime) -> tuple[Any, Any, dict[str, Any]]:
    observation = runtime.policy.get_observations().copy()
    action = runtime.policy.infer()
    runtime.policy.apply_action(action)
    for _ in range(DECIMATION):
        runtime.bam_ctrl.update()
        runtime.official.mujoco.mj_step(runtime.model, runtime.data)
    return observation, action, _base_state(runtime)


def _write_step_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rmse(values: Sequence[float], target: float) -> float:
    return math.sqrt(fmean((value - target) ** 2 for value in values))


def run_episode(
    config: BenchmarkConfig,
    command: CommandCase,
    episode: int,
) -> dict[str, Any]:
    """Run one deterministic, headless MuJoCo/BAM deployment episode.

    This loop intentionally does not calculate the training reward.  It tests
    the exported actor in the same nominal CPU/BAM deployment rehearsal used by
    the upstream interactive inference script.
    """

    episode_seed = config.seed + episode
    runtime, initial_state = create_runtime(
        config,
        episode_seed,
        command.initial_state_mode,
    )
    np = runtime.official.np
    rows: list[dict[str, Any]] = []

    runtime.policy.set_vel_cmd(0.0, 0.0, 0.0)
    warmup_steps = round(command.warmup_s / runtime.control_dt)
    for step in range(warmup_steps):
        observation, action, state = _advance_control_step(runtime)
        row: dict[str, Any] = {
            "phase": "warmup",
            "step": step,
            "time_s": (step + 1) * runtime.control_dt,
            "cmd_vx": 0.0,
            "cmd_vy": 0.0,
            "cmd_wz": 0.0,
            "actual_vx": float(state["body_velocity"][0]),
            "actual_vy": float(state["body_velocity"][1]),
            "actual_wz": state["yaw_rate"],
            "world_x": float(state["position"][0]),
            "world_y": float(state["position"][1]),
            "trunk_z": float(state["position"][2]),
            "roll_rad": state["roll"],
            "pitch_rad": state["pitch"],
            "yaw_rad": state["yaw"],
            "tilt_rad": state["tilt"],
            "net_yaw_rad": 0.0,
            "forward_distance_m": 0.0,
            "lateral_displacement_m": 0.0,
            "fallen": state["tilt"] >= math.radians(FALL_TILT_DEG),
        }
        row.update({f"obs_{index:02d}": float(value) for index, value in enumerate(observation)})
        row.update({f"action_{index:02d}": float(value) for index, value in enumerate(action)})
        rows.append(row)

    start_state = _base_state(runtime)
    start_position = start_state["position"].copy()
    start_yaw = start_state["yaw"]
    previous_yaw = start_yaw
    net_yaw = 0.0
    fall_time_s: float | None = None

    runtime.policy.set_vel_cmd(command.vx, command.vy, command.wz)
    test_steps = round(command.duration_s / runtime.control_dt)
    for step in range(test_steps):
        observation, action, state = _advance_control_step(runtime)
        net_yaw += wrapped_angle_delta(state["yaw"], previous_yaw)
        previous_yaw = state["yaw"]

        displacement = state["position"][:2] - start_position[:2]
        forward_distance = (
            math.cos(start_yaw) * float(displacement[0])
            + math.sin(start_yaw) * float(displacement[1])
        )
        lateral_displacement = (
            -math.sin(start_yaw) * float(displacement[0])
            + math.cos(start_yaw) * float(displacement[1])
        )
        finite = bool(
            np.isfinite(observation).all()
            and np.isfinite(action).all()
            and np.isfinite(runtime.data.qpos).all()
            and np.isfinite(runtime.data.qvel).all()
        )
        fallen = not finite or state["tilt"] >= math.radians(FALL_TILT_DEG)
        time_s = (step + 1) * runtime.control_dt
        if fallen and fall_time_s is None:
            fall_time_s = time_s

        row = {
            "phase": "test",
            "step": step,
            "time_s": time_s,
            "cmd_vx": command.vx,
            "cmd_vy": command.vy,
            "cmd_wz": command.wz,
            "actual_vx": float(state["body_velocity"][0]),
            "actual_vy": float(state["body_velocity"][1]),
            "actual_wz": state["yaw_rate"],
            "world_x": float(state["position"][0]),
            "world_y": float(state["position"][1]),
            "trunk_z": float(state["position"][2]),
            "roll_rad": state["roll"],
            "pitch_rad": state["pitch"],
            "yaw_rad": state["yaw"],
            "tilt_rad": state["tilt"],
            "net_yaw_rad": net_yaw,
            "forward_distance_m": forward_distance,
            "lateral_displacement_m": lateral_displacement,
            "fallen": fallen,
        }
        row.update({f"obs_{index:02d}": float(value) for index, value in enumerate(observation)})
        row.update({f"action_{index:02d}": float(value) for index, value in enumerate(action)})
        rows.append(row)
        if fallen:
            break

    raw_path = (
        config.output_dir
        / "raw"
        / f"{command.name}_seed{episode_seed}_episode{episode:03d}_steps.csv"
    )
    _write_step_csv(raw_path, rows)

    test_rows = [row for row in rows if row["phase"] == "test"]
    final = test_rows[-1]
    summary = {
        "episode": episode,
        "seed": episode_seed,
        "initial_state": initial_state,
        "steps_completed": len(test_rows),
        "duration_completed_s": len(test_rows) * runtime.control_dt,
        "rmse_vx": _rmse([row["actual_vx"] for row in test_rows], command.vx),
        "rmse_vy": _rmse([row["actual_vy"] for row in test_rows], command.vy),
        "rmse_wz": _rmse([row["actual_wz"] for row in test_rows], command.wz),
        "mean_actual_vx": fmean(row["actual_vx"] for row in test_rows),
        "mean_actual_vy": fmean(row["actual_vy"] for row in test_rows),
        "mean_actual_wz": fmean(row["actual_wz"] for row in test_rows),
        "net_yaw_rad": final["net_yaw_rad"],
        "net_yaw_deg": math.degrees(final["net_yaw_rad"]),
        "forward_distance_m": final["forward_distance_m"],
        "lateral_displacement_m": final["lateral_displacement_m"],
        "fallen": fall_time_s is not None,
        "fall_time_s": fall_time_s,
        "raw_steps_csv": str(raw_path.relative_to(PROJECT_ROOT)),
    }
    return summary


def describe(values: Sequence[float]) -> dict[str, float]:
    """Return compact population statistics for a completed episode batch."""

    return {
        "mean": fmean(values),
        "median": median(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize direction consistency and the main tracking metrics."""

    yaw_values = [episode["net_yaw_deg"] for episode in episodes]
    near_zero_deg = 1.0
    return {
        "episode_count": len(episodes),
        "fall_rate": fmean(float(episode["fallen"]) for episode in episodes),
        "yaw_direction_counts": {
            "positive": sum(value > near_zero_deg for value in yaw_values),
            "near_zero": sum(abs(value) <= near_zero_deg for value in yaw_values),
            "negative": sum(value < -near_zero_deg for value in yaw_values),
            "near_zero_threshold_deg": near_zero_deg,
        },
        "net_yaw_deg": describe(yaw_values),
        "mean_actual_vx": describe([episode["mean_actual_vx"] for episode in episodes]),
        "mean_actual_vy": describe([episode["mean_actual_vy"] for episode in episodes]),
        "mean_actual_wz": describe([episode["mean_actual_wz"] for episode in episodes]),
        "rmse_vx": describe([episode["rmse_vx"] for episode in episodes]),
        "rmse_vy": describe([episode["rmse_vy"] for episode in episodes]),
        "rmse_wz": describe([episode["rmse_wz"] for episode in episodes]),
        "forward_distance_m": describe(
            [episode["forward_distance_m"] for episode in episodes]
        ),
        "lateral_displacement_m": describe(
            [episode["lateral_displacement_m"] for episode in episodes]
        ),
    }


def write_summary(
    config: BenchmarkConfig,
    command: CommandCase,
    episodes: list[dict[str, Any]],
) -> Path:
    """Write a reviewable result artifact separate from ignored raw samples."""

    summary_dir = config.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{command.name}_{config.policy_path.stem}.json"
    official_commit = subprocess.run(
        ["git", "-C", str(config.microduck_rl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    policy_hash = hashlib.sha256(config.policy_path.read_bytes()).hexdigest()
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "onnx_cpu_bam_deployment_rehearsal",
        "reward_available": False,
        "microduck_rl_commit": official_commit,
        "policy_path": str(config.policy_path.relative_to(config.microduck_rl_root)),
        "policy_sha256": policy_hash,
        "command_set_path": str(config.command_set_path.relative_to(PROJECT_ROOT)),
        "runtime": {
            "observation_size": EXPECTED_OBSERVATION_SIZE,
            "action_size": EXPECTED_ACTION_SIZE,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "decimation": DECIMATION,
            "control_frequency_hz": 1.0 / (PHYSICS_TIMESTEP_S * DECIMATION),
            "actuator": "BAM M6 XL330",
            "bam_vin": BAM_VIN,
            "bam_vin_drop_gain": BAM_VIN_DROP_GAIN,
            "fall_proxy_tilt_deg": FALL_TILT_DEG,
        },
        "command": {
            "name": command.name,
            "vx": command.vx,
            "vy": command.vy,
            "wz": command.wz,
            "warmup_s": command.warmup_s,
            "duration_s": command.duration_s,
            "episodes": command.episodes,
            "initial_state_mode": command.initial_state_mode,
            "official_reset_ranges": (
                OFFICIAL_RESET_RANGES if command.initial_state_mode == "official_reset" else None
            ),
        },
        "aggregate": aggregate_episodes(episodes),
        "episodes": episodes,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_suite_summary(
    config: BenchmarkConfig,
    results: list[tuple[CommandCase, list[dict[str, Any]], Path]],
) -> Path:
    """Write one compact index for comparing every command in a command set."""

    summary_dir = config.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{config.command_set_path.stem}_{config.policy_path.stem}.json"
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "onnx_cpu_bam_command_response_suite",
        "reward_available": False,
        "command_set_path": str(config.command_set_path.relative_to(PROJECT_ROOT)),
        "policy_path": str(config.policy_path.relative_to(config.microduck_rl_root)),
        "policy_sha256": hashlib.sha256(config.policy_path.read_bytes()).hexdigest(),
        "seed": config.seed,
        "command_count": len(results),
        "total_episodes": sum(len(episodes) for _, episodes, _ in results),
        "paired_episode_seeds": True,
        "commands": [
            {
                "name": command.name,
                "vx": command.vx,
                "vy": command.vy,
                "wz": command.wz,
                "warmup_s": command.warmup_s,
                "duration_s": command.duration_s,
                "episodes": command.episodes,
                "initial_state_mode": command.initial_state_mode,
                "summary_json": str(summary_path.relative_to(PROJECT_ROOT)),
                "aggregate": aggregate_episodes(episodes),
            }
            for command, episodes, summary_path in results
        ],
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or run the MicroDuck ONNX locomotion benchmark."
    )
    parser.add_argument(
        "--microduck-rl-root",
        type=Path,
        default=DEFAULT_MICRODUCK_RL_ROOT,
        help="Official microduck_rl checkout (default: %(default)s)",
    )
    parser.add_argument("--policy", type=Path, required=True, help="Exported walking ONNX")
    parser.add_argument(
        "--command-set",
        type=Path,
        default=DEFAULT_COMMAND_SET,
        help="Versioned JSON command set (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Benchmark result root (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths and command definitions without starting MuJoCo",
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        microduck_rl_root=_resolved(args.microduck_rl_root),
        policy_path=_resolved(args.policy),
        command_set_path=_resolved(args.command_set),
        output_dir=_resolved(args.output_dir),
        seed=args.seed,
    )

    commands = load_command_set(config.command_set_path)
    validate_artifacts(config)

    print("MicroDuck locomotion benchmark contract is valid")
    print(f"  official checkout: {config.microduck_rl_root}")
    print(f"  policy:            {config.policy_path}")
    print(f"  command set:        {config.command_set_path}")
    print(f"  output directory:   {config.output_dir}")
    print(f"  seed:               {config.seed}")
    print(f"  expected I/O:       {EXPECTED_OBSERVATION_SIZE} obs -> {EXPECTED_ACTION_SIZE} actions")
    print("  planned cases:")
    suite_results = []
    for case in commands:
        print(
            f"    - {case.name}: cmd=({case.vx:+.2f}, {case.vy:+.2f}, "
            f"{case.wz:+.2f}), warmup={case.warmup_s:.1f}s, "
            f"duration={case.duration_s:.1f}s, episodes={case.episodes}, "
            f"initial_state={case.initial_state_mode}"
        )

    if args.validate_only:
        return 0

    for case in commands:
        episode_summaries = []
        for episode in range(case.episodes):
            episode_summary = run_episode(config, case, episode)
            episode_summaries.append(episode_summary)
            print(
                f"episode {episode}: yaw={episode_summary['net_yaw_deg']:+.1f} deg, "
                f"mean_vx={episode_summary['mean_actual_vx']:+.3f} m/s, "
                f"mean_wz={episode_summary['mean_actual_wz']:+.3f} rad/s, "
                f"fallen={episode_summary['fallen']}"
            )
        summary_path = write_summary(config, case, episode_summaries)
        suite_results.append((case, episode_summaries, summary_path))
        print(f"summary: {summary_path}")
    if len(suite_results) > 1:
        suite_summary_path = write_suite_summary(config, suite_results)
        print(f"suite summary: {suite_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
