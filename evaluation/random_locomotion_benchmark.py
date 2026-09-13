#!/usr/bin/env python3
"""Run a resumable, deterministic random-command locomotion benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.locomotion_benchmark import (
    BAM_VIN,
    BAM_VIN_DROP_GAIN,
    DECIMATION,
    EXPECTED_ACTION_SIZE,
    EXPECTED_OBSERVATION_SIZE,
    FALL_TILT_DEG,
    OFFICIAL_RESET_RANGES,
    PHYSICS_TIMESTEP_S,
    PROJECT_ROOT,
    BenchmarkConfig,
    CommandCase,
    run_episode,
    validate_artifacts,
)


DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "week02" / "05_random_400" / "run"
COMMAND_RANGES = {
    "vx": (-0.4, 0.4),
    "vy": (-0.3, 0.3),
    "wz": (-1.0, 1.0),
}
STANDING_FRACTION = 0.25
TURN_IN_PLACE_FRACTION = 0.15
TURN_MAGNITUDE_RANGE = (0.4, 1.0)


@dataclass(frozen=True)
class RandomCommandSample:
    """One command drawn from the documented final training distribution."""

    episode: int
    bucket: str
    vx: float
    vy: float
    wz: float


def _bucket_counts(episodes: int) -> dict[str, int]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    standing = round(episodes * STANDING_FRACTION)
    turn = round(episodes * TURN_IN_PLACE_FRACTION)
    return {
        "standing": standing,
        "turn_in_place": turn,
        "general": episodes - standing - turn,
    }


def build_command_plan(episodes: int, command_seed: int) -> list[RandomCommandSample]:
    """Build a fixed random plan with exact standing and turn-in-place quotas."""

    counts = _bucket_counts(episodes)
    rng = random.Random(command_seed)
    commands: list[tuple[str, float, float, float]] = []

    commands.extend(("standing", 0.0, 0.0, 0.0) for _ in range(counts["standing"]))

    turn_signs = [
        -1.0 if index % 2 == 0 else 1.0
        for index in range(counts["turn_in_place"])
    ]
    rng.shuffle(turn_signs)
    for sign in turn_signs:
        magnitude = rng.uniform(*TURN_MAGNITUDE_RANGE)
        commands.append(("turn_in_place", 0.0, 0.0, sign * magnitude))

    for _ in range(counts["general"]):
        commands.append(
            (
                "general",
                rng.uniform(*COMMAND_RANGES["vx"]),
                rng.uniform(*COMMAND_RANGES["vy"]),
                rng.uniform(*COMMAND_RANGES["wz"]),
            )
        )

    rng.shuffle(commands)
    return [
        RandomCommandSample(index, bucket, vx, vy, wz)
        for index, (bucket, vx, vy, wz) in enumerate(commands)
    ]


def _plan_document(
    plan: list[RandomCommandSample],
    *,
    reset_seed: int,
    command_seed: int,
    warmup_s: float,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reset_seed": reset_seed,
        "command_seed": command_seed,
        "warmup_s": warmup_s,
        "duration_s": duration_s,
        "initial_state_mode": "official_reset",
        "command_ranges": COMMAND_RANGES,
        "standing_fraction": STANDING_FRACTION,
        "turn_in_place_fraction": TURN_IN_PLACE_FRACTION,
        "turn_magnitude_range": TURN_MAGNITUDE_RANGE,
        "bucket_counts": _bucket_counts(len(plan)),
        "commands": [asdict(command) for command in plan],
    }


def _json_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_or_create_plan(path: Path, document: dict[str, Any]) -> str:
    plan_hash = _json_hash(document)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _json_hash(existing) != plan_hash:
            raise ValueError(
                f"existing plan differs from requested protocol: {path}; "
                "use a different output directory"
            )
    else:
        _write_json(path, document)
    return plan_hash


def _episode_record(
    sample: RandomCommandSample,
    episode_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode": sample.episode,
        "bucket": sample.bucket,
        "command": {"vx": sample.vx, "vy": sample.vy, "wz": sample.wz},
        **episode_summary,
    }


def run_random_benchmark(
    config: BenchmarkConfig,
    *,
    episodes: int,
    command_seed: int,
    warmup_s: float,
    duration_s: float,
) -> Path:
    plan = build_command_plan(episodes, command_seed)
    plan_document = _plan_document(
        plan,
        reset_seed=config.seed,
        command_seed=command_seed,
        warmup_s=warmup_s,
        duration_s=duration_s,
    )
    raw_dir = config.output_dir / "raw"
    progress_dir = raw_dir / "episode_summaries"
    plan_hash = _load_or_create_plan(raw_dir / "random_command_plan.json", plan_document)

    records: list[dict[str, Any]] = []
    for sample in plan:
        progress_path = progress_dir / f"episode_{sample.episode:04d}.json"
        if progress_path.exists():
            record = json.loads(progress_path.read_text(encoding="utf-8"))
            if record.get("command") != {
                "vx": sample.vx,
                "vy": sample.vy,
                "wz": sample.wz,
            } or record.get("bucket") != sample.bucket:
                raise ValueError(f"resume record does not match plan: {progress_path}")
            raw_csv = PROJECT_ROOT / record["raw_steps_csv"]
            if not raw_csv.is_file():
                raise FileNotFoundError(f"resume CSV is missing: {raw_csv}")
            records.append(record)
            print(f"episode {sample.episode:03d}: resumed", flush=True)
            continue

        command = CommandCase(
            name=f"random_velocity_{episodes}_{sample.episode:04d}_{sample.bucket}",
            vx=sample.vx,
            vy=sample.vy,
            wz=sample.wz,
            warmup_s=warmup_s,
            duration_s=duration_s,
            episodes=1,
            initial_state_mode="official_reset",
        )
        episode_summary = run_episode(config, command, sample.episode)
        record = _episode_record(sample, episode_summary)
        _write_json(progress_path, record)
        records.append(record)
        print(
            f"episode {sample.episode:03d}/{episodes - 1}: {sample.bucket}, "
            f"cmd=({sample.vx:+.3f}, {sample.vy:+.3f}, {sample.wz:+.3f}), "
            f"actual=({record['mean_actual_vx']:+.3f}, "
            f"{record['mean_actual_vy']:+.3f}, {record['mean_actual_wz']:+.3f}), "
            f"fallen={record['fallen']}",
            flush=True,
        )

    official_commit = subprocess.run(
        ["git", "-C", str(config.microduck_rl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": f"onnx_cpu_bam_random_velocity_{episodes}",
        "reward_available": False,
        "microduck_rl_commit": official_commit,
        "policy_path": str(config.policy_path.relative_to(config.microduck_rl_root)),
        "policy_sha256": hashlib.sha256(config.policy_path.read_bytes()).hexdigest(),
        "plan_sha256": plan_hash,
        "protocol": {key: value for key, value in plan_document.items() if key != "commands"},
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
            "official_reset_ranges": OFFICIAL_RESET_RANGES,
        },
        "episode_count": len(records),
        "episodes": records,
    }
    summary_path = (
        config.output_dir
        / "summary"
        / f"random_velocity_{episodes}_{config.policy_path.stem}.json"
    )
    _write_json(summary_path, summary)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable random-command MicroDuck ONNX benchmark."
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42, help="First official-reset seed")
    parser.add_argument("--command-seed", type=int, default=20260911)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.warmup_s) or args.warmup_s < 0:
        raise ValueError("warmup-s must be finite and >= 0")
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError("duration-s must be finite and > 0")

    config = BenchmarkConfig(
        microduck_rl_root=args.microduck_rl_root.expanduser().resolve(),
        policy_path=args.policy.expanduser().resolve(),
        command_set_path=Path(__file__).resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        seed=args.seed,
    )
    validate_artifacts(config)
    plan = build_command_plan(args.episodes, args.command_seed)
    counts = _bucket_counts(args.episodes)
    print("MicroDuck random locomotion benchmark contract is valid")
    print(f"  policy:          {config.policy_path}")
    print(f"  output:          {config.output_dir}")
    print(f"  episodes:        {len(plan)}")
    print(f"  reset seeds:     {args.seed}..{args.seed + args.episodes - 1}")
    print(f"  command seed:    {args.command_seed}")
    print(f"  buckets:         {counts}")
    print(f"  command ranges:  {COMMAND_RANGES}")
    if args.validate_only:
        return 0

    summary_path = run_random_benchmark(
        config,
        episodes=args.episodes,
        command_seed=args.command_seed,
        warmup_s=args.warmup_s,
        duration_s=args.duration_s,
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
