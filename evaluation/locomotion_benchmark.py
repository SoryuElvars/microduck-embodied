#!/usr/bin/env python3
"""Headless locomotion benchmark entry point.

Phase 0 of this module validates the benchmark contract and its external
artifacts.  The MuJoCo/BAM stepping loop will be added behind ``run_episode``
after this skeleton is verified against the known straight-line failure case.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


EXPECTED_OBSERVATION_SIZE = 61
EXPECTED_ACTION_SIZE = 14
DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_COMMAND_SET = Path(__file__).parent / "command_sets" / "straight_only.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "results" / "week02"


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


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved paths and reproducibility controls for one benchmark run."""

    microduck_rl_root: Path
    policy_path: Path
    command_set_path: Path
    output_dir: Path
    seed: int


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


def run_episode(config: BenchmarkConfig, command: CommandCase, episode: int) -> None:
    """Run one headless MuJoCo/BAM episode.

    Next implementation step:
      1. import the official inference helpers;
      2. create and reset MuJoCo, BAM and PolicyInference state;
      3. run a 50 Hz control loop without a viewer or wall-clock sleep;
      4. return per-step samples for metric aggregation.
    """

    raise NotImplementedError(
        "Phase 0 only validates the benchmark contract; "
        "the headless episode loop is the next implementation gate."
    )


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
    for case in commands:
        print(
            f"    - {case.name}: cmd=({case.vx:+.2f}, {case.vy:+.2f}, "
            f"{case.wz:+.2f}), warmup={case.warmup_s:.1f}s, "
            f"duration={case.duration_s:.1f}s, episodes={case.episodes}"
        )

    if args.validate_only:
        return 0

    for case in commands:
        for episode in range(case.episodes):
            run_episode(config, case, episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
