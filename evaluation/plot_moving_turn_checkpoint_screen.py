#!/usr/bin/env python3
"""Aggregate and plot the PointGoal moving-turn checkpoint quick screen."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
COMMAND_KEYS = {
    (0.25, 0.0, 0.0): "straight",
    (0.25, 0.0, 0.25): "gentle_left",
    (0.25, 0.0, -0.25): "gentle_right",
    (0.2, 0.0, 0.5): "turn_left",
    (0.2, 0.0, -0.5): "turn_right",
}
TURN_PAIRS = (
    ("gentle_left", "gentle_right", "gentle"),
    ("turn_left", "turn_right", "normal"),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_from_policy_path(policy_path: str) -> int:
    match = re.fullmatch(r"model_(\d+)", Path(policy_path).stem)
    if match is None:
        raise ValueError(f"cannot infer checkpoint from policy path: {policy_path}")
    return int(match.group(1))


def command_key(command: dict[str, Any]) -> str:
    signature = tuple(float(command[name]) for name in ("vx", "vy", "wz"))
    try:
        return COMMAND_KEYS[signature]
    except KeyError as error:
        raise ValueError(f"unexpected moving-turn command: {signature}") from error


def describe(values: Sequence[float]) -> dict[str, Any]:
    return {
        "values": list(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def load_checkpoint(suite_path: Path) -> dict[str, Any]:
    suite = load_json(suite_path)
    if suite.get("command_count") != 5 or suite.get("total_episodes") != 25:
        raise ValueError(f"{suite_path} is not a 5-command x 5-episode screen")

    summaries: dict[str, dict[str, Any]] = {}
    for item in suite["commands"]:
        summary_path = resolve_project_path(item["summary_json"])
        summary = load_json(summary_path)
        key = command_key(summary["command"])
        if key in summaries:
            raise ValueError(f"duplicate {key} command in {suite_path}")
        if len(summary["episodes"]) != 5:
            raise ValueError(f"{summary_path} does not contain five episodes")
        summaries[key] = summary
    if set(summaries) != set(COMMAND_KEYS.values()):
        raise ValueError(f"missing moving-turn command in {suite_path}")

    seed_sets = [
        tuple(int(episode["seed"]) for episode in summary["episodes"])
        for summary in summaries.values()
    ]
    if len(set(seed_sets)) != 1:
        raise ValueError(f"commands in {suite_path} do not use paired seeds")

    commands: dict[str, Any] = {}
    for key, summary in summaries.items():
        episodes = summary["episodes"]
        commands[key] = {
            "command": summary["command"],
            "mean_actual_vx": describe(
                [float(episode["mean_actual_vx"]) for episode in episodes]
            ),
            "mean_actual_wz": describe(
                [float(episode["mean_actual_wz"]) for episode in episodes]
            ),
            "net_yaw_deg": describe(
                [float(episode["net_yaw_deg"]) for episode in episodes]
            ),
            "fall_count": sum(int(episode["fallen"]) for episode in episodes),
        }

    mirror: dict[str, Any] = {}
    for left_key, right_key, name in TURN_PAIRS:
        left_by_seed = {
            int(episode["seed"]): episode
            for episode in summaries[left_key]["episodes"]
        }
        right_by_seed = {
            int(episode["seed"]): episode
            for episode in summaries[right_key]["episodes"]
        }
        residuals = [
            abs(
                float(left_by_seed[seed]["mean_actual_wz"])
                + float(right_by_seed[seed]["mean_actual_wz"])
            )
            for seed in seed_sets[0]
        ]
        mirror[name] = describe(residuals)

    return {
        "checkpoint": checkpoint_from_policy_path(suite["policy_path"]),
        "policy_path": suite["policy_path"],
        "policy_sha256": suite["policy_sha256"],
        "episode_count": suite["total_episodes"],
        "seeds": list(seed_sets[0]),
        "fall_count": sum(item["fall_count"] for item in commands.values()),
        "commands": commands,
        "turn_mirror_residual": mirror,
    }


def build_summary(suite_paths: Sequence[Path]) -> dict[str, Any]:
    checkpoints = sorted(
        (load_checkpoint(path) for path in suite_paths),
        key=lambda item: item["checkpoint"],
    )
    if len({item["checkpoint"] for item in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint inputs must be unique")
    seeds = checkpoints[0]["seeds"]
    if any(item["seeds"] != seeds for item in checkpoints):
        raise ValueError("all checkpoints must use the same paired seeds")
    return {
        "schema_version": 1,
        "benchmark_kind": "onnx_cpu_bam_pointgoal_moving_turn_checkpoint_screen",
        "checkpoint_count": len(checkpoints),
        "commands_per_checkpoint": 5,
        "episodes_per_command": 5,
        "total_episodes": sum(item["episode_count"] for item in checkpoints),
        "paired_seeds": seeds,
        "fall_count": sum(item["fall_count"] for item in checkpoints),
        "reward_available": False,
        "turn_mirror_residual_definition": "abs(mean_wz(left) + mean_wz(right)) per paired seed",
        "checkpoints": checkpoints,
    }


def write_summary(document: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def plot_summary(document: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoints = document["checkpoints"]
    labels = [str(item["checkpoint"]) for item in checkpoints]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)

    def error_line(axis: Any, values: list[dict[str, Any]], label: str, color: str) -> None:
        means = np.asarray([item["mean"] for item in values])
        stds = np.asarray([item["std"] for item in values])
        axis.errorbar(x, means, yerr=stds, fmt="o-", capsize=4, label=label, color=color)

    straight = [item["commands"]["straight"] for item in checkpoints]
    error_line(
        axes[0, 0],
        [item["net_yaw_deg"] for item in straight],
        "straight vx=0.25",
        "#6A1B9A",
    )
    axes[0, 0].axhline(0.0, color="#424242", linestyle="--", linewidth=1.1)
    axes[0, 0].set_title("Straight-command net yaw")
    axes[0, 0].set_ylabel("8 s net yaw (deg)")

    turn_styles = (
        ("gentle_left", "+0.25", "#D32F2F"),
        ("gentle_right", "-0.25", "#388E3C"),
        ("turn_left", "+0.50", "#F57C00"),
        ("turn_right", "-0.50", "#1976D2"),
    )
    for key, label, color in turn_styles:
        error_line(
            axes[0, 1],
            [item["commands"][key]["mean_actual_wz"] for item in checkpoints],
            label,
            color,
        )
    for target in (-0.5, -0.25, 0.25, 0.5):
        axes[0, 1].axhline(target, color="#9E9E9E", linestyle=":", linewidth=0.8)
    axes[0, 1].set_title("Moving-turn yaw response")
    axes[0, 1].set_ylabel("Mean actual wz (rad/s)")
    axes[0, 1].legend(ncol=2)

    for key, label, color in (
        ("straight", "straight", "#6A1B9A"),
        ("gentle_left", "gentle left", "#D32F2F"),
        ("gentle_right", "gentle right", "#388E3C"),
        ("turn_left", "left", "#F57C00"),
        ("turn_right", "right", "#1976D2"),
    ):
        error_line(
            axes[1, 0],
            [item["commands"][key]["mean_actual_vx"] for item in checkpoints],
            label,
            color,
        )
    axes[1, 0].set_title("Forward-speed response")
    axes[1, 0].set_ylabel("Mean actual vx (m/s)")
    axes[1, 0].legend(ncol=2)

    for key, label, color in (
        ("gentle", "gentle ±0.25", "#7B1FA2"),
        ("normal", "normal ±0.50", "#00796B"),
    ):
        error_line(
            axes[1, 1],
            [item["turn_mirror_residual"][key] for item in checkpoints],
            label,
            color,
        )
    axes[1, 1].axhline(0.0, color="#424242", linestyle="--", linewidth=1.1)
    axes[1, 1].set_title("Paired left/right mirror residual")
    axes[1, 1].set_ylabel("|wz(left) + wz(right)| (rad/s)")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Checkpoint iteration")
        axis.grid(alpha=0.23)
    fig.suptitle(
        "Candidate B PointGoal moving-turn quick screen — "
        f"{document['total_episodes']} episodes, {document['fall_count']} falls",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = build_summary([path.resolve() for path in args.suite])
    write_summary(document, args.summary_output.resolve())
    plot_summary(document, args.figure_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
