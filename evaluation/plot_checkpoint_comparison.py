#!/usr/bin/env python3
"""Aggregate and plot the paired five-seed locomotion checkpoint probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
CHECKPOINTS = (5000, 5250, 5500, 5750, 5999)
COMMAND_NAMES = {
    "forward": "forward_030_checkpoint_probe_5",
    "yaw_pos": "yaw_pos_050_checkpoint_probe_5",
    "yaw_neg": "yaw_neg_050_checkpoint_probe_5",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def describe(values: Sequence[float]) -> dict[str, Any]:
    return {
        "values": list(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def load_checkpoint(result_root: Path, checkpoint: int) -> dict[str, Any]:
    summary_dir = result_root / f"model_{checkpoint}" / "summary"
    suite_path = summary_dir / f"checkpoint_probe_5_microduck_velocity_flat_{checkpoint}.json"
    suite = load_json(suite_path)
    if suite["total_episodes"] != 15:
        raise ValueError(f"{suite_path} does not contain 15 episodes")

    summaries: dict[str, dict[str, Any]] = {}
    for key, command_name in COMMAND_NAMES.items():
        path = summary_dir / f"{command_name}_microduck_velocity_flat_{checkpoint}.json"
        summary = load_json(path)
        if summary["command"]["name"] != command_name:
            raise ValueError(f"command mismatch in {path}")
        if len(summary["episodes"]) != 5:
            raise ValueError(f"{path} does not contain five episodes")
        summaries[key] = summary

    seed_sets = [
        tuple(episode["seed"] for episode in item["episodes"])
        for item in summaries.values()
    ]
    if len(set(seed_sets)) != 1:
        raise ValueError(f"commands for model_{checkpoint} do not use paired seeds")
    fall_count = sum(
        int(episode["fallen"])
        for item in summaries.values()
        for episode in item["episodes"]
    )

    forward = summaries["forward"]["episodes"]
    yaw_pos = summaries["yaw_pos"]["episodes"]
    yaw_neg = summaries["yaw_neg"]["episodes"]
    pos_by_seed = {episode["seed"]: episode for episode in yaw_pos}
    neg_by_seed = {episode["seed"]: episode for episode in yaw_neg}
    mirror_residuals = [
        abs(pos_by_seed[seed]["mean_actual_wz"] + neg_by_seed[seed]["mean_actual_wz"])
        for seed in seed_sets[0]
    ]

    return {
        "checkpoint": checkpoint,
        "policy_sha256": suite["policy_sha256"],
        "episode_count": suite["total_episodes"],
        "seeds": list(seed_sets[0]),
        "fall_count": fall_count,
        "forward_net_yaw_deg": describe([episode["net_yaw_deg"] for episode in forward]),
        "forward_mean_actual_vx": describe(
            [episode["mean_actual_vx"] for episode in forward]
        ),
        "yaw_pos_mean_actual_wz": describe(
            [episode["mean_actual_wz"] for episode in yaw_pos]
        ),
        "yaw_neg_mean_actual_wz": describe(
            [episode["mean_actual_wz"] for episode in yaw_neg]
        ),
        "turn_mirror_residual": describe(mirror_residuals),
    }


def write_summary(result_root: Path, output_path: Path) -> dict[str, Any]:
    checkpoints = [load_checkpoint(result_root, checkpoint) for checkpoint in CHECKPOINTS]
    seeds = checkpoints[0]["seeds"]
    if any(item["seeds"] != seeds for item in checkpoints):
        raise ValueError("checkpoint probes do not use the same paired seeds")

    document = {
        "schema_version": 1,
        "benchmark_kind": "onnx_cpu_bam_checkpoint_comparison",
        "checkpoint_range": [CHECKPOINTS[0], CHECKPOINTS[-1]],
        "checkpoint_count": len(CHECKPOINTS),
        "commands_per_checkpoint": 3,
        "episodes_per_command": 5,
        "total_episodes": sum(item["episode_count"] for item in checkpoints),
        "paired_seeds": seeds,
        "fall_count": sum(item["fall_count"] for item in checkpoints),
        "turn_mirror_residual_definition": "abs(mean_wz(+0.5) + mean_wz(-0.5)) per paired seed",
        "turn_mirror_residual_note": "Low residual alone does not imply good tracking when both turn responses are near zero.",
        "checkpoints": checkpoints,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def plot_summary(document: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    checkpoints = document["checkpoints"]
    labels = [str(item["checkpoint"]) for item in checkpoints]
    x = np.arange(len(labels), dtype=float)
    offsets = np.asarray([-0.12, -0.06, 0.0, 0.06, 0.12])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    def points_and_mean(ax: Any, metric: str, color: str, label: str | None = None) -> None:
        for index, item in enumerate(checkpoints):
            values = np.asarray(item[metric]["values"], dtype=float)
            ax.scatter(
                np.full(values.shape, x[index]) + offsets,
                values,
                s=28,
                alpha=0.55,
                color=color,
                edgecolors="none",
            )
        means = np.asarray([item[metric]["mean"] for item in checkpoints])
        stds = np.asarray([item[metric]["std"] for item in checkpoints])
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o-",
            capsize=4,
            linewidth=1.8,
            color=color,
            label=label or "mean ± population SD",
        )

    ax = axes[0, 0]
    points_and_mean(ax, "forward_net_yaw_deg", "#6A1B9A")
    ax.axhline(0.0, color="#424242", linestyle="--", linewidth=1.2, label="target")
    ax.set_title("Forward command: 10 s net yaw")
    ax.set_ylabel("Net yaw (deg)")
    ax.legend(loc="best")

    ax = axes[0, 1]
    points_and_mean(ax, "forward_mean_actual_vx", "#1565C0")
    ax.axhline(0.3, color="#424242", linestyle="--", linewidth=1.2, label="target +0.30")
    ax.set_title("Forward command: episode-mean vx")
    ax.set_ylabel("vx (m/s)")
    ax.legend(loc="best")

    ax = axes[1, 0]
    colors = {"yaw_pos_mean_actual_wz": "#C62828", "yaw_neg_mean_actual_wz": "#2E7D32"}
    series = (
        ("yaw_pos_mean_actual_wz", "+0.50 command", -0.05),
        ("yaw_neg_mean_actual_wz", "-0.50 command", 0.05),
    )
    for metric, label, x_offset in series:
        color = colors[metric]
        for index, item in enumerate(checkpoints):
            values = np.asarray(item[metric]["values"], dtype=float)
            ax.scatter(
                np.full(values.shape, x[index] + x_offset) + offsets * 0.55,
                values,
                s=24,
                alpha=0.48,
                color=color,
                edgecolors="none",
            )
        means = np.asarray([item[metric]["mean"] for item in checkpoints])
        stds = np.asarray([item[metric]["std"] for item in checkpoints])
        ax.errorbar(x + x_offset, means, yerr=stds, fmt="o-", capsize=4, color=color, label=label)
    ax.axhline(0.5, color="#C62828", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(-0.5, color="#2E7D32", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="#757575", linewidth=0.9)
    ax.set_title("Turn commands: episode-mean wz")
    ax.set_ylabel("wz (rad/s)")
    ax.legend(loc="best")

    ax = axes[1, 1]
    points_and_mean(ax, "turn_mirror_residual", "#EF6C00")
    ax.axhline(0.0, color="#424242", linestyle="--", linewidth=1.2, label="ideal mirror residual")
    ax.set_title("Paired turn mirror residual")
    ax.set_ylabel("|wz(+) + wz(-)| (rad/s)")
    ax.legend(loc="best")

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Checkpoint iteration")
        ax.grid(True, alpha=0.23)

    fig.suptitle(
        "MicroDuck checkpoint probe — 3 commands × 5 paired seeds "
        f"({document['total_episodes']} episodes, {document['fall_count']} falls)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "week02" / "checkpoint_comparison",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "week02"
            / "summary"
            / "checkpoint_comparison_5000_5999.json"
        ),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "week02"
            / "figures"
            / "checkpoint_comparison_5000_5999.png"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = write_summary(args.result_root.resolve(), args.summary_output.resolve())
    plot_summary(document, args.figure_output.resolve())
    print(
        f"Loaded {document['checkpoint_count']} checkpoints / "
        f"{document['total_episodes']} episodes / {document['fall_count']} falls"
    )
    print(f"Saved summary: {args.summary_output.resolve()}")
    print(f"Saved figure: {args.figure_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
