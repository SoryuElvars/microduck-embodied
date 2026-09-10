#!/usr/bin/env python3
"""Aggregate and plot the three-checkpoint, eight-command response comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
CHECKPOINTS = (5500, 5750, 5999)
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
COMMANDS = (
    ("stationary_checkpoint_response_5", "stop", (0.0, 0.0, 0.0)),
    ("forward_010_checkpoint_response_5", "vx +0.1", (0.1, 0.0, 0.0)),
    ("forward_030_checkpoint_response_5", "vx +0.3", (0.3, 0.0, 0.0)),
    ("backward_030_checkpoint_response_5", "vx -0.3", (-0.3, 0.0, 0.0)),
    ("lateral_pos_020_checkpoint_response_5", "vy +0.2", (0.0, 0.2, 0.0)),
    ("lateral_neg_020_checkpoint_response_5", "vy -0.2", (0.0, -0.2, 0.0)),
    ("yaw_pos_050_checkpoint_response_5", "wz +0.5", (0.0, 0.0, 0.5)),
    ("yaw_neg_050_checkpoint_response_5", "wz -0.5", (0.0, 0.0, -0.5)),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_checkpoint(result_root: Path, checkpoint: int) -> dict[str, Any]:
    summary_dir = result_root / f"model_{checkpoint}" / "summary"
    suite_path = (
        summary_dir / f"checkpoint_response_5_microduck_velocity_flat_{checkpoint}.json"
    )
    suite = load_json(suite_path)
    if suite["command_count"] != 8 or suite["total_episodes"] != 40:
        raise ValueError(f"{suite_path} is not an 8-command, 40-episode suite")

    suite_commands = {command["name"]: command for command in suite["commands"]}
    commands = []
    checkpoint_seeds: tuple[int, ...] | None = None
    for name, label, target in COMMANDS:
        if name not in suite_commands:
            raise ValueError(f"{suite_path} is missing command {name}")
        summary_path = PROJECT_ROOT / suite_commands[name]["summary_json"]
        summary = load_json(summary_path)
        seeds = tuple(episode["seed"] for episode in summary["episodes"])
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"{summary_path} does not use seeds 42-46")
        if checkpoint_seeds is None:
            checkpoint_seeds = seeds
        elif seeds != checkpoint_seeds:
            raise ValueError(f"commands for model_{checkpoint} do not use paired seeds")
        commands.append(
            {
                "name": name,
                "label": label,
                "target": {"vx": target[0], "vy": target[1], "wz": target[2]},
                "aggregate": summary["aggregate"],
                "episodes": summary["episodes"],
                "summary_json": str(summary_path.relative_to(PROJECT_ROOT)),
            }
        )

    return {
        "checkpoint": checkpoint,
        "policy_path": suite["policy_path"],
        "policy_sha256": suite["policy_sha256"],
        "episode_count": suite["total_episodes"],
        "fall_count": sum(
            int(episode["fallen"])
            for command in commands
            for episode in command["episodes"]
        ),
        "seeds": list(checkpoint_seeds or ()),
        "commands": commands,
        "suite_summary_json": str(suite_path.relative_to(PROJECT_ROOT)),
    }


def write_summary(result_root: Path, output_path: Path) -> dict[str, Any]:
    checkpoints = [load_checkpoint(result_root, checkpoint) for checkpoint in CHECKPOINTS]
    document = {
        "schema_version": 1,
        "benchmark_kind": "onnx_cpu_bam_three_checkpoint_eight_command_comparison",
        "checkpoint_count": len(checkpoints),
        "command_count": len(COMMANDS),
        "episodes_per_command": len(EXPECTED_SEEDS),
        "paired_seeds": list(EXPECTED_SEEDS),
        "total_episodes": sum(item["episode_count"] for item in checkpoints),
        "fall_count": sum(item["fall_count"] for item in checkpoints),
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
    from matplotlib.colors import TwoSlopeNorm

    checkpoints = document["checkpoints"]
    row_labels = [str(item["checkpoint"]) for item in checkpoints]
    column_labels = [item[1] for item in COMMANDS]
    panels = (
        ("mean_actual_vx", "Episode-mean vx (m/s)", 0.3, ".3f"),
        ("mean_actual_vy", "Episode-mean vy (m/s)", 0.2, ".3f"),
        ("mean_actual_wz", "Episode-mean wz (rad/s)", 0.5, ".3f"),
        ("net_yaw_deg", "Net yaw over 10 s (deg)", 230.0, ".1f"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(17, 8.5))
    for ax, (metric, title, limit, number_format) in zip(
        axes.flat, panels, strict=True
    ):
        means = np.asarray(
            [
                [command["aggregate"][metric]["mean"] for command in item["commands"]]
                for item in checkpoints
            ],
            dtype=float,
        )
        stds = np.asarray(
            [
                [command["aggregate"][metric]["std"] for command in item["commands"]]
                for item in checkpoints
            ],
            dtype=float,
        )
        image = ax.imshow(
            means,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        for row in range(means.shape[0]):
            for column in range(means.shape[1]):
                color = "white" if abs(means[row, column]) > limit * 0.55 else "black"
                ax.text(
                    column,
                    row,
                    f"{means[row, column]:{number_format}}\n±{stds[row, column]:{number_format}}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=color,
                )
        ax.set_title(title)
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)), row_labels)
        ax.set_ylabel("Checkpoint iteration")
        fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)

    fig.suptitle(
        "MicroDuck checkpoint response matrix — 8 commands × 5 paired seeds "
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
            / "checkpoint_response_comparison_5500_5999.json"
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
            / "checkpoint_response_comparison_5500_5999.png"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = write_summary(args.result_root.resolve(), args.summary_output.resolve())
    plot_summary(document, args.figure_output.resolve())
    print(
        f"Loaded {document['checkpoint_count']} checkpoints / "
        f"{document['command_count']} commands / {document['total_episodes']} episodes / "
        f"{document['fall_count']} falls"
    )
    print(f"Saved summary: {args.summary_output.resolve()}")
    print(f"Saved figure: {args.figure_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
