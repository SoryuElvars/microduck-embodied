#!/usr/bin/env python3
"""Aggregate and plot a three-command checkpoint gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
COMMAND_KEYS = {
    (0.3, 0.0, 0.0): "forward",
    (0.0, 0.0, 0.5): "yaw_pos",
    (0.0, 0.0, -0.5): "yaw_neg",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def describe(values: Sequence[float], *, keep_values: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }
    if keep_values:
        result["values"] = list(values)
    return result


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
        raise ValueError(f"unexpected checkpoint-gate command: {signature}") from error


def raw_action_metrics(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    action_deltas: list[float] = []
    nonfinite_count = 0

    for summary in summaries:
        for episode in summary["episodes"]:
            previous_action: list[float] | None = None
            csv_path = resolve_project_path(episode["raw_steps_csv"])
            with csv_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    for key, value in row.items():
                        if key in {"phase", "fallen"} or value in {None, ""}:
                            continue
                        try:
                            numeric = float(value)
                        except ValueError:
                            continue
                        nonfinite_count += int(not math.isfinite(numeric))

                    if row["phase"] != "test":
                        previous_action = None
                        continue
                    action = [float(row[f"action_{index:02d}"]) for index in range(14)]
                    if previous_action is not None:
                        action_deltas.append(
                            math.sqrt(
                                sum(
                                    (current - previous) ** 2
                                    for current, previous in zip(
                                        action, previous_action, strict=True
                                    )
                                )
                            )
                        )
                    previous_action = action

    ordered = sorted(action_deltas)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "raw_numeric_nonfinite_count": nonfinite_count,
        "action_delta_l2": {
            **describe(action_deltas, keep_values=False),
            "p95": ordered[p95_index],
            "sample_count": len(action_deltas),
        },
    }


def load_checkpoint(suite_path: Path) -> dict[str, Any]:
    suite = load_json(suite_path)
    if suite.get("command_count") != 3 or suite.get("total_episodes") != 15:
        raise ValueError(f"{suite_path} is not a 3-command x 5-episode gate")

    summaries: dict[str, dict[str, Any]] = {}
    for item in suite["commands"]:
        summary_path = resolve_project_path(item["summary_json"])
        summary = load_json(summary_path)
        if summary["command"]["name"] != item["name"]:
            raise ValueError(f"command mismatch in {summary_path}")
        if len(summary["episodes"]) != 5:
            raise ValueError(f"{summary_path} does not contain five episodes")
        key = command_key(summary["command"])
        if key in summaries:
            raise ValueError(f"duplicate {key} command in {suite_path}")
        summaries[key] = summary
    if set(summaries) != set(COMMAND_KEYS.values()):
        raise ValueError(f"missing checkpoint-gate command in {suite_path}")

    seed_sets = [
        tuple(int(episode["seed"]) for episode in summary["episodes"])
        for summary in summaries.values()
    ]
    if len(set(seed_sets)) != 1:
        raise ValueError(f"commands in {suite_path} do not use paired seeds")

    forward = summaries["forward"]["episodes"]
    yaw_pos = summaries["yaw_pos"]["episodes"]
    yaw_neg = summaries["yaw_neg"]["episodes"]
    all_summaries = list(summaries.values())
    fall_count = sum(
        int(episode["fallen"])
        for summary in all_summaries
        for episode in summary["episodes"]
    )

    return {
        "checkpoint": checkpoint_from_policy_path(suite["policy_path"]),
        "policy_path": suite["policy_path"],
        "policy_sha256": suite["policy_sha256"],
        "episode_count": suite["total_episodes"],
        "seeds": list(seed_sets[0]),
        "fall_count": fall_count,
        "forward_net_yaw_deg": describe(
            [float(episode["net_yaw_deg"]) for episode in forward]
        ),
        "forward_mean_actual_vx": describe(
            [float(episode["mean_actual_vx"]) for episode in forward]
        ),
        "yaw_pos_mean_actual_wz": describe(
            [float(episode["mean_actual_wz"]) for episode in yaw_pos]
        ),
        "yaw_neg_mean_actual_wz": describe(
            [float(episode["mean_actual_wz"]) for episode in yaw_neg]
        ),
        "forward_rmse_wz": describe(
            [float(episode["rmse_wz"]) for episode in forward]
        ),
        "yaw_pos_rmse_wz": describe(
            [float(episode["rmse_wz"]) for episode in yaw_pos]
        ),
        "yaw_neg_rmse_wz": describe(
            [float(episode["rmse_wz"]) for episode in yaw_neg]
        ),
        **raw_action_metrics(all_summaries),
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
        "benchmark_kind": "onnx_cpu_bam_three_command_checkpoint_gate",
        "checkpoint_count": len(checkpoints),
        "commands_per_checkpoint": 3,
        "episodes_per_command": 5,
        "total_episodes": sum(item["episode_count"] for item in checkpoints),
        "paired_seeds": seeds,
        "fall_count": sum(item["fall_count"] for item in checkpoints),
        "reward_available": False,
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
    offsets = np.linspace(-0.12, 0.12, 5)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    def points_and_mean(ax: Any, metric: str, color: str) -> None:
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
            label="mean ± population SD",
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
    for metric, label, color, x_offset in (
        ("yaw_pos_mean_actual_wz", "+0.50 command", "#C62828", -0.045),
        ("yaw_neg_mean_actual_wz", "-0.50 command", "#2E7D32", 0.045),
    ):
        for index, item in enumerate(checkpoints):
            values = np.asarray(item[metric]["values"], dtype=float)
            ax.scatter(
                np.full(values.shape, x[index] + x_offset) + offsets * 0.5,
                values,
                s=24,
                alpha=0.48,
                color=color,
                edgecolors="none",
            )
        means = np.asarray([item[metric]["mean"] for item in checkpoints])
        stds = np.asarray([item[metric]["std"] for item in checkpoints])
        ax.errorbar(
            x + x_offset,
            means,
            yerr=stds,
            fmt="o-",
            capsize=4,
            color=color,
            label=label,
        )
    ax.axhline(0.5, color="#C62828", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(-0.5, color="#2E7D32", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="#757575", linewidth=0.9)
    ax.set_title("Turn commands: episode-mean wz")
    ax.set_ylabel("wz (rad/s)")
    ax.legend(loc="best")

    ax = axes[1, 1]
    width = 0.24
    for offset, metric, label, color in (
        (-width, "forward_rmse_wz", "forward", "#1565C0"),
        (0.0, "yaw_pos_rmse_wz", "wz +0.50", "#C62828"),
        (width, "yaw_neg_rmse_wz", "wz -0.50", "#2E7D32"),
    ):
        values = [item[metric]["mean"] for item in checkpoints]
        ax.bar(x + offset, values, width=width, color=color, alpha=0.8, label=label)
    ax.set_title("Yaw tracking RMSE by command")
    ax.set_ylabel("RMSE wz (rad/s)")
    ax.legend(loc="best")

    for ax in axes.flat:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Checkpoint iteration")
        ax.grid(True, alpha=0.23)

    fig.suptitle(
        "MicroDuck angular std sqrt(0.25) checkpoint gate — "
        f"3 commands × 5 paired seeds ({document['total_episodes']} episodes, "
        f"{document['fall_count']} falls)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        action="append",
        required=True,
        help="Suite summary JSON; repeat once per checkpoint",
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = build_summary([path.resolve() for path in args.suite])
    write_summary(document, args.summary_output.resolve())
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
