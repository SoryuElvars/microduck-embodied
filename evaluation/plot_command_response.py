#!/usr/bin/env python3
"""Plot the eight-command MicroDuck locomotion response matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite = load_json(path)
    commands = suite.get("commands")
    if not isinstance(commands, list) or len(commands) < 2:
        raise ValueError("suite summary must contain at least two commands")

    summaries = []
    for command in commands:
        summary_path = PROJECT_ROOT / command["summary_json"]
        summary = load_json(summary_path)
        if summary["command"]["name"] != command["name"]:
            raise ValueError(f"command mismatch in {summary_path}")
        summaries.append(summary)
    return suite, summaries


def short_label(command: dict[str, Any]) -> str:
    vx = float(command["vx"])
    vy = float(command["vy"])
    wz = float(command["wz"])
    if vx == vy == wz == 0.0:
        return "stop"
    if vx:
        return f"vx {vx:+.1f}"
    if vy:
        return f"vy {vy:+.1f}"
    return f"wz {wz:+.1f}"


def plot_command_response(suite_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    suite, summaries = load_suite(suite_path)
    commands = [summary["command"] for summary in summaries]
    labels = [short_label(command) for command in commands]
    x = np.arange(len(commands))

    dimensions = (
        ("vx", "mean_actual_vx", "rmse_vx", "#1565C0"),
        ("vy", "mean_actual_vy", "rmse_vy", "#2E7D32"),
        ("wz", "mean_actual_wz", "rmse_wz", "#C62828"),
    )
    target = {
        dimension: np.asarray([float(command[dimension]) for command in commands])
        for dimension, _, _, _ in dimensions
    }
    actual = {
        dimension: np.asarray(
            [float(summary["aggregate"][metric]["mean"]) for summary in summaries]
        )
        for dimension, metric, _, _ in dimensions
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    limits = (-0.56, 0.56)
    ax.plot(limits, limits, color="#424242", linestyle="--", linewidth=1.5, label="ideal")
    for dimension, _, _, color in dimensions:
        ax.scatter(
            target[dimension],
            actual[dimension],
            s=66,
            alpha=0.82,
            color=color,
            label=dimension,
        )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Commanded vs episode-mean velocity")
    ax.set_xlabel("Command (m/s or rad/s)")
    ax.set_ylabel("Actual mean (m/s or rad/s)")
    ax.legend(loc="best")

    ax = axes[0, 1]
    yaw_mean = actual["wz"]
    yaw_std = np.asarray(
        [float(summary["aggregate"]["mean_actual_wz"]["std"]) for summary in summaries]
    )
    ax.errorbar(
        x,
        yaw_mean,
        yerr=yaw_std,
        fmt="o",
        capsize=4,
        color="#6A1B9A",
        label="actual mean ± population SD",
    )
    ax.scatter(x, target["wz"], marker="x", s=80, color="#212121", label="target wz")
    ax.axhline(0.0, color="#757575", linewidth=1.0)
    ax.set_title("Yaw-rate response and left/right asymmetry")
    ax.set_ylabel("Yaw rate (rad/s)")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.legend(loc="best")

    ax = axes[1, 0]
    yaw_samples = [
        [float(episode["net_yaw_deg"]) for episode in summary["episodes"]]
        for summary in summaries
    ]
    boxes = ax.boxplot(yaw_samples, positions=x, widths=0.56, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]:
        box.set(facecolor="#D1C4E9", edgecolor="#5E35B1")
    target_yaw = np.asarray(
        [math.degrees(float(command["wz"]) * float(command["duration_s"])) for command in commands]
    )
    ax.scatter(x, target_yaw, marker="x", s=80, color="#212121", label="integrated target yaw")
    ax.axhline(0.0, color="#757575", linewidth=1.0)
    ax.set_title("Ten-second net-yaw distributions")
    ax.set_ylabel("Net yaw (deg)")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.legend(loc="best")

    ax = axes[1, 1]
    width = 0.24
    for offset, (dimension, _, rmse_metric, color) in zip(
        (-width, 0.0, width), dimensions, strict=True
    ):
        rmse = [float(summary["aggregate"][rmse_metric]["mean"]) for summary in summaries]
        ax.bar(x + offset, rmse, width=width, color=color, alpha=0.8, label=f"RMSE {dimension}")
    ax.set_title("Tracking RMSE by command")
    ax.set_ylabel("RMSE (m/s or rad/s)")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.legend(loc="best")

    for ax in axes.flat:
        ax.grid(True, alpha=0.23)

    fall_count = sum(
        int(episode["fallen"])
        for summary in summaries
        for episode in summary["episodes"]
    )
    total_episodes = int(suite["total_episodes"])
    fig.suptitle(
        f"MicroDuck model_5999 — 8-command response matrix "
        f"({total_episodes} episodes, {fall_count} falls)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Loaded {len(summaries)} commands / {total_episodes} episodes")
    print(f"Falls: {fall_count}")
    print(f"Saved figure: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a locomotion command-response suite")
    parser.add_argument("--suite", type=Path, required=True, help="Suite summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_command_response(args.suite.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
