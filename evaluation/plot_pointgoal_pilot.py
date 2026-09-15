#!/usr/bin/env python3
"""Plot PointGoal trajectories in each episode's initial body frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]


def world_to_initial_body(
    x_world_m: float,
    y_world_m: float,
    start_x_world_m: float,
    start_y_world_m: float,
    start_yaw_rad: float,
) -> tuple[float, float]:
    """Express a world-frame point in the initial robot body frame."""

    dx = x_world_m - start_x_world_m
    dy = y_world_m - start_y_world_m
    cosine = math.cos(start_yaw_rad)
    sine = math.sin(start_yaw_rad)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def load_plot_episodes(summary_paths: Sequence[Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        for episode in document["episodes"]:
            raw_path = PROJECT_ROOT / episode["raw_steps_csv"]
            with raw_path.open("r", encoding="utf-8", newline="") as stream:
                rows = [row for row in csv.DictReader(stream) if row["phase"] == "test"]
            start = episode["start_state"]
            trajectory = [
                world_to_initial_body(
                    float(row["world_x"]),
                    float(row["world_y"]),
                    float(start["x_world_m"]),
                    float(start["y_world_m"]),
                    float(start["yaw_rad"]),
                )
                for row in rows
            ]
            goal = episode["goal_world"]
            goal_local = world_to_initial_body(
                float(goal["x_world_m"]),
                float(goal["y_world_m"]),
                float(start["x_world_m"]),
                float(start["y_world_m"]),
                float(start["yaw_rad"]),
            )
            episodes.append(
                {
                    "goal_name": episode["goal_name"],
                    "seed": episode["seed"],
                    "success": episode["success"],
                    "termination_reason": episode["termination_reason"],
                    "trajectory": trajectory,
                    "goal_local": goal_local,
                    "success_radius_m": document["protocol"]["success_radius_m"],
                }
            )
    return episodes


def plot_episodes(
    episodes: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    if not episodes:
        raise ValueError("at least one episode is required")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    colors = plt.get_cmap("tab10")
    for index, episode in enumerate(episodes):
        trajectory = episode["trajectory"]
        if not trajectory:
            continue
        x_values = [point[0] for point in trajectory]
        y_values = [point[1] for point in trajectory]
        label = (
            f"{episode['goal_name']} seed {episode['seed']} "
            f"({episode['termination_reason']})"
        )
        color = colors(index % 10)
        axis.plot(x_values, y_values, linewidth=1.6, color=color, label=label)
        axis.scatter(x_values[-1], y_values[-1], s=30, color=color, marker="x")
        goal_x, goal_y = episode["goal_local"]
        axis.scatter(goal_x, goal_y, s=45, color=color, marker="*")
        axis.add_patch(
            Circle(
                (goal_x, goal_y),
                float(episode["success_radius_m"]),
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.7,
            )
        )

    axis.scatter(0.0, 0.0, s=50, color="black", marker="o", label="start")
    axis.axhline(0.0, color="0.8", linewidth=0.8)
    axis.axvline(0.0, color="0.8", linewidth=0.8)
    axis.set_xlabel("Initial-body forward x (m)")
    axis.set_ylabel("Initial-body left y (m)")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot PointGoal trajectories in the initial body frame."
    )
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="MicroDuck Classical PointGoal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_paths = [path.expanduser().resolve() for path in args.summary]
    output_path = args.output.expanduser().resolve()
    episodes = load_plot_episodes(summary_paths)
    plot_episodes(episodes, output_path, args.title)
    print(f"plotted {len(episodes)} episodes: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
