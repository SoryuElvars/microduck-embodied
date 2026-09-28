#!/usr/bin/env python3
"""Create reviewable Week 4 Classical PointGoal figures from frozen results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
CONTROLLERS = ("naive_p", "constrained")


def load_summary(summary_path: Path) -> dict[str, Any]:
    """Load a dual-controller PointGoal summary and check its minimum schema."""

    with summary_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    controllers = document.get("controllers")
    if not isinstance(controllers, dict):
        raise ValueError("summary does not contain controller results")
    missing = set(CONTROLLERS) - set(controllers)
    if missing:
        raise ValueError(f"summary is missing controllers: {sorted(missing)}")
    return document


def protocol_success_radius_m(summary: dict[str, Any]) -> float:
    """Read the task radius from the versioned protocol named by a run summary."""

    protocol_path = PROJECT_ROOT / str(summary["protocol_path"])
    with protocol_path.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    radii = {float(episode["success_radius"]) for episode in protocol["episodes"]}
    if len(radii) != 1:
        raise ValueError("plotting requires one frozen success radius")
    return radii.pop()


def episodes_by_id(summary: dict[str, Any], controller_id: str) -> dict[str, dict[str, Any]]:
    """Index one controller's results by frozen EpisodeSpec id."""

    episodes = summary["controllers"][controller_id]["episodes"]
    return {str(episode["episode_id"]): episode for episode in episodes}


def _body_frame(
    x_world: float,
    y_world: float,
    start_x: float,
    start_y: float,
    start_yaw: float,
) -> tuple[float, float]:
    dx = x_world - start_x
    dy = y_world - start_y
    cosine = math.cos(start_yaw)
    sine = math.sin(start_yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def load_test_rows(raw_steps_csv: str) -> list[dict[str, str]]:
    """Read only the navigation and post-arrival portion of a raw trajectory."""

    raw_path = PROJECT_ROOT / raw_steps_csv
    with raw_path.open(encoding="utf-8", newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if row["phase"] in {"navigate", "post_arrival"}
        ]


def paired_metric_values(
    summary: dict[str, Any], metric: str
) -> tuple[list[float], list[float]]:
    """Return values from scenes where both controllers succeeded."""

    naive = episodes_by_id(summary, "naive_p")
    constrained = episodes_by_id(summary, "constrained")
    shared_success_ids = [
        episode_id
        for episode_id, naive_episode in naive.items()
        if naive_episode["success"] and constrained[episode_id]["success"]
    ]
    return (
        [float(naive[episode_id][metric]) for episode_id in shared_success_ids],
        [float(constrained[episode_id][metric]) for episode_id in shared_success_ids],
    )


def plot_paired_metrics(summary: dict[str, Any], output_path: Path) -> None:
    """Plot paired controller evidence without pooling unmatched timeout cases."""

    import matplotlib.pyplot as plt

    metrics = (
        ("path_efficiency", "Path efficiency", "higher is better"),
        ("completion_time_s", "Completion time (s)", "lower is better"),
        ("post_arrival_stop_drift_m", "Post-arrival net displacement (m)", "lower is better"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(12.6, 4.0))
    for axis, (metric, label, direction) in zip(axes, metrics, strict=True):
        naive, constrained = paired_metric_values(summary, metric)
        lower = min(naive + constrained)
        upper = max(naive + constrained)
        pad = max((upper - lower) * 0.08, 0.01)
        axis.scatter(naive, constrained, s=14, alpha=0.65, color="#4477aa")
        axis.plot(
            [lower - pad, upper + pad],
            [lower - pad, upper + pad],
            color="0.35",
            linewidth=1.0,
            linestyle="--",
        )
        axis.set_xlim(lower - pad, upper + pad)
        axis.set_ylim(lower - pad, upper + pad)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"Naive P: {label}")
        axis.set_ylabel(f"Constrained: {label}")
        axis.set_title(direction)
        axis.grid(alpha=0.22)
    figure.suptitle("Week 4 formal: paired successes (n=198)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _trajectory_in_initial_body_frame(
    episode: dict[str, Any], rows: Iterable[dict[str, str]]
) -> list[tuple[float, float]]:
    start = episode["navigation_start_state"]
    return [
        _body_frame(
            float(row["world_x"]),
            float(row["world_y"]),
            float(start["x_world_m"]),
            float(start["y_world_m"]),
            float(start["yaw_rad"]),
        )
        for row in rows
    ]


def plot_case(summary: dict[str, Any], episode_id: str, output_path: Path) -> None:
    """Plot a matched scenario's paths and distance-to-goal time series."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    figure, (trajectory_axis, distance_axis) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    palette = {"naive_p": "#cc6677", "constrained": "#4477aa"}
    labels = {"naive_p": "Naive P", "constrained": "Constrained"}
    goal_local: tuple[float, float] | None = None
    success_radius: float | None = None

    for controller_id in CONTROLLERS:
        episode = episodes_by_id(summary, controller_id)[episode_id]
        rows = load_test_rows(str(episode["raw_steps_csv"]))
        trajectory = _trajectory_in_initial_body_frame(episode, rows)
        if not trajectory:
            raise ValueError(f"{controller_id}/{episode_id} has no test trajectory")
        start = episode["navigation_start_state"]
        goal = episode["goal_world"]
        goal_local = _body_frame(
            float(goal["x_world_m"]),
            float(goal["y_world_m"]),
            float(start["x_world_m"]),
            float(start["y_world_m"]),
            float(start["yaw_rad"]),
        )
        success_radius = protocol_success_radius_m(summary)
        color = palette[controller_id]
        x_values, y_values = zip(*trajectory, strict=True)
        trajectory_axis.plot(x_values, y_values, color=color, linewidth=1.7, label=labels[controller_id])
        trajectory_axis.scatter(x_values[-1], y_values[-1], color=color, marker="x", s=42)
        distance_axis.plot(
            [float(row["time_s"]) for row in rows],
            [float(row["distance_to_goal_m"]) for row in rows],
            color=color,
            linewidth=1.5,
            label=f"{labels[controller_id]} ({episode['termination_reason']})",
        )

    assert goal_local is not None and success_radius is not None
    trajectory_axis.scatter(0.0, 0.0, color="black", marker="o", s=38, label="start")
    trajectory_axis.scatter(*goal_local, color="black", marker="*", s=82, label="goal")
    trajectory_axis.add_patch(Circle(goal_local, success_radius, fill=False, color="0.25", linestyle="--"))
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set_xlabel("Initial-body forward x (m)")
    trajectory_axis.set_ylabel("Initial-body left y (m)")
    trajectory_axis.set_title("Matched trajectory")
    trajectory_axis.grid(alpha=0.22)
    trajectory_axis.legend(fontsize=8)
    distance_axis.axhline(success_radius, color="0.25", linestyle="--", linewidth=1.0, label="success radius")
    distance_axis.set_xlabel("Time (s)")
    distance_axis.set_ylabel("Distance to goal (m)")
    distance_axis.set_title("Position error")
    distance_axis.grid(alpha=0.22)
    distance_axis.legend(fontsize=8)
    figure.suptitle(f"Week 4 formal case: {episode_id}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = load_summary(args.summary.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    plot_paired_metrics(summary, output_dir / "paired_controller_metrics.png")
    for episode_id in args.case:
        plot_case(summary, episode_id, output_dir / f"case_{episode_id}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
