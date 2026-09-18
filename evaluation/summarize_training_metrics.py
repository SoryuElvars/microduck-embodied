#!/usr/bin/env python3
"""Summarize and compare RSL-RL TensorBoard training metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


METRICS = {
    "error_vel_yaw": "Metrics/twist/error_vel_yaw",
    "error_vel_xy": "Metrics/twist/error_vel_xy",
    "track_angular_velocity": "Episode_Reward/track_angular_velocity",
    "track_linear_velocity": "Episode_Reward/track_linear_velocity",
    "mean_reward": "Train/mean_reward",
    "mean_episode_length": "Train/mean_episode_length",
    "fell_over": "Episode_Termination/fell_over",
    "nan_state": "Episode_Termination/nan_state",
    "value_loss": "Loss/value",
    "surrogate_loss": "Loss/surrogate",
    "mean_action_std": "Policy/mean_std",
}

PLOT_METRICS = (
    ("error_vel_yaw", "Yaw velocity error", "lower is better"),
    ("track_angular_velocity", "Angular tracking reward", "definition differs"),
    ("error_vel_xy", "XY velocity error", "lower is better"),
    ("mean_reward", "Mean training reward", "reward definition differs"),
    ("fell_over", "Fell-over termination", "lower is better"),
    ("mean_episode_length", "Mean episode length", "higher is better"),
)


@dataclass(frozen=True)
class ScalarPoint:
    iteration: int
    value: float
    wall_time: float
    run_dir: str


def parse_series_arg(value: str) -> tuple[str, list[Path]]:
    """Parse LABEL=RUN_DIR[,RUN_DIR...] while preserving run order."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("series must be LABEL=RUN_DIR[,RUN_DIR...]")
    label, raw_paths = value.split("=", 1)
    paths = [Path(item) for item in raw_paths.split(",") if item]
    if not label or not paths:
        raise argparse.ArgumentTypeError("series label and at least one run are required")
    return label, paths


def window_stats(
    points: dict[int, ScalarPoint], checkpoint: int, window_size: int
) -> dict[str, float | int] | None:
    """Return statistics for the inclusive window ending at a checkpoint."""

    selected = [
        point.value
        for step, point in sorted(points.items())
        if checkpoint - window_size < step <= checkpoint
    ]
    if not selected:
        return None
    return {
        "count": len(selected),
        "mean": statistics.fmean(selected),
        "std": statistics.pstdev(selected),
        "min": min(selected),
        "max": max(selected),
        "last": selected[-1],
    }


def _event_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(matches) != 1:
        raise ValueError(f"expected one TensorBoard event file in {run_dir}, got {matches}")
    return matches[0]


def load_series(
    run_dirs: Sequence[Path],
) -> tuple[dict[str, dict[int, ScalarPoint]], list[dict[str, Any]]]:
    """Load runs in order; later runs replace duplicate resume iterations."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, dict[int, ScalarPoint]] = {name: {} for name in METRICS}
    run_summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        event_path = _event_file(run_dir)
        accumulator = EventAccumulator(
            str(event_path), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        available = set(accumulator.Tags()["scalars"])
        run_wall_times: list[float] = []
        run_steps: list[int] = []
        for metric_name, tag in METRICS.items():
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                point = ScalarPoint(
                    iteration=int(event.step),
                    value=float(event.value),
                    wall_time=float(event.wall_time),
                    run_dir=str(run_dir),
                )
                merged[metric_name][point.iteration] = point
                if metric_name == "mean_reward":
                    run_wall_times.append(point.wall_time)
                    run_steps.append(point.iteration)
        run_summaries.append(
            {
                "run_dir": str(run_dir),
                "event_file": str(event_path),
                "first_iteration": min(run_steps),
                "last_iteration": max(run_steps),
                "duration_s": max(run_wall_times) - min(run_wall_times),
            }
        )
    return merged, run_summaries


def _write_csv(
    output_path: Path, series: dict[str, dict[str, dict[int, ScalarPoint]]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("series", "iteration", "metric", "value", "wall_time", "run_dir"),
        )
        writer.writeheader()
        for label, metrics in series.items():
            for metric, points in metrics.items():
                for point in sorted(points.values(), key=lambda item: item.iteration):
                    writer.writerow(
                        {
                            "series": label,
                            "iteration": point.iteration,
                            "metric": metric,
                            "value": point.value,
                            "wall_time": point.wall_time,
                            "run_dir": point.run_dir,
                        }
                    )


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def _plot(
    output_path: Path,
    series: dict[str, dict[str, dict[int, ScalarPoint]]],
    rolling_window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, (metric, title, note) in zip(axes.flat, PLOT_METRICS, strict=True):
        for label, metrics in series.items():
            points = sorted(metrics[metric].values(), key=lambda item: item.iteration)
            steps = [point.iteration for point in points]
            values = [point.value for point in points]
            axis.plot(
                steps,
                _rolling_mean(values, rolling_window),
                linewidth=1.8,
                label=label,
            )
        axis.set_title(f"{title}\n({note}, {rolling_window}-iteration mean)")
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Candidate A vs Candidate B training metrics", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        type=parse_series_arg,
        metavar="LABEL=RUN[,RUN...]",
    )
    parser.add_argument("--checkpoints", default="250,500,750,1000,1250,1500,1750,1999")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = [int(item) for item in args.checkpoints.split(",")]
    all_series: dict[str, dict[str, dict[int, ScalarPoint]]] = {}
    run_metadata: dict[str, list[dict[str, Any]]] = {}
    for label, run_dirs in args.series:
        metrics, runs = load_series(run_dirs)
        all_series[label] = metrics
        run_metadata[label] = runs

    _write_csv(args.output_csv, all_series)
    summary: dict[str, Any] = {
        "protocol": {
            "window_size": args.window_size,
            "rolling_window": args.rolling_window,
            "checkpoints": checkpoints,
            "metric_tags": METRICS,
        },
        "series": {},
    }
    for label, metrics in all_series.items():
        summary["series"][label] = {
            "runs": run_metadata[label],
            "checkpoint_windows": {
                str(checkpoint): {
                    metric: stats
                    for metric, points in metrics.items()
                    if (stats := window_stats(points, checkpoint, args.window_size))
                    is not None
                }
                for checkpoint in checkpoints
            },
        }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot(args.output_figure, all_series, args.rolling_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
