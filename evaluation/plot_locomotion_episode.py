#!/usr/bin/env python3
"""Plot one headless MicroDuck locomotion episode from its raw CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


REQUIRED_COLUMNS = {
    "phase",
    "time_s",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "actual_vx",
    "actual_vy",
    "actual_wz",
    "world_x",
    "world_y",
    "yaw_rad",
    "net_yaw_rad",
}


def load_test_rows(path: Path) -> list[dict[str, str]]:
    """Load the command phase and reject incompatible benchmark CSV files."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = [row for row in reader if row["phase"] == "test"]

    if not rows:
        raise ValueError("CSV contains no phase=test rows")
    return rows


def centered_moving_average(values, window_samples: int, np):
    """Return a same-length centered mean without zero-padding edge artifacts."""

    if window_samples <= 1:
        return values.copy()
    window_samples = min(window_samples, len(values))
    left = (window_samples - 1) // 2
    right = window_samples // 2
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.convolve(padded, kernel, mode="valid")


def plot_episode(
    input_path: Path,
    output_path: Path,
    smooth_window_s: float,
    sample_rate_hz: float,
    title: str | None = None,
) -> None:
    """Generate the four-panel locomotion diagnostic figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = load_test_rows(input_path)

    def values(column: str):
        return np.asarray([float(row[column]) for row in rows], dtype=float)

    time_s = values("time_s")
    cmd_vx = values("cmd_vx")
    cmd_vy = values("cmd_vy")
    cmd_wz = values("cmd_wz")
    actual_vx = values("actual_vx")
    actual_vy = values("actual_vy")
    actual_wz = values("actual_wz")
    world_x = values("world_x")
    world_y = values("world_y")
    yaw_rad = values("yaw_rad")
    net_yaw_rad = values("net_yaw_rad")
    net_yaw_deg = np.degrees(net_yaw_rad)

    window_samples = max(1, round(smooth_window_s * sample_rate_hz))
    smooth_vx = centered_moving_average(actual_vx, window_samples, np)
    smooth_vy = centered_moving_average(actual_vy, window_samples, np)
    smooth_wz = centered_moving_average(actual_wz, window_samples, np)

    # Align every trajectory with its initial body frame so randomized reset
    # positions and headings do not change the geometric interpretation.
    initial_yaw = yaw_rad[0] - net_yaw_rad[0]
    dx = world_x - world_x[0]
    dy = world_y - world_y[0]
    body_x = np.cos(initial_yaw) * dx + np.sin(initial_yaw) * dy
    body_y = -np.sin(initial_yaw) * dx + np.cos(initial_yaw) * dy
    smooth_x = centered_moving_average(body_x, window_samples, np)
    smooth_y = centered_moving_average(body_y, window_samples, np)

    # Integrate the body-frame velocity command while its target heading turns.
    # The midpoint rule handles both straight and combined vx/vy/wz commands.
    elapsed = time_s - time_s[0]
    target_heading = cmd_wz * elapsed
    target_x = np.zeros_like(elapsed)
    target_y = np.zeros_like(elapsed)
    if len(elapsed) > 1:
        dt = np.diff(elapsed)
        mid_heading = 0.5 * (target_heading[:-1] + target_heading[1:])
        target_world_vx = cmd_vx[:-1] * np.cos(mid_heading) - cmd_vy[:-1] * np.sin(mid_heading)
        target_world_vy = cmd_vx[:-1] * np.sin(mid_heading) + cmd_vy[:-1] * np.cos(mid_heading)
        target_x[1:] = np.cumsum(target_world_vx * dt)
        target_y[1:] = np.cumsum(target_world_vy * dt)

    raw_style = {"linewidth": 0.7, "alpha": 0.25}
    smooth_style = {"linewidth": 2.4, "alpha": 0.95}
    target_style = {"linewidth": 1.7, "linestyle": "--", "color": "#333333"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(time_s, actual_vx, color="#72B7E2", label="Actual vx — 50 Hz raw", **raw_style)
    ax.plot(
        time_s,
        smooth_vx,
        color="#006BA4",
        label=f"Actual vx — {smooth_window_s:g} s mean",
        **smooth_style,
    )
    ax.plot(time_s, cmd_vx, label="Target vx", **target_style)
    ax.set_title("Forward velocity tracking")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(time_s, actual_vy, color="#A6CEE3", label="Actual vy — 50 Hz raw", **raw_style)
    ax.plot(
        time_s,
        smooth_vy,
        color="#1F78B4",
        label=f"Actual vy — {smooth_window_s:g} s mean",
        **smooth_style,
    )
    ax.plot(time_s, cmd_vy, label="Target vy", **target_style)
    ax.set_title("Lateral velocity tracking")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(time_s, actual_wz, color="#F4A582", label="Actual wz — 50 Hz raw", **raw_style)
    ax.plot(
        time_s,
        smooth_wz,
        color="#C0392B",
        label=f"Actual wz — {smooth_window_s:g} s mean",
        **smooth_style,
    )
    ax.plot(time_s, cmd_wz, label="Target wz", **target_style)
    ax.set_title("Yaw-rate tracking")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw rate (rad/s)")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(body_x, body_y, color="#9BD3AE", label="Actual XY — 50 Hz raw", **raw_style)
    ax.plot(
        smooth_x,
        smooth_y,
        color="#1B7837",
        label=f"Actual XY — {smooth_window_s:g} s mean",
        **smooth_style,
    )
    ax.plot(target_x, target_y, label="Target XY", **target_style)
    ax.scatter(body_x[0], body_y[0], color="#333333", s=35, label="Start", zorder=4)
    ax.scatter(body_x[-1], body_y[-1], color="#D62728", s=35, label="End", zorder=4)
    ax.set_title("Initial-body-frame XY trajectory")
    ax.set_xlabel("Forward X (m)")
    ax.set_ylabel("Left Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    figure_title = title or "MicroDuck model_5999 — command-tracking episode"
    fig.suptitle(
        f"{figure_title}\n"
        f"command=({cmd_vx.mean():+.3f}, {cmd_vy.mean():+.3f}, {cmd_wz.mean():+.3f}), "
        f"actual mean=({actual_vx.mean():+.3f}, {actual_vy.mean():+.3f}, {actual_wz.mean():+.3f}), "
        f"final yaw={net_yaw_deg[-1]:+.1f}°"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(rows)} test samples from {input_path}")
    print(f"Smoothing: {smooth_window_s:g} s ({window_samples} samples at {sample_rate_hz:g} Hz)")
    print(f"Saved figure: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one MicroDuck locomotion episode CSV")
    parser.add_argument("--input", type=Path, required=True, help="Raw episode CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG")
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=1.0,
        help="Centered moving-average window in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=50.0,
        help="CSV control sample rate (default: %(default)s)",
    )
    parser.add_argument("--title", help="Optional first line for the figure title")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smooth_window_s <= 0:
        raise ValueError("--smooth-window-s must be > 0")
    if args.sample_rate_hz <= 0:
        raise ValueError("--sample-rate-hz must be > 0")
    plot_episode(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.smooth_window_s,
        args.sample_rate_hz,
        args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
