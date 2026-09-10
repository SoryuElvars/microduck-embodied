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
    actual_wz = values("actual_wz")
    net_yaw_deg = np.degrees(values("net_yaw_rad"))
    world_x = values("world_x")
    world_y = values("world_y")

    window_samples = max(1, round(smooth_window_s * sample_rate_hz))
    smooth_vx = centered_moving_average(actual_vx, window_samples, np)
    smooth_wz = centered_moving_average(actual_wz, window_samples, np)
    smooth_yaw_deg = centered_moving_average(net_yaw_deg, window_samples, np)
    smooth_x = centered_moving_average(world_x, window_samples, np)
    smooth_y = centered_moving_average(world_y, window_samples, np)

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
    ax.set_ylabel("Yaw rate (rad/s)")
    ax.legend(loc="best")

    ax = axes[1, 0]
    target_yaw_deg = np.degrees(cmd_wz * time_s)
    ax.plot(time_s, net_yaw_deg, color="#CAB2D6", label="Net yaw — 50 Hz raw", **raw_style)
    ax.plot(
        time_s,
        smooth_yaw_deg,
        color="#6A3D9A",
        label=f"Net yaw — {smooth_window_s:g} s mean",
        **smooth_style,
    )
    ax.plot(time_s, target_yaw_deg, label="Target integrated yaw", **target_style)
    ax.set_title("Accumulated heading change")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Net yaw (deg)")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(world_x, world_y, color="#9BD3AE", label="XY path — 50 Hz raw", **raw_style)
    ax.plot(
        smooth_x,
        smooth_y,
        color="#1B7837",
        label=f"XY path — {smooth_window_s:g} s mean",
        **smooth_style,
    )
    initial_yaw = values("yaw_rad")[0] - values("net_yaw_rad")[0]
    elapsed = time_s - time_s[0]
    target_body_x = cmd_vx * elapsed
    target_body_y = cmd_vy * elapsed
    target_world_x = (
        world_x[0]
        + np.cos(initial_yaw) * target_body_x
        - np.sin(initial_yaw) * target_body_y
    )
    target_world_y = (
        world_y[0]
        + np.sin(initial_yaw) * target_body_x
        + np.cos(initial_yaw) * target_body_y
    )
    ax.plot(target_world_x, target_world_y, label="Target straight path", **target_style)
    ax.scatter(world_x[0], world_y[0], color="#333333", s=35, label="Start", zorder=4)
    ax.scatter(world_x[-1], world_y[-1], color="#D62728", s=35, label="End", zorder=4)
    ax.set_title("World-frame XY trajectory")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "MicroDuck model_5999 — straight command "
        f"(final yaw {net_yaw_deg[-1]:+.1f} deg, mean vx {actual_vx.mean():.3f} m/s)"
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
