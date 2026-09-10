#!/usr/bin/env python3
"""Plot an aggregate diagnostic figure for a locomotion episode batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_RAW_COLUMNS = {
    "phase",
    "time_s",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "world_x",
    "world_y",
    "yaw_rad",
    "net_yaw_rad",
}
PROJECT_ROOT = Path(__file__).parents[1]


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("summary must contain a non-empty episodes list")
    return summary


def load_test_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_RAW_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = [row for row in reader if row["phase"] == "test"]
    if not rows:
        raise ValueError(f"{path} contains no phase=test rows")
    return rows


def centered_moving_average(values, window_samples: int, np):
    if window_samples <= 1:
        return values.copy()
    window_samples = min(window_samples, len(values))
    left = (window_samples - 1) // 2
    right = window_samples // 2
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.convolve(padded, kernel, mode="valid")


def aligned_episode_arrays(rows: list[dict[str, str]], window_samples: int, np):
    def values(column: str):
        return np.asarray([float(row[column]) for row in rows], dtype=float)

    time_s = values("time_s")
    net_yaw_rad = values("net_yaw_rad")
    initial_yaw = values("yaw_rad")[0] - net_yaw_rad[0]
    world_x = values("world_x")
    world_y = values("world_y")
    dx = world_x - world_x[0]
    dy = world_y - world_y[0]
    aligned_x = np.cos(initial_yaw) * dx + np.sin(initial_yaw) * dy
    aligned_y = -np.sin(initial_yaw) * dx + np.cos(initial_yaw) * dy

    return {
        "time_s": time_s,
        "net_yaw_deg": np.degrees(net_yaw_rad),
        "aligned_x": centered_moving_average(aligned_x, window_samples, np),
        "aligned_y": centered_moving_average(aligned_y, window_samples, np),
        "cmd_vx": values("cmd_vx"),
        "cmd_vy": values("cmd_vy"),
        "cmd_wz": values("cmd_wz"),
    }


def plot_batch(
    summary_path: Path,
    output_path: Path,
    smooth_window_s: float,
    sample_rate_hz: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = load_summary(summary_path)
    episode_summaries = summary["episodes"]
    window_samples = max(1, round(smooth_window_s * sample_rate_hz))

    batches = []
    for episode in episode_summaries:
        raw_path = PROJECT_ROOT / episode["raw_steps_csv"]
        batches.append(aligned_episode_arrays(load_test_rows(raw_path), window_samples, np))

    common_length = min(len(batch["time_s"]) for batch in batches)
    time_s = batches[0]["time_s"][:common_length]
    yaw_matrix = np.stack([batch["net_yaw_deg"][:common_length] for batch in batches])
    aligned_x_matrix = np.stack([batch["aligned_x"][:common_length] for batch in batches])
    aligned_y_matrix = np.stack([batch["aligned_y"][:common_length] for batch in batches])

    mean_yaw = yaw_matrix.mean(axis=0)
    yaw_p10, yaw_p90 = np.percentile(yaw_matrix, (10, 90), axis=0)
    mean_x = aligned_x_matrix.mean(axis=0)
    mean_y = aligned_y_matrix.mean(axis=0)

    final_yaw = np.asarray([episode["net_yaw_deg"] for episode in episode_summaries])
    mean_vx = np.asarray([episode["mean_actual_vx"] for episode in episode_summaries])
    mean_wz = np.asarray([episode["mean_actual_wz"] for episode in episode_summaries])
    initial_z_mm = np.asarray(
        [episode["initial_state"]["z"] * 1000.0 for episode in episode_summaries]
    )

    command = summary["command"]
    cmd_vx = float(command["vx"])
    cmd_vy = float(command["vy"])
    cmd_wz = float(command["wz"])
    elapsed = time_s - time_s[0]
    target_yaw = np.degrees(cmd_wz * elapsed)
    target_x = cmd_vx * elapsed
    target_y = cmd_vy * elapsed

    episode_style = {"linewidth": 0.8, "alpha": 0.22}
    mean_style = {"linewidth": 2.7, "alpha": 0.95}
    target_style = {"linewidth": 1.7, "linestyle": "--", "color": "#333333"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    for yaw in yaw_matrix:
        ax.plot(time_s, yaw, color="#B39DDB", **episode_style)
    ax.fill_between(time_s, yaw_p10, yaw_p90, color="#9575CD", alpha=0.18, label="P10–P90")
    ax.plot(time_s, mean_yaw, color="#5E35B1", label="20-episode mean", **mean_style)
    ax.plot(time_s, target_yaw, label="Target integrated yaw", **target_style)
    ax.set_title("Accumulated heading across episodes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Net yaw (deg)")
    ax.legend(loc="best")

    ax = axes[0, 1]
    for aligned_x, aligned_y in zip(aligned_x_matrix, aligned_y_matrix, strict=True):
        ax.plot(aligned_x, aligned_y, color="#81C784", **episode_style)
    ax.plot(mean_x, mean_y, color="#1B7837", label="20-episode mean", **mean_style)
    ax.plot(target_x, target_y, label="Target straight path", **target_style)
    ax.scatter(0.0, 0.0, color="#333333", s=35, label="Aligned start", zorder=4)
    ax.set_title(f"Initial-body-frame XY paths ({smooth_window_s:g} s mean)")
    ax.set_xlabel("Forward displacement (m)")
    ax.set_ylabel("Lateral displacement (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    ax = axes[1, 0]
    box = ax.boxplot(
        final_yaw,
        positions=[1.0],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    box["boxes"][0].set(facecolor="#D1C4E9", edgecolor="#5E35B1")
    rng = np.random.default_rng(42)
    jitter = rng.normal(1.0, 0.035, size=len(final_yaw))
    ax.scatter(jitter, final_yaw, color="#5E35B1", alpha=0.75, s=28, label="Episode")
    ax.axhline(0.0, label="Target final yaw", **target_style)
    ax.axhline(final_yaw.mean(), color="#C0392B", linewidth=2.0, label="Batch mean")
    ax.set_xlim(0.65, 1.35)
    ax.set_xticks([])
    ax.set_title("Final net-yaw distribution")
    ax.set_ylabel("Final net yaw (deg)")
    ax.legend(loc="best")

    ax = axes[1, 1]
    scatter = ax.scatter(
        mean_vx,
        mean_wz,
        c=initial_z_mm,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.5,
        s=58,
    )
    ax.scatter(cmd_vx, cmd_wz, marker="*", s=180, color="#333333", label="Target")
    ax.axhline(cmd_wz, **target_style)
    ax.axvline(cmd_vx, **target_style)
    ax.set_title("Episode-mean forward speed vs yaw rate")
    ax.set_xlabel("Mean actual vx (m/s)")
    ax.set_ylabel("Mean actual wz (rad/s)")
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Initial trunk z (mm)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    positive_count = summary["aggregate"]["yaw_direction_counts"]["positive"]
    fig.suptitle(
        f"MicroDuck model_5999 — {len(episode_summaries)} official-reset episodes "
        f"({positive_count}/{len(episode_summaries)} positive yaw, "
        f"mean {final_yaw.mean():+.1f} deg)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(episode_summaries)} episodes from {summary_path}")
    print(f"Common plotted length: {common_length} samples")
    print(f"Smoothing: {smooth_window_s:g} s ({window_samples} samples)")
    print(f"Saved figure: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a MicroDuck locomotion batch summary")
    parser.add_argument("--summary", type=Path, required=True, help="Batch summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG")
    parser.add_argument("--smooth-window-s", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smooth_window_s <= 0:
        raise ValueError("--smooth-window-s must be > 0")
    if args.sample_rate_hz <= 0:
        raise ValueError("--sample-rate-hz must be > 0")
    plot_batch(
        args.summary.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.smooth_window_s,
        args.sample_rate_hz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
