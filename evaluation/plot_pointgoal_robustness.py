#!/usr/bin/env python3
"""Plot per-factor PointGoal robustness sensitivity from a suite summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


FACTOR_TITLES = {
    "foot_friction": "Ground friction",
    "actuator_delay": "Actuator delay",
    "motor_strength_proxy": "BAM voltage proxy",
    "backlash": "Backlash",
}


def load_factor_series(path: Path) -> dict[str, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("benchmark_kind") != "classical_pointgoal_onnx_bam_ood":
        raise ValueError("input is not a PointGoal robustness summary")
    conditions = document.get("conditions", [])
    nominal = next(
        (
            item
            for item in conditions
            if item["condition"]["factor"] == "nominal"
        ),
        None,
    )
    if nominal is None:
        raise ValueError("robustness summary has no nominal condition")

    result: dict[str, list[dict[str, Any]]] = {}
    for factor in FACTOR_TITLES:
        items = [nominal] + [
            item
            for item in conditions
            if item["condition"]["factor"] == factor
        ]
        rows: list[dict[str, Any]] = []
        for item in items:
            condition = item["condition"]
            aggregate = item["aggregate"]
            if condition["factor"] == "nominal":
                x_value: float | str = {
                    "foot_friction": 1.0,
                    "actuator_delay": 0.0,
                    "motor_strength_proxy": 100.0,
                    "backlash": "Normal",
                }[factor]
            elif factor == "foot_friction":
                x_value = float(condition["foot_friction"])
            elif factor == "actuator_delay":
                x_value = float(condition["actuator_delay_ms"])
            elif factor == "motor_strength_proxy":
                x_value = 100.0 * float(condition["bam_voltage_scale"])
            else:
                x_value = "2 deg"
            rows.append(
                {
                    "x": x_value,
                    "condition_id": condition["condition_id"],
                    "success_rate": float(aggregate["success_rate"]),
                    "fall_rate": float(aggregate["fall_rate"]),
                    "median_final_distance_m": float(
                        aggregate["median_final_distance_m"]
                    ),
                }
            )
        if factor != "backlash":
            rows.sort(key=lambda row: float(row["x"]))
        result[factor] = rows
    return result


def plot_factor_series(
    series: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
    factors: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    selected_factors = tuple(factors or FACTOR_TITLES)
    if len(selected_factors) == 1:
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axes = [axis]
    else:
        figure, axes_grid = plt.subplots(
            2, 2, figsize=(12, 8), constrained_layout=True
        )
        axes = list(axes_grid.flat)
    for axis, factor in zip(axes, selected_factors):
        rows = series[factor]
        labels = [
            f"{float(row['x']):g}" if isinstance(row["x"], (int, float)) else str(row["x"])
            for row in rows
        ]
        positions = list(range(len(rows)))
        success = [100.0 * row["success_rate"] for row in rows]
        fall = [100.0 * row["fall_rate"] for row in rows]
        axis.plot(
            positions,
            success,
            marker="o",
            color="#2878b5",
            label="Success rate",
        )
        axis.plot(
            positions,
            fall,
            marker="s",
            color="#d95319",
            label="Fall rate",
        )
        axis.set_ylim(-2, 102)
        axis.set_ylabel("Episode rate (%)")
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="center left", fontsize=8)
        axis.set_title(FACTOR_TITLES[factor])
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="MicroDuck PointGoal OOD sensitivity")
    parser.add_argument("--factor", choices=tuple(FACTOR_TITLES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    series = load_factor_series(args.summary.expanduser().resolve())
    output = args.output.expanduser().resolve()
    factors = None if args.factor is None else (args.factor,)
    plot_factor_series(series, output, args.title, factors)
    print(f"figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
