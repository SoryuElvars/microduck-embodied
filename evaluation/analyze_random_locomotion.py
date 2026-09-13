#!/usr/bin/env python3
"""Validate and summarize the random-command locomotion benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
SUCCESS_RELATIVE_TOLERANCE = 0.30
SUCCESS_ABSOLUTE_FLOORS = {
    "vx": 0.03,
    "vy": 0.03,
    "wz": 0.10,
}
AXIS_METRICS = {"vx": "rmse_vx", "vy": "rmse_vy", "wz": "rmse_wz"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def describe(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot describe an empty sequence")

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": fmean(ordered),
        "median": median(ordered),
        "std": pstdev(ordered),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def linear_response(
    episodes: list[dict[str, Any]],
    command_key: str,
    actual_key: str,
) -> dict[str, float]:
    targets = [float(item["command"][command_key]) for item in episodes]
    actuals = [float(item[actual_key]) for item in episodes]
    mean_target = fmean(targets)
    mean_actual = fmean(actuals)
    target_variance = fmean((value - mean_target) ** 2 for value in targets)
    actual_variance = fmean((value - mean_actual) ** 2 for value in actuals)
    covariance = fmean(
        (target - mean_target) * (actual - mean_actual)
        for target, actual in zip(targets, actuals, strict=True)
    )
    slope = covariance / target_variance if target_variance else 0.0
    correlation = (
        covariance / math.sqrt(target_variance * actual_variance)
        if target_variance and actual_variance
        else 0.0
    )
    return {
        "slope": slope,
        "intercept": mean_actual - slope * mean_target,
        "correlation": correlation,
        "episode_mean_absolute_error": fmean(
            abs(actual - target)
            for target, actual in zip(targets, actuals, strict=True)
        ),
    }


def success_tolerances(episode: dict[str, Any]) -> dict[str, float]:
    return {
        axis: max(
            floor,
            SUCCESS_RELATIVE_TOLERANCE * abs(float(episode["command"][axis])),
        )
        for axis, floor in SUCCESS_ABSOLUTE_FLOORS.items()
    }


def episode_passes(episode: dict[str, Any]) -> bool:
    tolerances = success_tolerances(episode)
    return not episode["fallen"] and all(
        float(episode[metric]) <= tolerances[axis]
        for axis, metric in AXIS_METRICS.items()
    )


def normalized_tracking_score(episode: dict[str, Any]) -> float:
    """Return lower-is-better tracking error normalized by success thresholds."""

    tolerances = success_tolerances(episode)
    return fmean(
        float(episode[metric]) / tolerances[axis]
        for axis, metric in AXIS_METRICS.items()
    )


def compact_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode": int(episode["episode"]),
        "seed": int(episode["seed"]),
        "bucket": str(episode["bucket"]),
        "command": episode["command"],
        "normalized_tracking_score": normalized_tracking_score(episode),
        "nominal_success": episode_passes(episode),
        "rmse_vx": float(episode["rmse_vx"]),
        "rmse_vy": float(episode["rmse_vy"]),
        "rmse_wz": float(episode["rmse_wz"]),
        "net_yaw_deg": float(episode["net_yaw_deg"]),
        "raw_steps_csv": str(episode["raw_steps_csv"]),
    }


def summarize_group(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot summarize an empty episode group")
    return {
        "episode_count": len(episodes),
        "fall_rate": fmean(float(item["fallen"]) for item in episodes),
        "nominal_success_rate": fmean(float(episode_passes(item)) for item in episodes),
        "axis_pass_rate": {
            metric: fmean(
                float(float(item[metric]) <= success_tolerances(item)[axis])
                for item in episodes
            )
            for axis, metric in AXIS_METRICS.items()
        },
        "rmse_vx": describe([float(item["rmse_vx"]) for item in episodes]),
        "rmse_vy": describe([float(item["rmse_vy"]) for item in episodes]),
        "rmse_wz": describe([float(item["rmse_wz"]) for item in episodes]),
        "net_yaw_deg": describe([float(item["net_yaw_deg"]) for item in episodes]),
    }


def validate_raw_csvs(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    expected_header: list[str] | None = None
    total_rows = 0
    nonfinite_values = 0
    missing_files: list[str] = []
    row_count_mismatches: list[str] = []

    for episode in episodes:
        relative_path = str(episode["raw_steps_csv"])
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            missing_files.append(relative_path)
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            if header is None:
                raise ValueError(f"CSV has no header: {path}")
            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise ValueError(f"CSV header differs: {path}")

            rows = 0
            for row in reader:
                rows += 1
                for key, value in row.items():
                    if key in {"phase", "fallen"}:
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        nonfinite_values += 1
                    else:
                        if not math.isfinite(numeric):
                            nonfinite_values += 1
            total_rows += rows
            warmup_steps = round(
                float(episode.get("warmup_s", 1.0))
                / (0.005 * 4)
            )
            expected_rows = warmup_steps + int(episode["steps_completed"])
            if rows != expected_rows:
                row_count_mismatches.append(relative_path)

    return {
        "csv_count": len(episodes) - len(missing_files),
        "column_count": len(expected_header or []),
        "total_rows": total_rows,
        "missing_file_count": len(missing_files),
        "row_count_mismatch_count": len(row_count_mismatches),
        "nonfinite_numeric_value_count": nonfinite_values,
        "missing_files": missing_files,
        "row_count_mismatches": row_count_mismatches,
    }


def process_summary(document: dict[str, Any], *, validate_csv: bool = True) -> dict[str, Any]:
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("benchmark summary has no episodes")
    if len(episodes) != int(document["episode_count"]):
        raise ValueError("episode_count does not match episode records")

    indices = [int(item["episode"]) for item in episodes]
    seeds = [int(item["seed"]) for item in episodes]
    if len(indices) != len(set(indices)) or len(seeds) != len(set(seeds)):
        raise ValueError("episode indices and reset seeds must be unique")

    buckets = Counter(str(item["bucket"]) for item in episodes)
    motion_episodes = [item for item in episodes if item["bucket"] != "standing"]
    by_bucket = {
        bucket: summarize_group(
            [item for item in episodes if item["bucket"] == bucket]
        )
        for bucket in sorted(buckets)
    }
    turns = [item for item in episodes if item["bucket"] == "turn_in_place"]
    turn_direction = {
        direction: summarize_group(group)
        | {
            "mean_command_wz": fmean(float(item["command"]["wz"]) for item in group),
            "mean_actual_wz": fmean(float(item["mean_actual_wz"]) for item in group),
        }
        for direction, group in {
            "negative": [item for item in turns if float(item["command"]["wz"]) < 0],
            "positive": [item for item in turns if float(item["command"]["wz"]) > 0],
        }.items()
    }

    finite_summary_values = all(
        math.isfinite(float(item[key]))
        for item in episodes
        for key in (
            "rmse_vx",
            "rmse_vy",
            "rmse_wz",
            "mean_actual_vx",
            "mean_actual_vy",
            "mean_actual_wz",
            "net_yaw_deg",
            "forward_distance_m",
            "lateral_displacement_m",
        )
    )
    successful_motion = sorted(
        (item for item in motion_episodes if episode_passes(item)),
        key=normalized_tracking_score,
    )
    failed_motion = sorted(
        (item for item in motion_episodes if not episode_passes(item)),
        key=normalized_tracking_score,
    )
    middle_failure = len(failed_motion) // 2
    result = {
        "schema_version": 1,
        "source_benchmark_kind": document["benchmark_kind"],
        "source_policy_sha256": document["policy_sha256"],
        "episode_count": len(episodes),
        "bucket_counts": dict(sorted(buckets.items())),
        "success_definition": {
            "name": "project_nominal_tracking_success",
            "official_metric": False,
            "requirements": {
                "fallen": False,
                "relative_rmse_tolerance": SUCCESS_RELATIVE_TOLERANCE,
                "absolute_rmse_floors": SUCCESS_ABSOLUTE_FLOORS,
            },
            "rationale": (
                "Each axis must stay within 30% of its command magnitude, with "
                "small absolute floors for zero and near-zero commands."
            ),
        },
        "data_quality": {
            "unique_episode_indices": len(set(indices)),
            "unique_reset_seeds": len(set(seeds)),
            "finite_summary_values": finite_summary_values,
        },
        "overall": summarize_group(episodes),
        "motion_only": summarize_group(motion_episodes),
        "by_bucket": by_bucket,
        "turn_in_place_by_direction": turn_direction,
        "response_fit": {
            "vx": linear_response(episodes, "vx", "mean_actual_vx"),
            "vy": linear_response(episodes, "vy", "mean_actual_vy"),
            "wz": linear_response(episodes, "wz", "mean_actual_wz"),
        },
        "representative_motion_episodes": {
            "best_success": (
                compact_episode(successful_motion[0]) if successful_motion else None
            ),
            "typical_failure": compact_episode(failed_motion[middle_failure]),
            "worst_failure": compact_episode(failed_motion[-1]),
        },
    }
    if validate_csv:
        result["data_quality"]["raw_csv"] = validate_raw_csvs(episodes)
    return result


def plot_results(document: dict[str, Any], processed: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = document["episodes"]
    bucket_colors = {
        "general": "#1565C0",
        "standing": "#616161",
        "turn_in_place": "#7B1FA2",
    }
    axes_config = (
        ("vx", "mean_actual_vx", "Linear x velocity (m/s)"),
        ("vy", "mean_actual_vy", "Linear y velocity (m/s)"),
        ("wz", "mean_actual_wz", "Yaw velocity (rad/s)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (command_key, actual_key, title) in zip(
        axes.flat[:3], axes_config, strict=True
    ):
        targets = [float(item["command"][command_key]) for item in episodes]
        actuals = [float(item[actual_key]) for item in episodes]
        for bucket in ("general", "standing", "turn_in_place"):
            points = [item for item in episodes if item["bucket"] == bucket]
            ax.scatter(
                [float(item["command"][command_key]) for item in points],
                [float(item[actual_key]) for item in points],
                s=18,
                alpha=0.55,
                color=bucket_colors[bucket],
                label=bucket.replace("_", " "),
            )
        lower = min(targets + actuals)
        upper = max(targets + actuals)
        padding = max((upper - lower) * 0.08, 0.02)
        limits = (lower - padding, upper + padding)
        ax.plot(limits, limits, color="#212121", linestyle="--", linewidth=1.2)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("Command")
        ax.set_ylabel("Episode mean actual")
        ax.grid(alpha=0.2)

    axes[0, 0].legend(loc="best", fontsize=8)
    ax = axes[1, 1]
    bucket_order = ["standing", "turn_in_place", "general", "motion_only", "overall"]
    rates = [
        processed["by_bucket"][bucket]["nominal_success_rate"]
        if bucket in processed["by_bucket"]
        else processed[bucket]["nominal_success_rate"]
        for bucket in bucket_order
    ]
    bars = ax.bar(
        [name.replace("_", " ") for name in bucket_order],
        rates,
        color=[
            bucket_colors["standing"],
            bucket_colors["turn_in_place"],
            bucket_colors["general"],
            "#EF6C00",
            "#2E7D32",
        ],
    )
    ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=3)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Nominal success rate")
    ax.set_title("Project-defined tracking success")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        f"MicroDuck random velocity benchmark: {len(episodes)} episodes, "
        f"fall rate {processed['overall']['fall_rate']:.1%}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    document: dict[str, Any],
    processed: dict[str, Any],
    source_path: Path,
    output: Path,
) -> None:
    overall = processed["overall"]
    motion = processed["motion_only"]
    raw = processed["data_quality"].get("raw_csv", {})
    turns = processed["turn_in_place_by_direction"]
    fits = processed["response_fit"]
    representatives = processed["representative_motion_episodes"]
    typical = representatives["typical_failure"]
    worst = representatives["worst_failure"]
    bucket_rows = "\n".join(
        f"| {bucket} | {summary['episode_count']} | "
        f"{summary['nominal_success_rate']:.1%} | {summary['fall_rate']:.1%} | "
        f"{summary['rmse_vx']['mean']:.3f} | {summary['rmse_vy']['mean']:.3f} | "
        f"{summary['rmse_wz']['mean']:.3f} |"
        for bucket, summary in processed["by_bucket"].items()
    )
    report = f"""# MicroDuck 400-Episode 随机速度基准

日期：2026-09-11

## 协议

- 模型：`{Path(document['policy_path']).name}`
- Episode：{processed['episode_count']}
- Reset seeds：{document['protocol']['reset_seed']}–{document['protocol']['reset_seed'] + processed['episode_count'] - 1}
- Command seed：{document['protocol']['command_seed']}
- 指令范围：`vx [-0.4, 0.4]`、`vy [-0.3, 0.3]`、`wz [-1.0, 1.0]`
- 显式评测配额：25% 静止、15% 原地转向、60% 全范围均匀采样
- 该配额来自最终训练配置中的 standing/turn buckets，但为便于分桶统计而设为互斥；不是对训练采样器执行顺序的逐项复刻
- 每个 Episode：1 秒零指令预热 + 10 秒测试，官方 reset，CPU MuJoCo + BAM M6
- 部署形态 ONNX 评测不加载 Reward Manager，因此 `Episode Return` 不可用

## 与固定八指令矩阵的关系

此前的固定矩阵包含 8 条轴向隔离指令 × 20 seeds，共 160 个 Episode；它适合
诊断前进、后退、横移和转向的单项缺陷。本次 400 个 Episode 覆盖同时出现的
vx/vy/wz 组合以及显式 standing/turn buckets。两套协议互补，结果分别报告，
不合并为一个成功率，也不把重复子集再次计数。

## 数据完整性

- 汇总 Episode：{processed['episode_count']}
- 原始 CSV：{raw.get('csv_count', 0)}
- CSV 总行数：{raw.get('total_rows', 0)}
- 每个 CSV 字段数：{raw.get('column_count', 0)}
- 缺失文件：{raw.get('missing_file_count', 0)}
- 行数不一致：{raw.get('row_count_mismatch_count', 0)}
- 非有限数值：{raw.get('nonfinite_numeric_value_count', 0)}

## 项目定义的 Nominal Success

这不是上游官方指标。本项目把“未跌倒，且每个轴的 RMSE 不超过该轴命令幅值
的 30%”定义为一次 nominal success。为避免零命令被数值噪声误判，vx/vy/wz
分别使用 0.03 m/s、0.03 m/s、0.10 rad/s 的最小容差。这个混合门槛也能避免
把低速指令完全不响应错误地计为成功。

- 运动指令 Nominal Success Rate：**{motion['nominal_success_rate']:.1%}**（排除静止 Episode）
- 全部 Episode Nominal Success Rate：{overall['nominal_success_rate']:.1%}（包含 100 个静止 Episode）
- Fall Rate：**{overall['fall_rate']:.1%}**
- vx / vy / wz 单轴通过率：{overall['axis_pass_rate']['rmse_vx']:.1%} / {overall['axis_pass_rate']['rmse_vy']:.1%} / {overall['axis_pass_rate']['rmse_wz']:.1%}

## 分桶结果

| Bucket | Episodes | Success Rate | Fall Rate | Mean RMSE vx | Mean RMSE vy | Mean RMSE wz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{bucket_rows}

## 指令响应

| 轴 | 响应斜率 | 相关系数 | Episode 均值绝对误差 |
| --- | ---: | ---: | ---: |
| vx | {fits['vx']['slope']:.3f} | {fits['vx']['correlation']:.3f} | {fits['vx']['episode_mean_absolute_error']:.3f} m/s |
| vy | {fits['vy']['slope']:.3f} | {fits['vy']['correlation']:.3f} | {fits['vy']['episode_mean_absolute_error']:.3f} m/s |
| wz | {fits['wz']['slope']:.3f} | {fits['wz']['correlation']:.3f} | {fits['wz']['episode_mean_absolute_error']:.3f} rad/s |

原地转向继续呈现方向和幅值相关的非线性：

- 正向：命令均值 {turns['positive']['mean_command_wz']:+.3f}，实际均值 {turns['positive']['mean_actual_wz']:+.3f} rad/s，成功率 {turns['positive']['nominal_success_rate']:.1%}
- 负向：命令均值 {turns['negative']['mean_command_wz']:+.3f}，实际均值 {turns['negative']['mean_actual_wz']:+.3f} rad/s，成功率 {turns['negative']['nominal_success_rate']:.1%}

静止 Episode 全部通过，但 300 个运动指令仅 {motion['nominal_success_rate']:.1%} 通过；因此不能用包含静止样本的 {overall['nominal_success_rate']:.1%} 作为控制器质量结论。

## 代表性运动样本

- 合格成功样本：无。
- 典型失败：Episode {typical['episode']}，seed {typical['seed']}，命令 `({typical['command']['vx']:+.3f}, {typical['command']['vy']:+.3f}, {typical['command']['wz']:+.3f})`，RMSE `({typical['rmse_vx']:.3f}, {typical['rmse_vy']:.3f}, {typical['rmse_wz']:.3f})`。
- 最严重失败：Episode {worst['episode']}，seed {worst['seed']}，命令 `({worst['command']['vx']:+.3f}, {worst['command']['vy']:+.3f}, {worst['command']['wz']:+.3f})`，RMSE `({worst['rmse_vx']:.3f}, {worst['rmse_vy']:.3f}, {worst['rmse_wz']:.3f})`。

## 结论

该模型在 400 个 Episode 中保持 0% Fall Rate，说明 nominal 稳定性较好；但
300 个运动指令的成功率为 0%，速度跟踪门禁明确失败。该结果完成第二周
Locomotion Benchmark 的随机指令基线，但该 ONNX 不能冻结为导航控制器。

源汇总：`{source_path.relative_to(PROJECT_ROOT)}`
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a random-command locomotion benchmark."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--skip-csv-validation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = args.input.expanduser().resolve()
    document = load_json(source_path)
    processed = process_summary(
        document,
        validate_csv=not args.skip_csv_validation,
    )
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(processed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_results(document, processed, args.output_figure.expanduser().resolve())
    write_report(
        document,
        processed,
        source_path,
        args.output_report.expanduser().resolve(),
    )
    print(json.dumps(processed["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
