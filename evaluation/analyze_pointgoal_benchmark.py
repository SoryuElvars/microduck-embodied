#!/usr/bin/env python3
"""Create a compact, reproducible analysis of a paired PointGoal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "week04"
    / "01_classical_pointgoal_benchmark"
    / "model_1500"
    / "full"
    / "summary"
    / "pointgoal_full_model_1500.json"
)
DEFAULT_PROTOCOL = (
    Path(__file__).parent
    / "pointgoal_protocols"
    / "week04_classical_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "week04"
    / "01_classical_pointgoal_benchmark"
    / "summaries"
    / "pointgoal_full_model_1500_processed.json"
)

CONTROLLERS = ("naive_p", "constrained")
METRIC_PREFERENCES = {
    "final_distance_m": "lower",
    "absolute_final_heading_error_rad": "lower",
    "path_length_to_arrival_m": "lower",
    "path_efficiency": "higher",
    "completion_time_s": "lower",
    "vx_saturation_fraction": "lower",
    "wz_saturation_fraction": "lower",
    "post_arrival_stop_drift_m": "lower",
    "post_arrival_max_drift_m": "lower",
    "post_arrival_max_goal_distance_m": "lower",
    "post_arrival_mean_planar_speed_mps": "lower",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile for sorted finite values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be within [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "mean": fmean(finite),
        "median": median(finite),
        "p10": percentile(finite, 0.10),
        "p25": percentile(finite, 0.25),
        "p75": percentile(finite, 0.75),
        "p90": percentile(finite, 0.90),
        "min": min(finite),
        "max": max(finite),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _episode_geometry(spec: Mapping[str, Any]) -> dict[str, float | str]:
    delta_x = float(spec["goal_x"]) - float(spec["start_x"])
    delta_y = float(spec["goal_y"]) - float(spec["start_y"])
    heading = _wrap_angle(math.atan2(delta_y, delta_x) - float(spec["start_yaw"]))
    absolute_heading = abs(heading)
    if absolute_heading <= math.pi / 4.0:
        bucket = "front_abs_heading_le_45_deg"
    elif absolute_heading <= 3.0 * math.pi / 4.0:
        bucket = "side_45_to_135_deg"
    else:
        bucket = "behind_abs_heading_gt_135_deg"
    return {
        "initial_distance_m": math.hypot(delta_x, delta_y),
        "relative_heading_rad": heading,
        "relative_heading_deg": math.degrees(heading),
        "heading_bucket": bucket,
    }


def _metric_value(episode: Mapping[str, Any], key: str) -> float | None:
    if key == "absolute_final_heading_error_rad":
        value = episode.get("final_heading_error_rad")
        return None if value is None else abs(float(value))
    value = episode.get(key)
    return None if value is None else float(value)


def _metric_values(episodes: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for episode in episodes:
        value = _metric_value(episode, key)
        if value is not None:
            values.append(value)
    return values


def _controller_analysis(
    episodes: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(episodes)
    successes = sum(bool(episode["success"]) for episode in episodes)
    reached = sum(bool(episode["reached_success_hold"]) for episode in episodes)
    outcomes = {
        "episode_count": total,
        "success_count": successes,
        "success_rate": successes / total,
        "success_rate_wilson_95": wilson_interval(successes, total),
        "reached_success_hold_count": reached,
        "fall_count": sum(bool(episode["fall"]) for episode in episodes),
        "timeout_count": sum(bool(episode["timeout"]) for episode in episodes),
        "invalid_state_count": sum(
            bool(episode["invalid_state"]) for episode in episodes
        ),
        "post_arrival_redeparture_count": sum(
            bool(episode["redeparted_after_arrival"]) for episode in episodes
        ),
    }

    failures = []
    for episode in episodes:
        if bool(episode["success"]):
            continue
        episode_id = str(episode["episode_id"])
        failures.append(
            {
                "episode_id": episode_id,
                "simulation_reset_seed": int(episode["simulation_reset_seed"]),
                "termination_reason": str(episode["termination_reason"]),
                "final_distance_m": float(episode["final_distance_m"]),
                **_episode_geometry(specs[episode_id]),
            }
        )

    heading_buckets: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        geometry = _episode_geometry(specs[str(episode["episode_id"])])
        bucket = str(geometry["heading_bucket"])
        heading_buckets.setdefault(bucket, {"episodes": []})["episodes"].append(episode)
    heading_summary = {}
    for bucket, bucket_data in heading_buckets.items():
        bucket_episodes = bucket_data["episodes"]
        bucket_successes = sum(bool(item["success"]) for item in bucket_episodes)
        heading_summary[bucket] = {
            "episode_count": len(bucket_episodes),
            "success_count": bucket_successes,
            "success_rate": bucket_successes / len(bucket_episodes),
            "timeout_count": sum(bool(item["timeout"]) for item in bucket_episodes),
            "completion_time_s_successes": describe(
                _metric_values(bucket_episodes, "completion_time_s")
            ),
        }

    return {
        "outcomes": outcomes,
        "failures": failures,
        "metrics": {
            key: describe(_metric_values(episodes, key))
            for key in METRIC_PREFERENCES
        },
        "heading_buckets": heading_summary,
    }


def _exact_mcnemar_two_sided(constrained_wins: int, naive_wins: int) -> float:
    discordant = constrained_wins + naive_wins
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(constrained_wins, naive_wins) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def _paired_analysis(
    naive_episodes: Sequence[Mapping[str, Any]],
    constrained_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    naive = {str(item["episode_id"]): item for item in naive_episodes}
    constrained = {str(item["episode_id"]): item for item in constrained_episodes}
    if set(naive) != set(constrained):
        raise ValueError("controller episode ids do not match")

    constrained_success_wins = sum(
        bool(constrained[key]["success"]) and not bool(naive[key]["success"])
        for key in naive
    )
    naive_success_wins = sum(
        bool(naive[key]["success"]) and not bool(constrained[key]["success"])
        for key in naive
    )
    metrics = {}
    for key, preference in METRIC_PREFERENCES.items():
        deltas = []
        constrained_better = 0
        naive_better = 0
        ties = 0
        for episode_id in naive:
            naive_value = _metric_value(naive[episode_id], key)
            constrained_value = _metric_value(constrained[episode_id], key)
            if naive_value is None or constrained_value is None:
                continue
            delta = float(constrained_value) - float(naive_value)
            deltas.append(delta)
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                ties += 1
            elif (delta < 0.0) == (preference == "lower"):
                constrained_better += 1
            else:
                naive_better += 1
        metrics[key] = {
            "preferred": preference,
            "delta_constrained_minus_naive": describe(deltas),
            "constrained_better_count": constrained_better,
            "naive_better_count": naive_better,
            "tie_count": ties,
        }

    return {
        "pair_count": len(naive),
        "success": {
            "constrained_wins": constrained_success_wins,
            "naive_wins": naive_success_wins,
            "ties": len(naive) - constrained_success_wins - naive_success_wins,
            "exact_mcnemar_two_sided_p": _exact_mcnemar_two_sided(
                constrained_success_wins, naive_success_wins
            ),
        },
        "metrics": metrics,
    }


def analyze(summary_path: Path, protocol_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    protocol = _load_json(protocol_path)
    if summary.get("run_mode") != "full":
        raise ValueError("analysis requires a full benchmark summary")
    if summary.get("protocol_status") != "frozen":
        raise ValueError("analysis requires a frozen protocol")
    if summary.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("summary protocol SHA does not match the current protocol")

    specs = {str(item["episode_id"]): item for item in protocol["episodes"]}
    if len(specs) != 200:
        raise ValueError("expected exactly 200 frozen EpisodeSpecs")
    controllers = summary.get("controllers", {})
    if tuple(controllers) != CONTROLLERS:
        raise ValueError(f"expected controller order {CONTROLLERS}")

    episode_sets = {
        controller_id: controllers[controller_id]["episodes"]
        for controller_id in CONTROLLERS
    }
    expected_ids = set(specs)
    for controller_id, episodes in episode_sets.items():
        ids = [str(item["episode_id"]) for item in episodes]
        if len(ids) != 200 or len(set(ids)) != 200 or set(ids) != expected_ids:
            raise ValueError(f"{controller_id} does not contain the frozen 200 episodes")

    return {
        "schema_version": 1,
        "analysis_kind": "classical_pointgoal_paired_full_v1",
        "source_summary": str(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "identity": {
            key: summary[key]
            for key in (
                "created_at",
                "protocol_id",
                "protocol_status",
                "episode_manifest_status",
                "protocol_sha256",
                "episode_manifest_sha256",
                "run_signature",
                "policy_sha256",
                "microduck_rl_commit",
            )
        },
        "integrity": {
            "frozen_episode_count": len(specs),
            "controller_order": list(CONTROLLERS),
            "paired_episode_ids_match": True,
            "protocol_file_matches_summary_sha": True,
        },
        "controllers": {
            controller_id: _controller_analysis(episode_sets[controller_id], specs)
            for controller_id in CONTROLLERS
        },
        "paired": _paired_analysis(
            episode_sets["naive_p"], episode_sets["constrained"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = args.summary.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    document = analyze(summary_path, protocol_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote processed full benchmark summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
