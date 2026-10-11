#!/usr/bin/env python3
"""Diagnostically re-score recorded RL trajectories with a stop-aware goal gate.

This does not change the frozen Week-04 benchmark or continue an episode beyond
its recorded two-second post-arrival observation. A missing hit is therefore
"not observed", not a counterfactual full-episode failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.pointgoal_protocol import load_pointgoal_protocol
from evaluation.pointgoal_benchmark import select_episode_specs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class StableArrivalGate:
    """Mirror the training-side sampled motion-window success rule."""

    radius_m: float
    hold_s: float
    dt_s: float
    max_planar_speed_mps: float
    max_abs_yaw_rate_radps: float
    stable_steps: int = 0
    speeds: deque[float] = field(init=False)
    yaw_rates: deque[float] = field(init=False)
    window_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.window_steps = max(1, math.ceil((self.hold_s - 1e-12) / self.dt_s))
        self.speeds = deque(maxlen=self.window_steps)
        self.yaw_rates = deque(maxlen=self.window_steps)

    def update(self, distance_m: float, vx: float, vy: float, wz: float) -> bool:
        self.speeds.append(math.hypot(vx, vy))
        self.yaw_rates.append(wz)
        stable_now = (
            len(self.speeds) == self.window_steps
            and distance_m <= self.radius_m
            and sum(self.speeds) / self.window_steps <= self.max_planar_speed_mps
            and abs(sum(self.yaw_rates) / self.window_steps) <= self.max_abs_yaw_rate_radps
        )
        self.stable_steps = self.stable_steps + 1 if stable_now else 0
        return self.stable_steps >= self.window_steps


def rescore_rows(
    rows: Sequence[dict[str, str]],
    *,
    k: int,
    radius_m: float,
    hold_s: float,
    max_planar_speed_mps: float,
    max_abs_yaw_rate_radps: float,
) -> dict[str, object]:
    if k not in (2, 5, 10) or not rows:
        raise ValueError("expected K=2/5/10 and a nonempty trajectory")
    navigation = [row for row in rows if row["phase"] in ("navigate", "post_arrival")]
    if not navigation:
        raise ValueError("trajectory contains no navigation steps")
    gate = StableArrivalGate(radius_m, hold_s, 0.02 * k, max_planar_speed_mps, max_abs_yaw_rate_radps)
    first_hit: dict[str, object] | None = None
    for step_index, row in enumerate(navigation, start=1):
        if step_index % k:
            continue
        values = [
            float(row[key])
            for key in ("time_s", "distance_to_goal_m", "actual_vx", "actual_vy", "actual_wz")
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trajectory contains a non-finite state")
        time_s, distance_m, vx, vy, wz = values
        if gate.update(distance_m, vx, vy, wz):
            first_hit = {
                "time_s": time_s,
                "phase": row["phase"],
                "mean_planar_speed_mps": sum(gate.speeds) / gate.window_steps,
                "abs_mean_yaw_rate_radps": abs(sum(gate.yaw_rates) / gate.window_steps),
            }
            break
    return {
        "observed_stable_arrival": first_hit is not None,
        "first_hit": first_hit,
        "recorded_end_time_s": float(navigation[-1]["time_s"]),
        "recorded_post_arrival_steps": sum(row["phase"] == "post_arrival" for row in navigation),
        "window_steps": gate.window_steps,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).parent / "pointgoal_protocols" / "week04_classical_v1.json")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=(2, 5, 10), required=True)
    parser.add_argument("--max-planar-speed-mps", type=float, required=True)
    parser.add_argument("--max-abs-yaw-rate-radps", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if args.max_planar_speed_mps <= 0 or args.max_abs_yaw_rate_radps <= 0:
        parser.error("speed thresholds must be positive")

    protocol_path = args.protocol.resolve()
    raw_dir = args.raw_dir.resolve()
    protocol = load_pointgoal_protocol(protocol_path)
    if args.smoke:
        mode_name = "smoke"
        specs = [protocol.episode_map[protocol.smoke_episode_ids["front"]]]
    else:
        mode_name = "full" if args.full else "quick"
        specs = list(select_episode_specs(protocol, mode_name))
    inputs = [
        (spec, raw_dir / f"{spec.episode_id}_rl_k{args.k}_seed{spec.simulation_reset_seed}_steps.csv")
        for spec in specs
    ]
    missing = [str(path) for _, path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen {mode_name} trajectories: {missing}")
    input_hashes = {spec.episode_id: _sha256(path) for spec, path in inputs}
    print(f"validated K={args.k}, episodes={len(inputs)}, protocol={_sha256(protocol_path)}")
    if args.validate_only:
        return 0

    episodes = []
    for spec, path in inputs:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        result = rescore_rows(
            rows,
            k=args.k,
            radius_m=spec.success_radius,
            hold_s=spec.success_hold_time,
            max_planar_speed_mps=args.max_planar_speed_mps,
            max_abs_yaw_rate_radps=args.max_abs_yaw_rate_radps,
        )
        episodes.append({"episode_id": spec.episode_id, **result})
    payload = {
        "kind": "recorded_trajectory_stable_arrival_diagnostic",
        "mode": mode_name,
        "k": args.k,
        "thresholds": {
            "max_planar_speed_mps": args.max_planar_speed_mps,
            "max_abs_yaw_rate_radps": args.max_abs_yaw_rate_radps,
        },
        "protocol_sha256": _sha256(protocol_path),
        "input_sha256": {episode["episode_id"]: input_hashes[episode["episode_id"]] for episode in episodes},
        "observed_stable_arrivals": sum(episode["observed_stable_arrival"] for episode in episodes),
        "episodes": episodes,
        "scope_note": "Only the recorded original navigation plus two-second post-arrival continuation is scored; non-hits are not full 20-second failures.",
    }
    output_dir = args.output_dir.resolve() / payload["mode"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"observed stable arrivals: {payload['observed_stable_arrivals']}/{len(episodes)}; {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
