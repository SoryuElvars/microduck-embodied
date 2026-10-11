#!/usr/bin/env python3
"""Diagnose the frozen locomotion bridge inside the PointGoal training env.

This bypasses the learned high-level actor and sends deterministic normalized
navigation actions through ``FrozenLocomotionAction``.  It therefore measures
the exact TorchScript + Warp path used while training the high-level policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_pointgoal_navigation_env_cfg import (
    make_microduck_pointgoal_navigation_env_cfg,
)


MAX_FORWARD_SPEED_MPS = 0.25
MAX_YAW_RATE_RADPS = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--command-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    cases: list[dict[str, Any]] = []
    for raw in payload["commands"]:
        case = dict(defaults)
        case.update(raw)
        cases.append(case)
    return cases


def _raw_action(vx: float, wz: float, num_envs: int, device: str) -> torch.Tensor:
    raw_vx = 2.0 * vx / MAX_FORWARD_SPEED_MPS - 1.0
    raw_wz = wz / MAX_YAW_RATE_RADPS
    return torch.tensor([raw_vx, raw_wz], device=device).repeat(num_envs, 1)


def _steps(seconds: float, dt: float) -> int:
    return max(0, int(math.ceil(seconds / dt - 1.0e-9)))


def _run_case(
    env: ManagerBasedRlEnv,
    case: dict[str, Any],
    *,
    num_envs: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    env.reset(seed=seed)
    action_term = env.action_manager.get_term("navigation")
    robot = env.scene["robot"]
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    failed = torch.zeros_like(active)
    dt = float(env.step_dt)

    phases = (
        (float(case.get("warmup_s", 0.0)), 0.0, 0.0, False),
        (
            float(case.get("lead_in_s", 0.0)),
            float(case.get("lead_in_vx", 0.0)),
            float(case.get("lead_in_wz", 0.0)),
            False,
        ),
        (
            float(case["duration_s"]),
            float(case["vx"]),
            float(case["wz"]),
            True,
        ),
    )

    samples: dict[str, list[float]] = {
        "command_vx": [],
        "command_wz": [],
        "actual_vx": [],
        "actual_vy": [],
        "actual_wz": [],
    }
    for seconds, vx, wz, record in phases:
        action = _raw_action(vx, wz, num_envs, device)
        for _ in range(_steps(seconds, dt)):
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            valid = active & ~done
            failed |= active & done
            if record and valid.any():
                command = action_term.applied_command[valid]
                linear = robot.data.root_link_lin_vel_b[valid]
                angular = robot.data.root_link_ang_vel_b[valid]
                samples["command_vx"].extend(command[:, 0].detach().cpu().tolist())
                samples["command_wz"].extend(command[:, 2].detach().cpu().tolist())
                samples["actual_vx"].extend(linear[:, 0].detach().cpu().tolist())
                samples["actual_vy"].extend(linear[:, 1].detach().cpu().tolist())
                samples["actual_wz"].extend(angular[:, 2].detach().cpu().tolist())
            active &= ~done

    def mean(name: str) -> float:
        return fmean(samples[name]) if samples[name] else float("nan")

    vx_errors = [
        actual - command
        for actual, command in zip(samples["actual_vx"], samples["command_vx"])
    ]
    wz_errors = [
        actual - command
        for actual, command in zip(samples["actual_wz"], samples["command_wz"])
    ]

    def rmse(values: list[float]) -> float:
        return math.sqrt(fmean(value * value for value in values)) if values else float("nan")

    return {
        "name": case["name"],
        "target": {"vx": float(case["vx"]), "wz": float(case["wz"])},
        "sample_count": len(samples["actual_vx"]),
        "failure_count": int(failed.sum().item()),
        "failure_rate": float(failed.float().mean().item()),
        "mean_command_vx_mps": mean("command_vx"),
        "mean_actual_vx_mps": mean("actual_vx"),
        "vx_rmse_mps": rmse(vx_errors),
        "mean_abs_lateral_speed_mps": (
            fmean(abs(value) for value in samples["actual_vy"])
            if samples["actual_vy"]
            else float("nan")
        ),
        "mean_command_wz_radps": mean("command_wz"),
        "mean_actual_wz_radps": mean("actual_wz"),
        "wz_rmse_radps": rmse(wz_errors),
    }


def main() -> None:
    args = _parse_args()
    policy = args.policy.expanduser().resolve()
    cases = _load_cases(args.command_set.expanduser().resolve())
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if not policy.is_file():
        raise FileNotFoundError(policy)

    result: dict[str, Any] = {
        "policy": str(policy),
        "command_set": str(args.command_set.expanduser().resolve()),
        "device": args.device,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "results_by_k": {},
    }
    for k in args.k:
        cfg = make_microduck_pointgoal_navigation_env_cfg(
            high_level_k=k,
            play=False,
            low_level_policy_path=policy,
        )
        cfg.scene.num_envs = args.num_envs
        cfg.episode_length_s = 60.0
        cfg.terminations.pop("goal_reached", None)
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            result["results_by_k"][str(k)] = [
                _run_case(
                    env,
                    case,
                    num_envs=args.num_envs,
                    device=args.device,
                    seed=args.seed + index,
                )
                for index, case in enumerate(cases)
            ]
        finally:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
