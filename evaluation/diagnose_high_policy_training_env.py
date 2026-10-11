#!/usr/bin/env python3
"""Evaluate one high-level checkpoint in its batched Warp training environment."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab_microduck.tasks.microduck_pointgoal_navigation_env_cfg import (
    make_microduck_pointgoal_navigation_env_cfg,
    make_microduck_pointgoal_navigation_rl_cfg,
)


TASK_IDS = {
    2: "Mjlab-PointGoal-Navigation-K2-MicroDuck",
    5: "Mjlab-PointGoal-Navigation-K5-MicroDuck",
    10: "Mjlab-PointGoal-Navigation-K10-MicroDuck",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--low-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=sorted(TASK_IDS), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    low_policy = args.low_policy.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not low_policy.is_file():
        raise FileNotFoundError(low_policy)

    env_cfg = make_microduck_pointgoal_navigation_env_cfg(
        high_level_k=args.k,
        play=False,
        low_level_policy_path=low_policy,
    )
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = make_microduck_pointgoal_navigation_rl_cfg(args.k)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_IDS[args.k]) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    try:
        obs_dict, _ = base_env.reset(seed=args.seed)
        obs = TensorDict(obs_dict, batch_size=[args.num_envs])
        pointgoal = base_env.command_manager.get_term("point_goal")
        initial_distance = torch.linalg.vector_norm(pointgoal.command, dim=1)
        last_distance = initial_distance.clone()
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        terminal_step = torch.full(
            (args.num_envs,), -1, dtype=torch.long, device=args.device
        )
        reasons = {
            name: torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            for name in ("goal_reached", "fell_over", "nan_state", "time_out")
        }
        action_vx: list[float] = []
        action_wz: list[float] = []
        max_steps = int(math.ceil(env_cfg.episode_length_s / base_env.step_dt)) + 2

        with torch.inference_mode():
            for step in range(1, max_steps + 1):
                distance_before = torch.linalg.vector_norm(pointgoal.command, dim=1)
                last_distance[active] = distance_before[active]
                actions = policy(obs)
                if active.any():
                    action_vx.extend(actions[active, 0].detach().cpu().tolist())
                    action_wz.extend(actions[active, 1].detach().cpu().tolist())
                obs, _, dones, _ = env.step(actions)
                just_done = active & dones.bool()
                if just_done.any():
                    terminal_step[just_done] = step
                    for name in reasons:
                        reasons[name][just_done] = base_env.termination_manager.get_term(name)[
                            just_done
                        ]
                    active &= ~just_done
                if not active.any():
                    break

        unresolved = int(active.sum().item())
        result = {
            "checkpoint": str(checkpoint),
            "low_policy": str(low_policy),
            "k": args.k,
            "step_dt_s": float(base_env.step_dt),
            "num_envs": args.num_envs,
            "seed": args.seed,
            "completed": args.num_envs - unresolved,
            "unresolved": unresolved,
            "successes": int(reasons["goal_reached"].sum().item()),
            "falls": int(reasons["fell_over"].sum().item()),
            "invalid_states": int(reasons["nan_state"].sum().item()),
            "timeouts": int(reasons["time_out"].sum().item()),
            "success_rate": float(reasons["goal_reached"].float().mean().item()),
            "mean_initial_distance_m": float(initial_distance.mean().item()),
            "mean_last_distance_m": float(last_distance.mean().item()),
            "mean_distance_reduction_m": float((initial_distance - last_distance).mean().item()),
            "mean_terminal_time_s": float(
                terminal_step[terminal_step >= 0].float().mean().item()
                * base_env.step_dt
            ),
            "mean_raw_action_vx": fmean(action_vx),
            "mean_raw_action_wz": fmean(action_wz),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
