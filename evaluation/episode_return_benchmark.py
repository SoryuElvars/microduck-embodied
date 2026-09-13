#!/usr/bin/env python3
"""Record exact Episode Return with MicroDuck's official reward manager.

Run this script from the official ``microduck_rl`` uv environment.  Unlike the
ONNX deployment rehearsal, this evaluator loads the PyTorch checkpoint into the
official mjlab task, so the total reward and every weighted reward term are
available at each environment step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from evaluation.locomotion_benchmark import CommandCase, load_command_set
except ModuleNotFoundError:  # Direct execution from the evaluation directory.
    from locomotion_benchmark import CommandCase, load_command_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"
DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"
DEFAULT_COMMAND_SET = PROJECT_ROOT / "evaluation" / "command_sets" / "checkpoint_response_5.json"
DEFAULT_CHECKPOINT = (
    DEFAULT_MICRODUCK_RL_ROOT
    / "logs"
    / "rsl_rl"
    / "velocity"
    / "2026-09-05_22-08-33_baseline-flat-resume-5000"
    / "model_5999.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "week02"
    / "01_baseline_5999"
    / "summaries"
    / "episode_return_8x5_model_5999.json"
)

NOMINAL_DISABLED_EVENTS = (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_com",
    "randomize_head_com",
    "randomize_mass_inertia",
    "randomize_joint_friction",
    "randomize_joint_damping",
    "randomize_armature",
    "randomize_base_orientation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot describe an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_episodes(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate zero episodes")
    term_names = list(episodes[0]["reward_terms"].keys())
    return {
        "episode_count": len(episodes),
        "terminated_count": sum(bool(item["terminated"]) for item in episodes),
        "time_out_count": sum(bool(item["time_out"]) for item in episodes),
        "episode_return": describe([float(item["episode_return"]) for item in episodes]),
        "mean_reward_rate": describe(
            [float(item["mean_reward_rate"]) for item in episodes]
        ),
        "episode_duration_s": describe(
            [float(item["episode_duration_s"]) for item in episodes]
        ),
        "reward_terms": {
            name: {
                "return": describe(
                    [float(item["reward_terms"][name]["return"]) for item in episodes]
                ),
                "mean_rate": describe(
                    [
                        float(item["reward_terms"][name]["mean_rate"])
                        for item in episodes
                    ]
                ),
            }
            for name in term_names
        },
    }


def build_document(
    *,
    task: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    microduck_rl_commit: str,
    command_set: Path,
    base_seed: int,
    duration_s: float,
    step_dt_s: float,
    common_step_counter: int,
    reward_scale_by_dt: bool,
    reward_weights: dict[str, float],
    disabled_events: Sequence[str],
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    commands: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        commands.setdefault(str(episode["command_name"]), []).append(episode)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "official_mjlab_fixed_command_episode_return",
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "microduck_rl_commit": microduck_rl_commit,
        "command_set": str(command_set),
        "protocol": {
            "profile": "nominal",
            "base_seed": base_seed,
            "paired_environment_slots": len(commands[next(iter(commands))]),
            "command_count": len(commands),
            "episode_count": len(episodes),
            "episode_duration_s": duration_s,
            "step_dt_s": step_dt_s,
            "warmup_s": 0.0,
            "observation_corruption": False,
            "fixed_zero_head_and_body_commands": True,
            "disabled_events": list(disabled_events),
            "note": (
                "The same environment slots and reset seed are reused for every command, "
                "so reset-level randomness is paired across commands."
            ),
        },
        "reward_contract": {
            "common_step_counter": common_step_counter,
            "scale_by_dt": reward_scale_by_dt,
            "episode_return_definition": "sum of reward_buf over environment steps",
            "mean_reward_rate_definition": "episode_return / elapsed episode seconds",
            "weights": reward_weights,
        },
        "aggregate": aggregate_episodes(episodes),
        "commands": [
            {
                "name": name,
                "command": command_episodes[0]["command"],
                "aggregate": aggregate_episodes(command_episodes),
                "episodes": command_episodes,
            }
            for name, command_episodes in commands.items()
        ],
    }


def _configure_nominal_env(env_cfg: Any, *, num_envs: int, seed: int, duration_s: float) -> list[str]:
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.episode_length_s = duration_s
    env_cfg.auto_reset = False

    for group in env_cfg.observations.values():
        group.enable_corruption = False

    disabled = []
    for name in NOMINAL_DISABLED_EVENTS:
        if name in env_cfg.events:
            env_cfg.events.pop(name)
            disabled.append(name)
    # These curricula only widen the two reset-time CoM randomizers removed
    # above. Leaving them active would make reset look up missing event terms.
    env_cfg.curriculum.pop("com_range", None)
    env_cfg.curriculum.pop("head_com_range", None)

    twist_cfg = env_cfg.commands["twist"]
    twist_cfg.resampling_time_range = (1.0e6, 1.0e6)
    twist_cfg.rel_standing_envs = 0.0
    twist_cfg.rel_heading_envs = 0.0
    twist_cfg.rel_world_envs = 0.0
    twist_cfg.rel_forward_envs = 0.0
    twist_cfg.rel_turn_in_place_envs = 0.0
    twist_cfg.init_velocity_prob = 0.0
    for name in ("head_pose", "body_pose"):
        env_cfg.commands[name].resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands[name].zero_command_prob = 1.0
    return disabled


def _set_fixed_commands(env: Any, command: CommandCase, active: Any | None = None) -> None:
    import torch

    twist = env.command_manager.get_term("twist")
    values = torch.tensor(
        [command.vx, command.vy, command.wz],
        dtype=twist.vel_command_b.dtype,
        device=env.device,
    )
    twist.vel_command_b[:] = values
    twist.vel_command_w[:] = values
    for flag_name in (
        "is_heading_env",
        "is_standing_env",
        "is_world_env",
        "is_forward_env",
    ):
        getattr(twist, flag_name).zero_()
    if active is not None:
        inactive = ~active
        twist.vel_command_b[inactive] = 0.0
        twist.vel_command_w[inactive] = 0.0
    env.command_manager.get_term("head_pose")._command.zero_()
    env.command_manager.get_term("body_pose")._command.zero_()


def _run_command(
    *,
    wrapped_env: Any,
    policy: Any,
    command: CommandCase,
    reset_seed: int,
) -> list[dict[str, Any]]:
    import torch

    env = wrapped_env.unwrapped
    env.reset(seed=reset_seed)
    _set_fixed_commands(env, command)
    obs = wrapped_env.get_observations()

    num_envs = env.num_envs
    term_names = list(env.reward_manager.active_terms)
    episode_return = torch.zeros(num_envs, device=env.device)
    term_return = torch.zeros((num_envs, len(term_names)), device=env.device)
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    timed_out = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    active = torch.ones(num_envs, dtype=torch.bool, device=env.device)

    reward_manager = env.reward_manager
    scale = env.step_dt if reward_manager._scale_by_dt else 1.0
    max_steps = env.max_episode_length + 1

    for _ in range(max_steps):
        _set_fixed_commands(env, command, active)
        with torch.inference_mode():
            actions = policy(obs)
        actions = actions.clone()
        actions[~active] = 0.0
        obs, reward, dones, _extras = wrapped_env.step(actions)

        step_terms = reward_manager._step_reward * scale
        if not torch.allclose(
            step_terms.sum(dim=1), reward, atol=1.0e-5, rtol=1.0e-5
        ):
            raise RuntimeError("reward term sum does not match reward_buf")

        episode_return[active] += reward[active]
        term_return[active] += step_terms[active]
        episode_steps[active] += 1

        done_now = active & dones.bool()
        terminated[done_now] = env.reset_terminated[done_now]
        timed_out[done_now] = env.reset_time_outs[done_now]
        active &= ~done_now
        if not active.any():
            break

        # auto_reset is disabled so completed slots must be reset before the
        # remaining slots can take another step. Their later rewards are ignored.
        reset_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            env.reset(env_ids=reset_ids)
            _set_fixed_commands(env, command, active)
            obs = wrapped_env.get_observations()
    else:
        raise RuntimeError(f"command {command.name} exceeded max episode steps")

    returns = episode_return.detach().cpu().tolist()
    term_returns = term_return.detach().cpu().tolist()
    steps = episode_steps.detach().cpu().tolist()
    terminated_list = terminated.detach().cpu().tolist()
    time_out_list = timed_out.detach().cpu().tolist()
    records = []
    for pair_id in range(num_envs):
        elapsed = steps[pair_id] * env.step_dt
        records.append(
            {
                "command_name": command.name,
                "command": {"vx": command.vx, "vy": command.vy, "wz": command.wz},
                "pair_id": pair_id,
                "reset_seed": reset_seed,
                "steps": steps[pair_id],
                "episode_duration_s": elapsed,
                "terminated": bool(terminated_list[pair_id]),
                "time_out": bool(time_out_list[pair_id]),
                "episode_return": returns[pair_id],
                "mean_reward_per_step": returns[pair_id] / steps[pair_id],
                "mean_reward_rate": returns[pair_id] / elapsed,
                "reward_terms": {
                    name: {
                        "return": term_returns[pair_id][term_index],
                        "mean_rate": term_returns[pair_id][term_index] / elapsed,
                    }
                    for term_index, name in enumerate(term_names)
                },
            }
        )
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    import mjlab.tasks  # noqa: F401 -- populate the task registry
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    checkpoint = args.checkpoint.expanduser().resolve()
    command_set = args.command_set.expanduser().resolve()
    microduck_rl_root = args.microduck_rl_root.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not command_set.is_file():
        raise FileNotFoundError(f"command set not found: {command_set}")

    commands = load_command_set(command_set)
    if args.max_commands is not None:
        commands = commands[: args.max_commands]
    if not commands:
        raise ValueError("command set contains no selected commands")
    episodes_per_command = args.episodes_per_command or commands[0].episodes
    duration_s = args.duration_s or commands[0].duration_s
    if episodes_per_command <= 0 or duration_s <= 0:
        raise ValueError("episodes-per-command and duration-s must be positive")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    disabled_events = _configure_nominal_env(
        env_cfg,
        num_envs=episodes_per_command,
        seed=args.seed,
        duration_s=duration_s,
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        # Loading restores the checkpoint's common_step_counter. Reset once so
        # the curriculum manager applies the corresponding final reward weights.
        env.reset(seed=args.seed)
        reward_weights = {
            name: float(env.reward_manager.get_term_cfg(name).weight)
            for name in env.reward_manager.active_terms
        }
        common_step_counter = int(env.common_step_counter)

        episodes: list[dict[str, Any]] = []
        for index, command in enumerate(commands, start=1):
            print(
                f"[{index}/{len(commands)}] {command.name}: "
                f"({command.vx:+.2f}, {command.vy:+.2f}, {command.wz:+.2f})"
            )
            episodes.extend(
                _run_command(
                    wrapped_env=wrapped,
                    policy=policy,
                    command=command,
                    reset_seed=args.seed,
                )
            )

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=microduck_rl_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        document = build_document(
            task=args.task,
            checkpoint=checkpoint.relative_to(microduck_rl_root),
            checkpoint_sha256=_sha256(checkpoint),
            microduck_rl_commit=commit,
            command_set=command_set.relative_to(PROJECT_ROOT),
            base_seed=args.seed,
            duration_s=duration_s,
            step_dt_s=float(env.step_dt),
            common_step_counter=common_step_counter,
            reward_scale_by_dt=bool(env.reward_manager._scale_by_dt),
            reward_weights=reward_weights,
            disabled_events=disabled_events,
            episodes=episodes,
        )
    finally:
        wrapped.close()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary: {output}")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--command-set", type=Path, default=DEFAULT_COMMAND_SET)
    parser.add_argument("--microduck-rl-root", type=Path, default=DEFAULT_MICRODUCK_RL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-command", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--max-commands", type=int)
    parser.add_argument("--device")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    document = run(args)
    print(
        "episode return mean: "
        f"{document['aggregate']['episode_return']['mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
