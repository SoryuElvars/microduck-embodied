#!/usr/bin/env python3
"""Analyze bilateral action symmetry in the eight-command benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_MICRODUCK_RL_ROOT = Path.home() / "projects" / "microduck_rl"

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
JOINT_SHORT_NAMES = (
    "L hip yaw",
    "L hip roll",
    "L hip pitch",
    "L knee",
    "L ankle",
    "neck pitch",
    "head pitch",
    "head yaw",
    "head roll",
    "R hip yaw",
    "R hip roll",
    "R hip pitch",
    "R knee",
    "R ankle",
)

# Kept identical to microduck_rl/src/mjlab_microduck/tasks/symmetry.py.
JOINT_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
JOINT_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
              -1.0, -1.0, -1.0, -1.0, -1.0)
OBS_PERM = (
    (0, 1, 2)
    + (3, 4, 5)
    + tuple(6 + joint for joint in JOINT_PERM)
    + tuple(20 + joint for joint in JOINT_PERM)
    + tuple(34 + joint for joint in JOINT_PERM)
    + (48, 49, 50)
    + (51, 52, 53, 54)
    + (55, 56, 57, 58, 59, 60)
)
OBS_SIGN = (
    (-1.0, 1.0, -1.0)
    + (1.0, -1.0, 1.0)
    + JOINT_SIGN
    + JOINT_SIGN
    + JOINT_SIGN
    + (1.0, -1.0, -1.0)
    + (1.0, 1.0, -1.0, -1.0)
    + (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
)

MIRROR_PAIRS = (
    (
        "yaw_positive_vs_negative",
        "yaw_pos_050_official_reset_20",
        "yaw_neg_050_official_reset_20",
    ),
    (
        "lateral_positive_vs_negative",
        "lateral_pos_020_official_reset_20",
        "lateral_neg_020_official_reset_20",
    ),
)


def mirror_vector(values: Sequence[float], permutation: Sequence[int], signs: Sequence[float]):
    """Mirror one vector without requiring NumPy (also used by unit tests)."""

    return [float(values[source]) * float(sign) for source, sign in zip(permutation, signs)]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_episode_arrays(path: Path, np) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        phase_index = header.index("phase")
        step_index = header.index("step")
        observation_indices = [header.index(f"obs_{index:02d}") for index in range(61)]
        action_indices = [header.index(f"action_{index:02d}") for index in range(14)]
        steps = []
        observations = []
        actions = []
        for row in reader:
            if row[phase_index] != "test":
                continue
            steps.append(int(row[step_index]))
            observations.append([float(row[index]) for index in observation_indices])
            actions.append([float(row[index]) for index in action_indices])
    if not steps:
        raise ValueError(f"{path} contains no phase=test rows")
    return {
        "steps": np.asarray(steps, dtype=np.int32),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }


def describe_joint_error(difference, reference, np) -> dict[str, Any]:
    rmse = np.sqrt(np.mean(np.square(difference), axis=0))
    mae = np.mean(np.abs(difference), axis=0)
    reference_rms = np.sqrt(np.mean(np.square(reference), axis=0))
    relative_rmse = rmse / np.maximum(reference_rms, 1.0e-8)
    return {
        "sample_count": int(difference.shape[0]),
        "overall_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "per_joint": [
            {
                "index": index,
                "joint": JOINT_NAMES[index],
                "rmse": float(rmse[index]),
                "mae": float(mae[index]),
                "reference_rms": float(reference_rms[index]),
                "relative_rmse": float(relative_rmse[index]),
            }
            for index in range(len(JOINT_NAMES))
        ],
    }


def action_rms(actions, np) -> list[float]:
    return [float(value) for value in np.sqrt(np.mean(np.square(actions), axis=0))]


def load_benchmark_data(suite_path: Path, np):
    suite = load_json(suite_path)
    command_summaries: dict[str, dict[str, Any]] = {}
    episode_data: dict[str, dict[int, dict[str, Any]]] = {}
    for command in suite["commands"]:
        summary_path = PROJECT_ROOT / command["summary_json"]
        summary = load_json(summary_path)
        name = summary["command"]["name"]
        command_summaries[name] = summary
        episode_data[name] = {}
        for episode in summary["episodes"]:
            raw_path = PROJECT_ROOT / episode["raw_steps_csv"]
            episode_data[name][int(episode["seed"])] = load_episode_arrays(raw_path, np)
    return suite, command_summaries, episode_data


def policy_equivariance(policy_path: Path, episode_data, sample_stride: int, np):
    import onnxruntime as ort

    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    permutation = np.asarray(OBS_PERM, dtype=np.int64)
    signs = np.asarray(OBS_SIGN, dtype=np.float32)
    action_permutation = np.asarray(JOINT_PERM, dtype=np.int64)
    action_signs = np.asarray(JOINT_SIGN, dtype=np.float32)

    by_command: dict[str, Any] = {}
    all_differences = []
    all_references = []
    all_replay_differences = []
    all_replay_references = []
    for command_name, episodes in episode_data.items():
        command_differences = []
        command_references = []
        command_replay_differences = []
        command_replay_references = []
        for episode in episodes.values():
            observations = episode["observations"][::sample_stride]
            actions = episode["actions"][::sample_stride]
            mirrored_observations = observations[:, permutation] * signs
            expected_actions = actions[:, action_permutation] * action_signs
            replayed_actions = np.stack(
                [
                    session.run(
                        [output_name],
                        {input_meta.name: observation.reshape(1, 61)},
                    )[0][0]
                    for observation in observations
                ]
            )
            actual_actions = np.stack(
                [
                    session.run(
                        [output_name],
                        {input_meta.name: observation.reshape(1, 61)},
                    )[0][0]
                    for observation in mirrored_observations
                ]
            )
            command_differences.append(actual_actions - expected_actions)
            command_references.append(expected_actions)
            command_replay_differences.append(replayed_actions - actions)
            command_replay_references.append(actions)
        differences = np.concatenate(command_differences)
        references = np.concatenate(command_references)
        replay_differences = np.concatenate(command_replay_differences)
        replay_references = np.concatenate(command_replay_references)
        by_command[command_name] = {
            "mirror_error": describe_joint_error(differences, references, np),
            "replay_error": describe_joint_error(
                replay_differences, replay_references, np
            ),
        }
        all_differences.append(differences)
        all_references.append(references)
        all_replay_differences.append(replay_differences)
        all_replay_references.append(replay_references)

    result = {
        "definition": "pi(mirror(observation)) - mirror(pi(observation))",
        "sample_stride_control_steps": sample_stride,
        "replay_validation": describe_joint_error(
            np.concatenate(all_replay_differences),
            np.concatenate(all_replay_references),
            np,
        ),
        "overall": describe_joint_error(
            np.concatenate(all_differences),
            np.concatenate(all_references),
            np,
        ),
        "by_source_command": by_command,
    }
    if result["replay_validation"]["overall_rmse"] > 1.0e-6:
        raise ValueError(
            "ONNX replay did not reproduce recorded actions: "
            f"RMSE={result['replay_validation']['overall_rmse']:.3e}"
        )
    return result


def closed_loop_pair_analysis(episode_data, np):
    action_permutation = np.asarray(JOINT_PERM, dtype=np.int64)
    action_signs = np.asarray(JOINT_SIGN, dtype=np.float32)
    results = []
    for pair_name, positive_name, negative_name in MIRROR_PAIRS:
        positive_episodes = episode_data[positive_name]
        negative_episodes = episode_data[negative_name]
        seeds = sorted(set(positive_episodes) & set(negative_episodes))
        if set(positive_episodes) != set(negative_episodes):
            raise ValueError(f"paired seed mismatch for {pair_name}")

        differences = []
        expected_negative_actions = []
        positive_actions = []
        negative_actions_in_positive_frame = []
        for seed in seeds:
            positive = positive_episodes[seed]
            negative = negative_episodes[seed]
            if not np.array_equal(positive["steps"], negative["steps"]):
                raise ValueError(f"step mismatch for {pair_name}, seed {seed}")
            expected_negative = positive["actions"][:, action_permutation] * action_signs
            differences.append(negative["actions"] - expected_negative)
            expected_negative_actions.append(expected_negative)
            positive_actions.append(positive["actions"])
            negative_actions_in_positive_frame.append(
                negative["actions"][:, action_permutation] * action_signs
            )

        difference = np.concatenate(differences)
        reference = np.concatenate(expected_negative_actions)
        positive = np.concatenate(positive_actions)
        negative_mirrored = np.concatenate(negative_actions_in_positive_frame)
        positive_rms = np.asarray(action_rms(positive, np))
        negative_rms = np.asarray(action_rms(negative_mirrored, np))
        results.append(
            {
                "name": pair_name,
                "positive_command": positive_name,
                "negative_command": negative_name,
                "paired_seeds": seeds,
                "mirror_error": describe_joint_error(difference, reference, np),
                "action_rms_positive_frame": {
                    "positive_command": [float(value) for value in positive_rms],
                    "mirrored_negative_command": [float(value) for value in negative_rms],
                    "negative_to_positive_ratio": [
                        float(value)
                        for value in negative_rms / np.maximum(positive_rms, 1.0e-8)
                    ],
                },
            }
        )
    return results


def bilateral_leg_amplitude(episode_data, np):
    results = {}
    for command_name, episodes in episode_data.items():
        actions = np.concatenate([episode["actions"] for episode in episodes.values()])
        rms = np.asarray(action_rms(actions, np))
        leg_pairs = []
        for offset, joint_type in enumerate(
            ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
        ):
            left = float(rms[offset])
            right = float(rms[9 + offset])
            denominator = max(0.5 * (left + right), 1.0e-8)
            leg_pairs.append(
                {
                    "joint_type": joint_type,
                    "left_rms": left,
                    "right_rms": right,
                    "signed_relative_difference": (left - right) / denominator,
                    "absolute_relative_difference": abs(left - right) / denominator,
                }
            )
        results[command_name] = leg_pairs
    return results


def plot_results(document: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(JOINT_NAMES))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    ax = axes[0, 0]
    overall = document["policy_equivariance"]["overall"]["per_joint"]
    values = [joint["rmse"] for joint in overall]
    ax.bar(x, values, color="#5E35B1", alpha=0.82)
    ax.set_title("Intrinsic ONNX mirror-equivariance error")
    ax.set_ylabel("Action RMSE")
    ax.set_xticks(x, JOINT_SHORT_NAMES, rotation=48, ha="right")

    ax = axes[0, 1]
    colors = ("#C62828", "#1565C0")
    for pair, color in zip(document["closed_loop_pairs"], colors, strict=True):
        values = [joint["rmse"] for joint in pair["mirror_error"]["per_joint"]]
        ax.plot(x, values, marker="o", linewidth=2.0, color=color, label=pair["name"])
    ax.set_title("Closed-loop paired mirror error")
    ax.set_ylabel("Action RMSE")
    ax.set_xticks(x, JOINT_SHORT_NAMES, rotation=48, ha="right")
    ax.legend(loc="best")

    for axis, pair, title in zip(
        axes[1],
        document["closed_loop_pairs"],
        ("Yaw commands: action activation", "Lateral commands: action activation"),
        strict=True,
    ):
        rms = pair["action_rms_positive_frame"]
        width = 0.38
        axis.bar(
            x - width / 2,
            rms["positive_command"],
            width=width,
            color="#00897B",
            alpha=0.82,
            label="positive command",
        )
        axis.bar(
            x + width / 2,
            rms["mirrored_negative_command"],
            width=width,
            color="#F9A825",
            alpha=0.82,
            label="mirrored negative command",
        )
        axis.set_title(title)
        axis.set_ylabel("Action RMS")
        axis.set_xticks(x, JOINT_SHORT_NAMES, rotation=48, ha="right")
        axis.legend(loc="best")

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.24)
    fig.suptitle("MicroDuck model_5999 — bilateral action-symmetry diagnosis")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(
    suite_path: Path,
    microduck_rl_root: Path,
    output_json: Path,
    output_figure: Path,
    sample_stride: int,
) -> None:
    import numpy as np

    suite, _, episode_data = load_benchmark_data(suite_path, np)
    policy_path = microduck_rl_root / suite["policy_path"]
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy not found: {policy_path}")
    symmetry_path = (
        microduck_rl_root
        / "src"
        / "mjlab_microduck"
        / "tasks"
        / "symmetry.py"
    )
    if not symmetry_path.is_file():
        raise FileNotFoundError(f"official symmetry definition not found: {symmetry_path}")
    official_commit = subprocess.run(
        ["git", "-C", str(microduck_rl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "bilateral_action_symmetry",
        "source_suite": str(suite_path.relative_to(PROJECT_ROOT)),
        "policy_path": str(policy_path.relative_to(microduck_rl_root)),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "microduck_rl_commit": official_commit,
        "joint_order": list(JOINT_NAMES),
        "official_mirror_contract": {
            "source": "microduck_rl/src/mjlab_microduck/tasks/symmetry.py",
            "source_sha256": hashlib.sha256(symmetry_path.read_bytes()).hexdigest(),
            "joint_permutation": list(JOINT_PERM),
            "joint_sign": list(JOINT_SIGN),
            "observation_permutation": list(OBS_PERM),
            "observation_sign": list(OBS_SIGN),
        },
        "policy_equivariance": policy_equivariance(
            policy_path, episode_data, sample_stride, np
        ),
        "closed_loop_pairs": closed_loop_pair_analysis(episode_data, np),
        "bilateral_leg_action_rms": bilateral_leg_amplitude(episode_data, np),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_results(document, output_figure)
    print(f"Loaded {sum(len(episodes) for episodes in episode_data.values())} episodes")
    print(
        "ONNX mirror-equivariance overall RMSE: "
        f"{document['policy_equivariance']['overall']['overall_rmse']:.6f}"
    )
    print(
        "ONNX recorded-action replay RMSE: "
        f"{document['policy_equivariance']['replay_validation']['overall_rmse']:.3e}"
    )
    for pair in document["closed_loop_pairs"]:
        print(
            f"{pair['name']} closed-loop mirror RMSE: "
            f"{pair['mirror_error']['overall_rmse']:.6f}"
        )
    print(f"Saved summary: {output_json}")
    print(f"Saved figure: {output_figure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze MicroDuck action symmetry")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--microduck-rl-root",
        type=Path,
        default=DEFAULT_MICRODUCK_RL_ROOT,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Control-step stride for ONNX mirror queries (default: 5 = 10 Hz)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")
    analyze(
        args.suite.expanduser().resolve(),
        args.microduck_rl_root.expanduser().resolve(),
        args.output_json.expanduser().resolve(),
        args.output_figure.expanduser().resolve(),
        args.sample_stride,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
