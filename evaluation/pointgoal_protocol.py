"""Versioned, frozen EpisodeSpec manifests for the Week 4 PointGoal benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.locomotion_benchmark import sample_initial_state
from navigation.classical_navigator import (
    ConstrainedGoToGoalConfig,
    NaivePConfig,
    wrap_angle,
)


PROTOCOL_STATUSES = {"draft", "frozen"}
MANIFEST_STATUSES = {"frozen"}
CONTROLLER_IDS = ("naive_p", "constrained")
SMOKE_DIRECTIONS_RAD = {
    "front": 0.0,
    "left": math.pi / 2,
    "right": -math.pi / 2,
    "behind": math.pi,
}


@dataclass(frozen=True)
class EpisodeSpec:
    """One immutable start/goal scenario; all distances use metres and seconds."""

    episode_id: str
    simulation_reset_seed: int
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    timeout: float
    success_radius: float
    success_hold_time: float
    post_arrival_observation_time: float

    def validate(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if self.simulation_reset_seed < 0:
            raise ValueError("simulation_reset_seed must be >= 0")
        values = (
            self.start_x,
            self.start_y,
            self.start_yaw,
            self.goal_x,
            self.goal_y,
            self.timeout,
            self.success_radius,
            self.success_hold_time,
            self.post_arrival_observation_time,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"episode {self.episode_id!r} contains a non-finite value")
        if self.timeout <= 0:
            raise ValueError(f"episode {self.episode_id!r} timeout must be > 0")
        if self.success_radius <= 0:
            raise ValueError(f"episode {self.episode_id!r} success_radius must be > 0")
        if self.success_hold_time <= 0:
            raise ValueError(
                f"episode {self.episode_id!r} success_hold_time must be > 0"
            )
        if self.post_arrival_observation_time <= 0:
            raise ValueError(
                f"episode {self.episode_id!r} post_arrival_observation_time must be > 0"
            )
        if math.hypot(self.goal_x - self.start_x, self.goal_y - self.start_y) <= self.success_radius:
            raise ValueError(f"episode {self.episode_id!r} goal begins inside success radius")


@dataclass(frozen=True)
class ClassicalPointGoalProtocol:
    source_path: Path
    protocol_id: str
    protocol_status: str
    episode_manifest_status: str
    description: str
    initial_state_mode: str
    warmup_s: float
    smoke_timeout_s: float
    navigator_frequency_hz: float
    controller_order: tuple[str, ...]
    naive_p: NaivePConfig
    constrained: ConstrainedGoToGoalConfig
    episodes: tuple[EpisodeSpec, ...]
    smoke_episode_ids: Mapping[str, str]
    quick_episode_ids: tuple[str, ...]
    episode_manifest_sha256: str
    generation: Mapping[str, Any]

    @property
    def episode_map(self) -> dict[str, EpisodeSpec]:
        return {episode.episode_id: episode for episode in self.episodes}


def _manifest_sha256(episodes: list[dict[str, Any]] | tuple[EpisodeSpec, ...]) -> str:
    records = [asdict(item) if isinstance(item, EpisodeSpec) else item for item in episodes]
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def load_pointgoal_protocol(path: Path) -> ClassicalPointGoalProtocol:
    """Load and cross-check a Week 4 Classical PointGoal protocol."""

    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError("PointGoal protocol schema_version must be 1")
    if document.get("task_definition", {}).get("goal_type") != "position_pointgoal":
        raise ValueError("task_definition.goal_type must be 'position_pointgoal'")
    if document.get("task_definition", {}).get("requires_final_yaw") is not False:
        raise ValueError("PointGoal v1 must not require a final yaw")

    protocol_status = str(document.get("protocol_status", ""))
    if protocol_status not in PROTOCOL_STATUSES:
        raise ValueError(f"protocol_status must be one of: {sorted(PROTOCOL_STATUSES)}")
    manifest_status = str(document.get("episode_manifest_status", ""))
    if manifest_status not in MANIFEST_STATUSES:
        raise ValueError("episode_manifest_status must be 'frozen'")

    runtime = _require_mapping(document.get("runtime"), "runtime")
    controllers = _require_mapping(document.get("controllers"), "controllers")
    shared = _require_mapping(controllers.get("shared_limits"), "controllers.shared_limits")
    naive_raw = _require_mapping(controllers.get("naive_p"), "controllers.naive_p")
    constrained_raw = _require_mapping(
        controllers.get("constrained"), "controllers.constrained"
    )
    goal_tolerance = float(shared.get("goal_tolerance_m"))
    naive = NaivePConfig(
        **naive_raw,
        max_forward_speed_mps=float(shared.get("max_forward_speed_mps")),
        max_yaw_rate_radps=float(shared.get("max_yaw_rate_radps")),
        goal_tolerance_m=goal_tolerance,
    )
    constrained = ConstrainedGoToGoalConfig(
        **constrained_raw,
        max_forward_speed_mps=float(shared.get("max_forward_speed_mps")),
        max_yaw_rate_radps=float(shared.get("max_yaw_rate_radps")),
        goal_tolerance_m=goal_tolerance,
    )
    naive.validate()
    constrained.validate()

    raw_episodes = document.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("episodes must be a non-empty list")
    episodes = tuple(EpisodeSpec(**raw) for raw in raw_episodes)
    for episode in episodes:
        episode.validate()
        sampled = sample_initial_state("official_reset", episode.simulation_reset_seed)
        expected = (sampled["x"], sampled["y"], sampled["yaw"])
        actual = (episode.start_x, episode.start_y, episode.start_yaw)
        if any(abs(left - right) > 1e-8 for left, right in zip(expected, actual)):
            raise ValueError(
                f"episode {episode.episode_id!r} start pose does not match its reset seed"
            )
        if not math.isclose(
            episode.success_hold_time,
            constrained.arrival_hold_time_s,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"episode {episode.episode_id!r} success_hold_time differs from "
                "the Constrained stop-hold time"
            )
    success_radii = {episode.success_radius for episode in episodes}
    if len(success_radii) != 1:
        raise ValueError("all EpisodeSpecs must share one success_radius")
    success_radius = next(iter(success_radii))
    if goal_tolerance >= success_radius:
        raise ValueError(
            "controller goal_tolerance_m must be smaller than the task success_radius"
        )
    episode_ids = [episode.episode_id for episode in episodes]
    reset_seeds = [episode.simulation_reset_seed for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode_id values must be unique")
    if len(reset_seeds) != len(set(reset_seeds)):
        raise ValueError("simulation_reset_seed values must be unique")

    expected_manifest_sha = _manifest_sha256(list(raw_episodes))
    recorded_manifest_sha = str(document.get("episode_manifest_sha256", ""))
    if recorded_manifest_sha != expected_manifest_sha:
        raise ValueError("episode manifest SHA-256 does not match the frozen episodes")

    selections = _require_mapping(document.get("selections"), "selections")
    smoke_raw = _require_mapping(selections.get("smoke"), "selections.smoke")
    if set(smoke_raw) != set(SMOKE_DIRECTIONS_RAD):
        raise ValueError("smoke selection must contain front/left/right/behind")
    smoke_ids = {name: str(value) for name, value in smoke_raw.items()}
    quick_raw = selections.get("quick")
    if not isinstance(quick_raw, list) or len(quick_raw) != 10:
        raise ValueError("quick selection must contain exactly 10 episode ids")
    quick_ids = tuple(str(value) for value in quick_raw)
    selected_ids = list(smoke_ids.values()) + list(quick_ids)
    unknown = set(selected_ids) - set(episode_ids)
    if unknown:
        raise ValueError(f"selections reference unknown episodes: {sorted(unknown)}")
    if len(set(smoke_ids.values())) != len(smoke_ids):
        raise ValueError("smoke direction episode ids must be unique")
    if len(set(quick_ids)) != len(quick_ids):
        raise ValueError("quick episode ids must be unique")

    controller_order = tuple(str(value) for value in controllers.get("order", ()))
    if controller_order != CONTROLLER_IDS:
        raise ValueError(f"controllers.order must be {list(CONTROLLER_IDS)}")
    if str(runtime.get("initial_state_mode")) != "official_reset":
        raise ValueError("runtime.initial_state_mode must be 'official_reset'")
    warmup_s = float(runtime.get("warmup_s"))
    smoke_timeout_s = float(runtime.get("smoke_timeout_s"))
    navigator_frequency_hz = float(runtime.get("navigator_frequency_hz"))
    if not math.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("runtime.warmup_s must be finite and >= 0")
    if not math.isfinite(navigator_frequency_hz) or navigator_frequency_hz <= 0:
        raise ValueError("runtime.navigator_frequency_hz must be finite and > 0")
    if not math.isfinite(smoke_timeout_s) or smoke_timeout_s <= 0:
        raise ValueError("runtime.smoke_timeout_s must be finite and > 0")

    return ClassicalPointGoalProtocol(
        source_path=path.resolve(),
        protocol_id=str(document.get("protocol_id", "")),
        protocol_status=protocol_status,
        episode_manifest_status=manifest_status,
        description=str(document.get("description", "")),
        initial_state_mode="official_reset",
        warmup_s=warmup_s,
        smoke_timeout_s=smoke_timeout_s,
        navigator_frequency_hz=navigator_frequency_hz,
        controller_order=controller_order,
        naive_p=naive,
        constrained=constrained,
        episodes=episodes,
        smoke_episode_ids=smoke_ids,
        quick_episode_ids=quick_ids,
        episode_manifest_sha256=recorded_manifest_sha,
        generation=_require_mapping(document.get("generation"), "generation"),
    )


def generate_week04_classical_v1(
    *,
    manifest_seed: int = 20260919,
    episode_count: int = 200,
    reset_seed_start: int = 40000,
) -> dict[str, Any]:
    """Deterministically generate the frozen random Start/Goal manifest."""

    if episode_count < 10:
        raise ValueError("episode_count must be >= 10")
    rng = random.Random(manifest_seed)
    episodes: list[dict[str, Any]] = []
    relative_headings: dict[str, float] = {}
    for index in range(episode_count):
        episode_id = f"w04e{index:03d}"
        reset_seed = reset_seed_start + index
        initial = sample_initial_state("official_reset", reset_seed)
        distance = rng.uniform(1.0, 1.8)
        relative_heading = rng.uniform(-math.pi, math.pi)
        world_heading = initial["yaw"] + relative_heading
        goal_x = initial["x"] + distance * math.cos(world_heading)
        goal_y = initial["y"] + distance * math.sin(world_heading)
        episodes.append(
            {
                "episode_id": episode_id,
                "simulation_reset_seed": reset_seed,
                "start_x": round(initial["x"], 9),
                "start_y": round(initial["y"], 9),
                "start_yaw": round(initial["yaw"], 9),
                "goal_x": round(goal_x, 9),
                "goal_y": round(goal_y, 9),
                "timeout": 20.0,
                "success_radius": 0.2,
                "success_hold_time": 0.5,
                "post_arrival_observation_time": 2.0,
            }
        )
        relative_headings[episode_id] = relative_heading

    available = set(relative_headings)
    smoke: dict[str, str] = {}
    for direction, target in SMOKE_DIRECTIONS_RAD.items():
        selected = min(
            available,
            key=lambda episode_id: abs(
                wrap_angle(relative_headings[episode_id] - target)
            ),
        )
        smoke[direction] = selected
        available.remove(selected)

    return {
        "schema_version": 1,
        "protocol_id": "week04_classical_v1",
        "protocol_status": "frozen",
        "episode_manifest_status": "frozen",
        "description": (
            "Week 4 paired Classical PointGoal benchmark. The 200 random Start/Goal "
            "EpisodeSpecs and controller parameters are frozen after smoke and "
            "quick-screen review."
        ),
        "task_definition": {
            "goal_type": "position_pointgoal",
            "input": "[delta_x_body, delta_y_body] equivalent to [distance, heading_error]",
            "navigator_output": "[vx, 0, wz]",
            "requires_final_yaw": False,
            "heading_error_note": "Heading error points toward the positional goal; it is not a final-yaw target.",
        },
        "runtime": {
            "initial_state_mode": "official_reset",
            "warmup_s": 1.0,
            "smoke_timeout_s": 5.0,
            "navigator_frequency_hz": 10.0,
        },
        "controllers": {
            "order": list(CONTROLLER_IDS),
            "shared_limits": {
                "max_forward_speed_mps": 0.25,
                "max_yaw_rate_radps": 0.5,
                "goal_tolerance_m": 0.15,
            },
            "naive_p": {
                "distance_gain": 0.8,
                "heading_gain": 1.2,
            },
            "constrained": {
                "distance_gain": 0.8,
                "heading_gain": 1.2,
                "min_turn_speed_mps": 0.05,
                "max_forward_accel_mps2": 0.5,
                "max_yaw_accel_radps2": 1.0,
                "heading_slowdown_floor": 0.2,
                "arrival_hold_time_s": 0.5,
            },
        },
        "generation": {
            "generator": "evaluation/generate_pointgoal_protocol.py",
            "generator_version": 1,
            "manifest_seed": manifest_seed,
            "episode_count": episode_count,
            "simulation_reset_seed_start": reset_seed_start,
            "start_distribution": "official_reset",
            "goal_distance_uniform_m": [1.0, 1.8],
            "goal_heading_uniform_rad": [-math.pi, math.pi],
        },
        "selections": {
            "smoke": smoke,
            "quick": [episode["episode_id"] for episode in episodes[:10]],
            "full": "all_200_episodes",
        },
        "quick_gate": {
            "maximum_fall_rate": 0.0,
            "maximum_invalid_state_rate": 0.0,
            "note": "Quick results are diagnostic and are never pooled with formal results.",
        },
        "episode_manifest_sha256": _manifest_sha256(episodes),
        "episodes": episodes,
    }
