from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evaluation.pointgoal_benchmark import (
    aggregate_controller,
    load_progress,
    paired_comparison,
    run_episode,
    select_episode_specs,
    write_progress,
)
from evaluation.pointgoal_protocol import EpisodeSpec, load_pointgoal_protocol
from navigation.mujoco_backend import BackendStep
from navigation.types import RobotState, VelocityCommand


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "pointgoal_protocols"
    / "week04_classical_v1.json"
)


def state(*, x: float, y: float, yaw: float, time_s: float) -> RobotState:
    return RobotState(
        time_s=time_s,
        x_world_m=x,
        y_world_m=y,
        z_world_m=0.125,
        yaw_rad=yaw,
        roll_rad=0.0,
        pitch_rad=0.0,
        vx_body_mps=0.0,
        vy_body_mps=0.0,
        wz_body_radps=0.0,
        tilt_rad=0.0,
        fallen=False,
    )


class KinematicBackend:
    control_dt_s = 0.1

    def __init__(self, spec: EpisodeSpec) -> None:
        self.spec = spec
        self.current = state(
            x=spec.start_x,
            y=spec.start_y,
            yaw=spec.start_yaw,
            time_s=0.0,
        )

    def reset(self, episode_seed: int, initial_state_mode: str) -> RobotState:
        self.assert_reset_arguments(episode_seed, initial_state_mode)
        self.current = state(
            x=self.spec.start_x,
            y=self.spec.start_y,
            yaw=self.spec.start_yaw,
            time_s=0.0,
        )
        return self.current

    def assert_reset_arguments(self, episode_seed: int, initial_state_mode: str) -> None:
        if episode_seed != self.spec.simulation_reset_seed:
            raise AssertionError("wrong reset seed")
        if initial_state_mode != "official_reset":
            raise AssertionError("wrong reset mode")

    def observe(self) -> RobotState:
        return self.current

    def step(self, command: VelocityCommand) -> BackendStep:
        dt = self.control_dt_s
        yaw = self.current.yaw_rad + command.wz_radps * dt
        x = self.current.x_world_m + command.vx_mps * math.cos(yaw) * dt
        y = self.current.y_world_m + command.vx_mps * math.sin(yaw) * dt
        self.current = RobotState(
            **{
                **self.current.__dict__,
                "time_s": self.current.time_s + dt,
                "x_world_m": x,
                "y_world_m": y,
                "yaw_rad": yaw,
                "vx_body_mps": command.vx_mps,
                "wz_body_radps": command.wz_radps,
            }
        )
        return BackendStep(self.current, (), ())


def episode_result(
    episode_id: str,
    *,
    success: bool,
    final_distance: float,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "success": success,
        "reached_success_hold": success,
        "fall": False,
        "timeout": not success,
        "invalid_state": False,
        "final_distance_m": final_distance,
        "path_length_to_arrival_m": 1.0 if success else None,
        "path_efficiency": 0.8 if success else None,
        "completion_time_s": 4.0 if success else None,
        "vx_saturation_fraction": 0.2,
        "wz_saturation_fraction": 0.1,
        "post_arrival_stop_drift_m": 0.01 if success else None,
        "post_arrival_max_drift_m": 0.02 if success else None,
        "redeparted_after_arrival": False,
    }


class PointGoalBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_pointgoal_protocol(PROTOCOL_PATH)

    def test_smoke_and_quick_use_frozen_shared_subsets(self) -> None:
        smoke = select_episode_specs(self.protocol, "smoke")
        quick = select_episode_specs(self.protocol, "quick")

        self.assertEqual(len(smoke), 4)
        self.assertEqual(len(quick), 10)
        self.assertEqual(
            [episode.episode_id for episode in quick],
            list(self.protocol.quick_episode_ids),
        )

    def test_full_is_blocked_until_controller_protocol_is_frozen(self) -> None:
        draft_protocol = replace(self.protocol, protocol_status="draft")
        with self.assertRaisesRegex(ValueError, "protocol_status='frozen'"):
            select_episode_specs(draft_protocol, "full")

        self.assertEqual(len(select_episode_specs(self.protocol, "full")), 200)

    def test_episode_runs_post_arrival_observation_and_stops(self) -> None:
        spec = EpisodeSpec(
            episode_id="unit_front",
            simulation_reset_seed=7,
            start_x=0.0,
            start_y=0.0,
            start_yaw=0.0,
            goal_x=0.5,
            goal_y=0.0,
            timeout=5.0,
            success_radius=0.2,
            success_hold_time=0.2,
            post_arrival_observation_time=0.3,
        )
        backend = KinematicBackend(spec)
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(
                backend=backend,  # type: ignore[arg-type]
                protocol=self.protocol,
                spec=spec,
                controller_id="naive_p",
                output_dir=Path(directory),
            )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["post_arrival_duration_s"], 0.3)
        self.assertFalse(result["redeparted_after_arrival"])
        self.assertGreater(result["post_arrival_stop_drift_m"], 0.0)
        self.assertLessEqual(
            result["final_distance_m"], self.protocol.naive_p.goal_tolerance_m
        )

    def test_paired_comparison_reports_constrained_minus_naive(self) -> None:
        comparison = paired_comparison(
            {
                "naive_p": [episode_result("e0", success=False, final_distance=0.8)],
                "constrained": [episode_result("e0", success=True, final_distance=0.1)],
            }
        )

        self.assertEqual(comparison["aggregate"]["constrained_success_wins"], 1)
        pair = comparison["episodes"][0]
        self.assertEqual(pair["success_delta_constrained_minus_naive"], 1)
        self.assertAlmostEqual(
            pair["final_distance_m_delta_constrained_minus_naive"], -0.7
        )

    def test_aggregate_reports_post_arrival_metrics_separately(self) -> None:
        aggregate = aggregate_controller(
            [episode_result("e0", success=True, final_distance=0.1)]
        )

        self.assertEqual(aggregate["success_rate"], 1.0)
        self.assertEqual(aggregate["post_arrival_redeparture_rate"], 0.0)
        self.assertEqual(aggregate["median_post_arrival_stop_drift_m"], 0.01)

    def test_progress_rejects_a_different_run_signature(self) -> None:
        spec = self.protocol.episodes[0]
        result = episode_result(spec.episode_id, success=True, final_distance=0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            write_progress(
                path,
                signature="signature-a",
                controller_id="naive_p",
                spec=spec,
                episode=result,
            )
            loaded = load_progress(
                path,
                signature="signature-a",
                controller_id="naive_p",
                episode_id=spec.episode_id,
            )
            self.assertEqual(loaded, json.loads(json.dumps(result)))
            with self.assertRaisesRegex(ValueError, "stale progress signature"):
                load_progress(
                    path,
                    signature="signature-b",
                    controller_id="naive_p",
                    episode_id=spec.episode_id,
                )


if __name__ == "__main__":
    unittest.main()
