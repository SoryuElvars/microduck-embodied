from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.plot_pointgoal_robustness import load_factor_series


def condition(condition_id: str, factor: str, **values: object) -> dict[str, object]:
    return {
        "condition": {
            "condition_id": condition_id,
            "factor": factor,
            "foot_friction": None,
            "actuator_delay_ms": 0,
            "bam_voltage_scale": 1.0,
            "scene_xml_path": None,
            **values,
        },
        "aggregate": {
            "success_rate": 1.0,
            "fall_rate": 0.0,
            "median_final_distance_m": 0.1,
        },
    }


class PointGoalRobustnessPlotTests(unittest.TestCase):
    def test_series_includes_nominal_as_factor_baseline(self) -> None:
        document = {
            "benchmark_kind": "classical_pointgoal_onnx_bam_ood",
            "conditions": [
                condition("nominal", "nominal"),
                condition("delay_40ms", "actuator_delay", actuator_delay_ms=40),
                condition("motor_80pct", "motor_strength_proxy", bam_voltage_scale=0.8),
                condition("friction_0p3", "foot_friction", foot_friction=0.3),
                condition("backlash", "backlash", scene_xml_path="backlash.xml"),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            series = load_factor_series(path)

        self.assertEqual([row["x"] for row in series["actuator_delay"]], [0.0, 40.0])
        self.assertEqual(
            [row["x"] for row in series["motor_strength_proxy"]],
            [80.0, 100.0],
        )
        self.assertEqual(
            [row["x"] for row in series["foot_friction"]],
            [0.3, 1.0],
        )
        self.assertEqual([row["x"] for row in series["backlash"]], ["Normal", "2 deg"])


if __name__ == "__main__":
    unittest.main()
