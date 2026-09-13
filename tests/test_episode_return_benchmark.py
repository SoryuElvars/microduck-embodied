import unittest

from evaluation.episode_return_benchmark import aggregate_episodes, describe


def _episode(name: str, value: float, terminated: bool = False) -> dict:
    return {
        "command_name": name,
        "command": {"vx": 0.3, "vy": 0.0, "wz": 0.0},
        "terminated": terminated,
        "time_out": not terminated,
        "episode_return": value,
        "mean_reward_rate": value / 10.0,
        "episode_duration_s": 10.0,
        "reward_terms": {
            "tracking": {"return": value * 0.75, "mean_rate": value * 0.075},
            "penalty": {"return": value * 0.25, "mean_rate": value * 0.025},
        },
    }


class EpisodeReturnBenchmarkTests(unittest.TestCase):
    def test_describe_uses_population_std(self):
        stats = describe([1.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["std_population"], 1.0)

    def test_aggregate_keeps_return_and_term_breakdown(self):
        result = aggregate_episodes([_episode("forward", 10.0), _episode("forward", 20.0)])
        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(result["time_out_count"], 2)
        self.assertEqual(result["terminated_count"], 0)
        self.assertEqual(result["episode_return"]["mean"], 15.0)
        self.assertEqual(result["reward_terms"]["tracking"]["return"]["mean"], 11.25)

    def test_aggregate_counts_early_termination(self):
        result = aggregate_episodes([_episode("forward", 5.0, terminated=True)])
        self.assertEqual(result["terminated_count"], 1)
        self.assertEqual(result["time_out_count"], 0)

    def test_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            describe([])
        with self.assertRaises(ValueError):
            aggregate_episodes([])


if __name__ == "__main__":
    unittest.main()
