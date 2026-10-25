"""边界运动、轨迹记录与真实配对评估入口测试；布置状态仅用于回归。"""

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from agents import RuleBasedSearchAgent
from core import Action, AgentAttributes
from environments import Continuous2DConfig, Continuous2DSearchEnv
from evaluation.evaluator import ExperimentScenario, run_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BoundaryMotionTests(unittest.TestCase):
    def environment(self, position, *, dt=0.5, energy_cost=0):
        environment = Continuous2DSearchEnv(Continuous2DConfig(
            width=10, height=10, dt=dt, max_steps=4, energy_cost=energy_cost,
            target_positions=((0, 0),), collision_distance=0,
        ))
        agent = RuleBasedSearchAgent("A", AgentAttributes(max_speed=1, sensor_range=0.01, communication_range=20))
        environment.add_agent(agent)
        environment.reset(seed=0)
        agent.state.position = np.array(position, dtype=float)
        agent.state.yaw = 0.3
        return environment, agent

    def test_outward_commands_at_each_wall_have_zero_motion_velocity(self):
        cases = (([0, 5], [-1, 0]), ([10, 5], [1, 0]), ([5, 0], [0, -1]), ([5, 10], [0, 1]))
        for position, command in cases:
            with self.subTest(position=position):
                environment, agent = self.environment(position)
                environment.apply_action("A", Action.continuous("A", command))
                np.testing.assert_array_equal(agent.state.position, position)
                np.testing.assert_array_equal(agent.state.velocity, [0, 0])

    def test_partial_crossing_records_step_average_displacement_velocity(self):
        environment, agent = self.environment([9.75, 5])
        environment.apply_action("A", Action.continuous("A", [1, 0]))
        np.testing.assert_array_equal(agent.state.position, [10, 5])
        np.testing.assert_allclose(agent.state.velocity, [0.5, 0])

    def test_tangential_component_survives_and_yaw_follows_real_motion(self):
        environment, agent = self.environment([10, 5])
        environment.apply_action("A", Action.continuous("A", [1, 1]))
        np.testing.assert_allclose(agent.state.velocity, [0, 2 ** -0.5])
        self.assertAlmostEqual(agent.state.yaw, np.pi / 2)

    def test_stationary_corner_preserves_previous_heading(self):
        environment, agent = self.environment([10, 10])
        environment.apply_action("A", Action.continuous("A", [1, 1]))
        np.testing.assert_array_equal(agent.state.velocity, [0, 0])
        self.assertEqual(agent.state.yaw, 0.3)

    def test_interior_motion_and_noop_keep_expected_behavior(self):
        environment, agent = self.environment([4, 5])
        environment.apply_action("A", Action.continuous("A", [0.6, 0.8]))
        np.testing.assert_allclose(agent.state.position, [4.3, 5.4])
        np.testing.assert_allclose(agent.state.velocity, [0.6, 0.8])
        environment.apply_action("A", Action.noop("A"))
        np.testing.assert_array_equal(agent.state.velocity, [0, 0])

    def test_energy_keeps_command_effort_cost_when_wall_blocks_motion(self):
        environment, agent = self.environment([10, 5], energy_cost=0.2)
        environment.apply_action("A", Action.continuous("A", [1, 0]))
        self.assertAlmostEqual(agent.state.energy, 0.9)
        np.testing.assert_array_equal(agent.state.velocity, [0, 0])

    def test_self_and_received_neighbor_observations_use_motion_velocity(self):
        environment, agent = self.environment([10, 5])
        environment.add_agent(RuleBasedSearchAgent("B", agent.attributes))
        environment.get_agent("B").state.position = np.array([8., 5.])
        result = environment.step({"A": Action.continuous("A", [1, 0]), "B": Action.noop("B")})
        np.testing.assert_array_equal(result.observations["A"].self_features[2:4], [0, 0])
        self.assertEqual(len(result.observations["B"].neighbors), 1)
        np.testing.assert_array_equal(result.observations["B"].neighbors[0].features[2:4], [0, 0])

    def test_recovered_node_reports_zero_velocity_until_its_next_action(self):
        environment, agent = self.environment([2, 2])
        environment.add_agent(RuleBasedSearchAgent("B", agent.attributes))
        environment.get_agent("B").state.position = np.array([5., 5.])
        environment.schedule_node_fault("A", start_time=0.5, duration=0.5)
        environment.step({"A": Action.continuous("A", [1, 0]), "B": Action.noop("B")})
        self.assertFalse(agent.state.active)
        np.testing.assert_array_equal(agent.state.position, [2.5, 2])
        result = environment.step({"B": Action.noop("B")})
        self.assertTrue(agent.state.active)
        np.testing.assert_array_equal(agent.state.position, [2.5, 2])
        np.testing.assert_array_equal(agent.state.velocity, [0, 0])
        np.testing.assert_array_equal(result.observations["A"].self_features[2:4], [0, 0])


class MotionTrajectoryTests(unittest.TestCase):
    def module(self):
        self.assertTrue((PROJECT_ROOT / "evaluation/trajectory.py").is_file(), "运动轨迹记录尚未实现")
        return importlib.import_module("evaluation.trajectory")

    def test_trajectory_recording_does_not_change_episode_metrics(self):
        self.assertIn("trajectory", inspect.signature(run_episode).parameters, "评估轨迹接入尚未实现")
        scenario = ExperimentScenario("nominal", 2, episodes=1, base_seed=50000, max_steps=5)
        original = run_episode(scenario, policy="rule", episode=0)
        rows = []
        recorded = run_episode(scenario, policy="rule", episode=0, trajectory=rows)
        self.assertEqual(original, recorded)
        self.assertEqual(len(rows), original.steps)
        self.assertEqual(rows[0]["step"], 1)
        self.assertEqual(len(rows[0]["agents"]), 2)

    def test_recorder_distinguishes_command_from_displacement_at_wall(self):
        module = self.module()
        environment, agent = BoundaryMotionTests().environment([10, 5])
        before = {"A": agent.state.position.copy()}
        actions = {"A": Action.continuous("A", [1, 0])}
        result = environment.step(actions)
        row = module.motion_snapshot(environment, actions, before, result.info)
        state = row["agents"][0]
        np.testing.assert_array_equal(state["command_velocity"], [1, 0])
        np.testing.assert_array_equal(state["displacement_velocity"], [0, 0])
        self.assertTrue(state["boundary_clipped"])
        self.assertTrue(state["on_boundary"])
        self.assertIsNone(row["min_active_pair_distance"])

    def test_offline_nodes_are_excluded_from_velocity_mismatch_counts(self):
        module = self.module()
        row = dict(step=1, timestamp=0.5, collision_count=0, command_spread_rms=0,
                   min_active_pair_distance=None, agents=[dict(
                       agent_id="A", had_action=True, active=False, on_boundary=False, boundary_clipped=False,
                       command_velocity=[1, 0], displacement_velocity=[1, 0], state_velocity=[0, 0])])
        result = module.summarize_motion([row])
        self.assertEqual(result["velocity_checked_samples"], 0)
        self.assertIsNone(result["velocity_mismatch_fraction"])

    def test_recovered_nodes_without_action_are_included_in_velocity_checks(self):
        module = self.module()
        row = dict(step=1, timestamp=1.0, collision_count=0, command_spread_rms=0,
                   min_active_pair_distance=None, agents=[dict(
                       agent_id="A", had_action=False, active=True, on_boundary=False, boundary_clipped=False,
                       command_velocity=None, displacement_velocity=[0, 0], state_velocity=[1, 0])])
        result = module.summarize_motion([row])
        self.assertEqual(result["velocity_checked_samples"], 1)
        self.assertEqual(result["velocity_mismatch_fraction"], 1)

    def test_real_motion_cli_saves_hashes_and_refuses_overwrite(self):
        self.module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/evaluate_motion.py"),
                       "--episodes", "1", "--agent-counts", "2", "--max-steps", "4", "--seed", "50000",
                       "--trajectory-episodes", "1", "--packet-loss-rate", "0.2", "--output-dir", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            provenance = json.loads((output / "evaluation_provenance.json").read_text(encoding="utf-8"))
            files = provenance["result_sha256"]
            self.assertIn("motion_metrics.jsonl", files)
            self.assertEqual(len(list((output / "trajectories").glob("*.jsonl"))), 6)
            self.assertTrue(list((output / "trajectories").glob("*.png")))
            for name, expected in files.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            previous = (output / "episode_metrics.jsonl").read_bytes()
            rejected = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(previous, (output / "episode_metrics.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
