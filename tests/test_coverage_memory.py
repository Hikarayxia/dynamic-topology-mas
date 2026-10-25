"""本机覆盖记忆与模型兼容契约；手工布置的位置仅用于测试，不作为实验数据。"""

from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from agents import RuleBasedSearchAgent
from core import Action, AgentAttributes, Observation
from environments import Continuous2DConfig, Continuous2DSearchEnv
from algorithms import SharedGraphActorCritic
from training import MAPPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocalCoverageTests(unittest.TestCase):
    def _memory(self, width=12, height=12):
        self.assertTrue((PROJECT_ROOT / "environments/coverage_memory.py").is_file(), "本机覆盖记忆尚未实现")
        return importlib.import_module("environments.coverage_memory").LocalCoverageMemory(width, height)

    def _config(self, **kwargs):
        self.assertIn("coverage_memory", Continuous2DConfig.__dataclass_fields__, "环境记忆开关尚未实现")
        return Continuous2DConfig(coverage_memory=True, **kwargs)

    def _environment(self, count=2, **kwargs):
        config = self._config(width=12, height=12, max_steps=10, energy_cost=0,
                              collision_penalty=0, target_positions=((11, 11),), **kwargs)
        environment = Continuous2DSearchEnv(config)
        for index in range(count):
            environment.add_agent(RuleBasedSearchAgent(
                f"A{index}", AgentAttributes(max_speed=1, sensor_range=0.001, communication_range=50)))
        environment.reset(seed=17)
        return environment

    def test_local_patch_has_hand_checked_order_and_boundary_mask(self):
        memory = self._memory()
        memory.observe([2, 2], 0)
        result = memory.features([6, 2])
        np.testing.assert_array_equal(result[:9], [0, 0, 0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result[9:], [0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_sensor_circle_marks_only_reached_cell_centers(self):
        memory = self._memory()
        memory.observe([2, 2], 4)
        np.testing.assert_array_equal(memory.features([6, 6])[:9], [1, 1, 0, 1, 0, 0, 0, 0, 0])

    def test_history_persists_and_returned_features_do_not_alias_storage(self):
        memory = self._memory()
        memory.observe([2, 2], 0)
        memory.observe([6, 2], 0)
        result = memory.features([6, 2])
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 1)
        result[:] = 0
        self.assertEqual(memory.features([6, 2])[3], 1)

    def test_partial_boundary_cell_uses_actual_cell_center(self):
        memory = self._memory(width=10, height=10)
        memory.observe([9, 9], 0)
        result = memory.features([10, 10])
        self.assertEqual(result[4], 1)
        self.assertEqual(result[9 + 4], 1)
        self.assertEqual(result[9 + 8], 0)

    def test_invalid_positions_and_radii_are_rejected(self):
        memory = self._memory()
        for position, radius in (([-1, 2], 1), ([float("nan"), 0], 1), ([2, 2], -1), ([2, 2], float("inf"))):
            with self.subTest(position=position, radius=radius), self.assertRaises(ValueError):
                memory.observe(position, radius)

    def test_default_observations_keep_legacy_dimensions(self):
        environment = Continuous2DSearchEnv(Continuous2DConfig(max_steps=2))
        environment.add_agent(RuleBasedSearchAgent("A"))
        observations = environment.reset(seed=9)
        self.assertEqual(observations["A"].task_features.shape, (2,))

    def test_enabled_dimensions_are_independent_of_agent_count(self):
        for count in (2, 5):
            environment = self._environment(count)
            observations = environment._build_observations()
            self.assertTrue(all(obs.task_features.shape == (20,) for obs in observations.values()))

    def test_teammate_coverage_is_not_shared_even_when_connected(self):
        environment = self._environment()
        environment.get_agent("A0").state.position = np.array([2., 2.])
        environment.get_agent("A1").state.position = np.array([6., 6.])
        result = environment.step({name: Action.noop(name) for name in environment.agents})
        self.assertEqual(result.observations["A1"].task_features[2], 0)
        self.assertEqual(result.observations["A1"].task_features[2 + 4], 1)
        self.assertEqual(result.observations["A0"].task_features[2 + 4], 1)

    def test_repeated_queries_do_not_change_memory(self):
        environment = self._environment()
        before = environment._build_observations()["A0"].task_features.copy()
        after = environment._build_observations()["A0"].task_features
        np.testing.assert_array_equal(before, after)

    def test_reset_clears_previous_episode_history(self):
        environment = self._environment()
        environment.get_agent("A0").state.position = np.array([2., 2.])
        environment.step({name: Action.noop(name) for name in environment.agents})
        environment.reset(seed=17)
        environment.get_agent("A0").state.position = np.array([6., 6.])
        self.assertEqual(environment._build_observations()["A0"].task_features[2], 0)

    def test_node_offline_freezes_history_and_recovery_keeps_it(self):
        environment = self._environment()
        environment.get_agent("A0").state.position = np.array([2., 2.])
        environment.step({name: Action.noop(name) for name in environment.agents})
        environment.set_agent_active("A0", False)
        environment.get_agent("A0").state.position = np.array([6., 6.])
        environment.step({"A1": Action.noop("A1")})
        environment.set_agent_active("A0", True)
        features = environment._build_observations()["A0"].task_features[2:]
        self.assertEqual(features[0], 1)
        self.assertEqual(features[4], 0)

    def test_removed_and_readmitted_agent_does_not_inherit_memory(self):
        environment = self._environment()
        environment.get_agent("A0").state.position = np.array([2., 2.])
        environment.step({name: Action.noop(name) for name in environment.agents})
        agent = environment.remove_agent("A0")
        agent.state.position = np.array([6., 6.])
        environment.add_agent(agent)
        self.assertEqual(environment._build_observations()["A0"].task_features[2], 0)

    def test_target_locations_do_not_enter_local_memory(self):
        first = self._environment()
        second = Continuous2DSearchEnv(replace(first.config, target_positions=((0, 11),)))
        for index in range(2):
            second.add_agent(RuleBasedSearchAgent(f"A{index}", first.get_agent(f"A{index}").attributes))
        second.reset(seed=17)
        for name in first.agents:
            np.testing.assert_array_equal(first._build_observations()[name].task_features[2:],
                                          second._build_observations()[name].task_features[2:])


class CoverageModelTests(unittest.TestCase):
    def _model(self, enabled):
        module = importlib.import_module("scripts.train_mappo")
        self.assertTrue(hasattr(module, "build_model"), "兼容初始化入口尚未实现")
        return module.build_model(8, "attention", coverage_memory=enabled)

    def test_initial_policy_and_rng_match_when_new_inputs_start_at_zero_weight(self):
        torch.manual_seed(3)
        old = self._model(False)
        old_next = torch.rand(4)
        torch.manual_seed(3)
        new = self._model(True)
        new_next = torch.rand(4)
        self.assertTrue(torch.equal(old_next, new_next))
        before = Observation("A", np.ones(6), task_features=np.array([0.2, 0.8]), neighbor_feature_dim=5)
        after = replace(before, task_features=np.r_[before.task_features, np.ones(18)])
        self.assertTrue(torch.allclose(old.act([before], deterministic=True)[0],
                                       new.act([after], deterministic=True)[0], atol=1e-7))
        self.assertEqual(new.architecture_config()["observation_profile"], "local_coverage_v1")
        self.assertNotIn("observation_profile", old.architecture_config())

    def test_zero_initialized_memory_inputs_receive_trainable_gradients(self):
        torch.manual_seed(4)
        model = self._model(True)
        observation = Observation("A", np.ones(6), task_features=np.ones(20), neighbor_feature_dim=5)
        model.encode([observation]).square().sum().backward()
        gradient = model.encoder.task_encoder[0].weight.grad[:, 2:]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_manual_checkpoint_restore_rejects_profile_mismatch_before_mutation(self):
        model = self._model(True)
        with tempfile.TemporaryDirectory() as directory:
            path = MAPPOTrainer(model).save_checkpoint(Path(directory) / "model.pt")
            # 维度相同但含义不同的自定义旧输入，不能静默载入覆盖记忆权重。
            wrong = MAPPOTrainer(SharedGraphActorCritic(6, 5, 20, hidden_dim=8, aggregation="attention"))
            before = {key: value.clone() for key, value in wrong.model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "观测"):
                wrong.load_checkpoint(path)
            self.assertTrue(all(torch.equal(value, wrong.model.state_dict()[key]) for key, value in before.items()))

    def test_real_cli_checkpoint_auto_selects_memory_observations_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = subprocess.run([
                sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/train_mappo.py"),
                "--coverage-memory", "--value-loss-type", "huber", "--updates", "1", "--rollout-steps", "8",
                "--update-epochs", "1", "--agents", "2", "--hidden-dim", "8", "--max-steps", "8",
                "--output-dir", str(root / "training"),
            ], capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertEqual(train.returncode, 0, train.stderr)
            config = json.loads((root / "training/run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["observation_profile"], "local_coverage_v1")
            self.assertTrue(config["environment"]["coverage_memory"])
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/evaluate_policies.py"),
                       "--checkpoint", str(root / "training/mappo_checkpoint.pt"), "--episodes", "1",
                       "--agent-counts", "2,3", "--max-steps", "8"]
            evaluation = subprocess.run(command + ["--output-dir", str(root / "evaluation")],
                                        capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
            scenarios = json.loads((root / "evaluation/scenario_config.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["coverage_memory"] for row in scenarios))
            rejected = subprocess.run(command + ["--no-coverage-memory", "--output-dir", str(root / "wrong")],
                                      capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "wrong").exists())

    def test_small_ablation_runs_and_resumes_with_memory_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/run_ablation.py"),
                       "--coverage-memory", "--value-loss-type", "huber", "--aggregations", "attention",
                       "--seeds", "0", "--updates", "1", "--rollout-steps", "8", "--update-epochs", "1",
                       "--train-agents", "2", "--agent-counts", "2,3", "--max-steps", "8",
                       "--hidden-dim", "8", "--episodes", "1", "--output-dir", directory]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(directory)
            report = json.loads((root / "summary/ablation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["completed_runs"], 1)
            path = root / "attention/seed_0/training/mappo_checkpoint.pt"
            before = path.read_bytes()
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
