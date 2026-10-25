"""训练诊断的手算、只读契约与真实入口测试；临时数据不作为研究结果。"""

import copy
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from core import Observation
from scripts.train_mappo import build_environment, build_model
from training import MAPPOConfig, MAPPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingDiagnosticsTests(unittest.TestCase):
    def _module(self):
        self.assertTrue((PROJECT_ROOT / "training/diagnostics.py").is_file(), "可重复训练诊断尚未实现")
        return importlib.import_module("training.diagnostics")

    def _sample(self, *, memory=True, value_coef=0.5):
        torch.manual_seed(7)
        model = build_model(8, "attention", coverage_memory=memory)
        trainer = MAPPOTrainer(model, MAPPOConfig(rollout_steps=4, value_loss_type="huber",
                                                 value_coef=value_coef), seed=19)
        rollout, _, _ = trainer.collect_rollout(build_environment(model, 2, 5))
        return trainer, rollout

    def test_gradient_norm_ratio_and_alignment_have_hand_checked_values(self):
        result = self._module().gradient_comparison(torch.tensor([3., 4.]), torch.tensor([-4., 3.]))
        self.assertEqual(result["actor_norm"], 5)
        self.assertEqual(result["weighted_value_norm"], 5)
        self.assertEqual(result["value_actor_ratio"], 1)
        self.assertEqual(result["cosine_similarity"], 0)

    def test_zero_actor_gradient_keeps_undefined_statistics_null(self):
        result = self._module().gradient_comparison(torch.zeros(2), torch.tensor([3., 4.]))
        self.assertIsNone(result["value_actor_ratio"])
        self.assertIsNone(result["cosine_similarity"])

    def test_coverage_statistics_and_sensitivity_use_history_without_mutating_inputs(self):
        model = build_model(8, "attention", coverage_memory=True)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.encoder.task_encoder[0].weight[0, 2] = 1
            model.encoder.output[0].weight[0, 16] = 1
            model.actor_mean[0].weight[0, 0] = 1
            model.actor_mean[2].weight[0, 0] = 1
        first = Observation("A", np.zeros(6), task_features=np.r_[0., 1., 1., np.zeros(8), np.ones(9)],
                            neighbor_feature_dim=5)
        second = replace(first, agent_id="B", task_features=np.r_[0., 1., np.zeros(9), np.ones(9)])
        before = first.task_features.copy()
        result = self._module().coverage_diagnostics(model, [first, second])
        self.assertAlmostEqual(result["covered_valid_fraction"], 1 / 18)
        self.assertEqual(result["unique_history_patterns"], 2)
        self.assertEqual(result["mask_groups_with_history_variation"], 1)
        self.assertGreater(result["history_zero_action_mean_abs_change"], 0.1)
        np.testing.assert_array_equal(first.task_features, before)

    def test_zero_initialized_history_columns_have_zero_action_sensitivity(self):
        model = build_model(8, "attention", coverage_memory=True)
        observation = Observation("A", np.ones(6), task_features=np.ones(20), neighbor_feature_dim=5)
        result = self._module().coverage_diagnostics(model, [observation])
        self.assertEqual(result["history_zero_action_max_abs_change"], 0)

    def test_legacy_coverage_diagnostic_is_unavailable_instead_of_zero(self):
        trainer, rollout = self._sample(memory=False)
        result = self._module().diagnose_rollout(trainer, rollout)
        self.assertIsNone(result["coverage"])

    def test_diagnosis_preserves_model_optimizer_gradients_random_state_and_rollout(self):
        trainer, rollout = self._sample()
        for parameter in trainer.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        weights = copy.deepcopy(trainer.model.state_dict())
        gradients = [parameter.grad.clone() for parameter in trainer.model.parameters()]
        optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        observations = trainer._observations
        actions = [step.actions.clone() for step in rollout]
        steps = trainer.environment_steps
        result = self._module().diagnose_rollout(trainer, rollout)
        self.assertEqual(result["rollout_steps"], 4)
        self.assertLess(result["max_preupdate_ratio_error"], 1e-4)
        self.assertTrue(all(torch.equal(value, trainer.model.state_dict()[key]) for key, value in weights.items()))
        self.assertTrue(all(torch.equal(before, parameter.grad) for before, parameter in zip(gradients, trainer.model.parameters())))
        self.assertEqual(optimizer, trainer.optimizer.state_dict())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(observations, trainer._observations)
        self.assertEqual(steps, trainer.environment_steps)
        self.assertTrue(all(torch.equal(before, step.actions) for before, step in zip(actions, rollout)))

    def test_changed_old_probabilities_are_visible_in_ratio_diagnostic(self):
        trainer, rollout = self._sample()
        for step in rollout:
            step.log_probs += 0.1
        result = self._module().diagnose_rollout(trainer, rollout)
        self.assertAlmostEqual(result["max_preupdate_ratio_error"], 0.09516258, places=5)

    def test_disabled_value_loss_has_zero_shared_gradient_and_no_cosine(self):
        trainer, rollout = self._sample(value_coef=0)
        result = self._module().diagnose_rollout(trainer, rollout)
        self.assertEqual(result["shared_gradients"]["weighted_value_norm"], 0)
        self.assertIsNone(result["shared_gradients"]["cosine_similarity"])

    def test_empty_rollout_is_rejected(self):
        trainer, _ = self._sample()
        with self.assertRaises(ValueError):
            self._module().diagnose_rollout(trainer, [])

    def test_motion_shift_requires_explicit_opt_in_and_is_recorded(self):
        module = importlib.import_module("scripts.diagnose_training")
        trainer, _ = self._sample()
        environment = build_environment(trainer.model, 2, 5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = trainer.save_checkpoint(root / "model.pt")
            config = root / "run_config.json"
            config.write_text(json.dumps(dict(
                motion_profile="command_velocity_v1", model=trainer.model.architecture_config(),
                training=asdict(trainer.config), environment=asdict(environment.config),
                communication=asdict(environment.communication_config), arguments=dict(
                    seed=19, agents=2, max_steps=5, coverage_memory=True, observation_mode="communicated",
                    neighbor_state_ttl=1.0, link_fault=True, node_fault=True),
            )), encoding="utf-8")
            with self.assertRaises(ValueError):
                module.run_diagnostic(checkpoint, config, seed=30000)
            report = module.run_diagnostic(checkpoint, config, seed=30000, allow_motion_shift=True)
            self.assertEqual(report["training_motion_profile"], "command_velocity_v1")
            self.assertNotEqual(report["motion_profile"], report["training_motion_profile"])
            self.assertTrue(report["motion_shift"])

    def test_cli_is_reproducible_and_refuses_overwrite_or_mismatched_configuration(self):
        self._module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            result = subprocess.run([
                sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/train_mappo.py"),
                "--coverage-memory", "--updates", "1", "--rollout-steps", "4", "--update-epochs", "1",
                "--agents", "2", "--max-steps", "4", "--hidden-dim", "8", "--output-dir", str(training),
            ], capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoint = training / "mappo_checkpoint.pt"
            source = checkpoint.read_bytes()
            config = training / "run_config.json"
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/diagnose_training.py"),
                       "--checkpoint", str(checkpoint), "--run-config", str(config), "--seed", "30000"]
            reports = []
            for name in ("first", "second"):
                completed = subprocess.run(command + ["--output-dir", str(root / name)],
                                           capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                reports.append((root / name / "training_diagnostics.json").read_bytes())
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(source, checkpoint.read_bytes())
            rejected = subprocess.run(command + ["--output-dir", str(root / "first")],
                                      capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(reports[0], (root / "first/training_diagnostics.json").read_bytes())
            data = json.loads(config.read_text(encoding="utf-8"))
            data["environment"]["coverage_memory"] = False
            config.write_text(json.dumps(data), encoding="utf-8")
            rejected = subprocess.run(command + ["--output-dir", str(root / "wrong")],
                                      capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "wrong").exists())


if __name__ == "__main__":
    unittest.main()
