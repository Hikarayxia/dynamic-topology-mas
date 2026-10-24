"""价值损失稳健化与既有 PPO 契约测试；合成回报只用于临时单元测试。"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from algorithms import SharedGraphActorCritic
from core import Observation
from environments import Continuous2DConfig
from evaluation.ablation import _training_signature
from interaction import CommunicationConfig
from training import MAPPOConfig, MAPPOTrainer
from training.mappo import RolloutStep


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingStabilityTests(unittest.TestCase):
    def _trainer(self, loss_type):
        self.assertIn("value_loss_type", MAPPOConfig.__dataclass_fields__, "可选稳健价值损失尚未实现")
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8)
        return MAPPOTrainer(model, replace(MAPPOConfig(update_epochs=1), value_loss_type=loss_type))

    @staticmethod
    def _observation():
        return Observation("A", np.zeros(6), task_features=np.zeros(2), neighbor_feature_dim=5)

    def test_mse_keeps_existing_quadratic_loss(self):
        trainer = self._trainer("mse")
        value = trainer.value_loss(torch.tensor([0.5, 2.0]), torch.zeros(2))
        self.assertAlmostEqual(float(value), 2.125)

    def test_huber_uses_quadratic_near_zero_and_linear_tail(self):
        trainer = self._trainer("huber")
        value = trainer.value_loss(torch.tensor([0.5, 2.0]), torch.zeros(2))
        self.assertAlmostEqual(float(value), 0.8125)

    def test_huber_bounds_large_error_gradient_without_changing_target(self):
        trainer = self._trainer("huber")
        predictions = torch.tensor([100.0], requires_grad=True)
        targets = torch.tensor([0.0])
        trainer.value_loss(predictions, targets).backward()
        self.assertAlmostEqual(float(predictions.grad[0]), 1.0)
        self.assertEqual(float(targets[0]), 0.0)

    def test_update_actually_uses_selected_value_loss(self):
        for loss_type, expected in (("mse", 100.0), ("huber", 9.5)):
            with self.subTest(loss_type=loss_type):
                trainer = self._trainer(loss_type)
                with torch.no_grad():
                    for parameter in trainer.model.critic.parameters():
                        parameter.zero_()
                    actions, log_probs, value = trainer.model.act([self._observation()])
                rollout = [RolloutStep(("A",), (self._observation(),), actions, log_probs, 10.0, True, float(value))]
                result = trainer.update(rollout)
                self.assertAlmostEqual(result["value_loss"], expected)
                self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_unknown_value_loss_is_rejected_before_training(self):
        self.assertIn("value_loss_type", MAPPOConfig.__dataclass_fields__, "可选稳健价值损失尚未实现")
        with self.assertRaises(ValueError):
            replace(MAPPOConfig(), value_loss_type="invalid")

    def test_checkpoint_preserves_huber_configuration(self):
        trainer = self._trainer("huber")
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save_checkpoint(Path(directory) / "model.pt")
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["config"]["value_loss_type"], "huber")
            restored = self._trainer("huber")
            restored.load_checkpoint(path)
            self.assertTrue(all(torch.equal(value, restored.model.state_dict()[name])
                                for name, value in trainer.model.state_dict().items()))

    def test_train_cli_saves_selected_value_loss_and_finite_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/train_mappo.py"),
                "--updates", "1", "--rollout-steps", "4", "--update-epochs", "1", "--agents", "2",
                "--max-steps", "4", "--hidden-dim", "8", "--value-loss-type", "huber",
                "--output-dir", directory,
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(directory)
            config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["arguments"]["value_loss_type"], "huber")
            self.assertEqual(config["training"]["value_loss_type"], "huber")
            checkpoint = torch.load(root / "mappo_checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["config"]["value_loss_type"], "huber")
            metrics = json.loads((root / "training_metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(metrics["environment_steps"], 4)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_checkpoint_rejects_loss_change_before_mutating_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._trainer("huber")
            path = source.save_checkpoint(Path(directory) / "model.pt")
            target = self._trainer("mse")
            before = {name: value.clone() for name, value in target.model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "价值损失"):
                target.load_checkpoint(path)
            self.assertTrue(all(torch.equal(value, target.model.state_dict()[name])
                                for name, value in before.items()))

    def test_legacy_mse_checkpoint_without_new_field_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._trainer("mse")
            path = source.save_checkpoint(Path(directory) / "model.pt")
            saved = torch.load(path, weights_only=True)
            saved["config"].pop("value_loss_type")
            torch.save(saved, path)
            restored = self._trainer("mse")
            restored.load_checkpoint(path)
            self.assertTrue(all(torch.equal(value, restored.model.state_dict()[name])
                                for name, value in source.model.state_dict().items()))

    def test_ablation_forwards_huber_and_resumes_matching_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/run_ablation.py"),
                       "--aggregations", "attention", "--seeds", "0", "--value-loss-type", "huber",
                       "--output-dir", directory, "--dry-run"]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((Path(directory) / "ablation_plan.json").read_text(encoding="utf-8"))
            train = plan[0]["train"]
            self.assertEqual(train[train.index("--value-loss-type") + 1], "huber")
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)

    def test_legacy_default_ablation_plan_without_loss_flag_still_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "scripts/run_ablation.py"),
                       "--aggregations", "attention", "--seeds", "0", "--output-dir", directory, "--dry-run"]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            path = Path(directory) / "ablation_plan.json"
            plan = json.loads(path.read_text(encoding="utf-8"))
            train = plan[0]["train"]
            if "--value-loss-type" in train:
                index = train.index("--value-loss-type")
                del train[index:index + 2]
            path.write_text(json.dumps(plan), encoding="utf-8")
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)

    def test_training_signature_rejects_mislabeled_value_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training").mkdir()
            # 故意制造参数标签为 Huber、实际优化配置为 MSE 的临时协议错配。
            config = dict(
                arguments=dict(seed=0, aggregation="mean", updates=1, rollout_steps=4,
                               update_epochs=1, value_loss_type="huber"),
                training=asdict(MAPPOConfig(rollout_steps=4, update_epochs=1)),
                model=SharedGraphActorCritic(6, 5, 2, hidden_dim=8).architecture_config(),
                environment=asdict(Continuous2DConfig()), communication=asdict(CommunicationConfig()),
            )
            (root / "training/run_config.json").write_text(json.dumps(config), encoding="utf-8")
            metrics = dict(update=1, environment_steps=4, mean_team_reward=0, actor_loss=0,
                           value_loss=0, entropy=0, total_loss=0, episodes_finished=0, successful_episodes=0)
            (root / "training/training_metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "价值损失"):
                _training_signature(root, dict(seed=0, aggregation="mean", train=["python", "train_mappo.py"]))


if __name__ == "__main__":
    unittest.main()
