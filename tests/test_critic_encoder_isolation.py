"""价值梯度隔离的数学行为、真实更新及历史配置兼容性。"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from algorithms import SharedGraphActorCritic
from core import Observation, NeighborObservation
from scripts.train_mappo import build_environment
from training import MAPPOConfig, MAPPOTrainer


class CriticEncoderIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8, aggregation="attention")
        self.observations = [Observation(
            "A", np.arange(6) / 10,
            neighbors=(NeighborObservation("B", np.ones(5)),),
            task_features=np.array([0.1, 0.9]), neighbor_feature_dim=5,
        )]
        with torch.no_grad():
            self.actions = self.model.act(self.observations, deterministic=True)[0]

    def test_detaching_preserves_forward_numbers_and_blocks_only_value_encoder_gradient(self):
        shared = self.model.evaluate_actions(self.observations, self.actions)
        detached = self.model.evaluate_actions(
            self.observations, self.actions, detach_critic_encoder=True)
        for first, second in zip(shared, detached):
            self.assertTrue(torch.equal(first, second))
        detached[2].backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in self.model.critic.parameters()))
        self.model.zero_grad(set_to_none=True)
        self.model.evaluate_actions(self.observations, self.actions)[2].backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in self.model.encoder.parameters()))

    def test_actor_gradient_still_trains_encoder_in_detached_mode(self):
        log_probs, _, _ = self.model.evaluate_actions(
            self.observations, self.actions + 0.1, detach_critic_encoder=True)
        (-log_probs.mean()).backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in self.model.encoder.parameters()))

    def test_critic_only_update_cannot_change_actor_in_detached_mode(self):
        before = self.actions.clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        _, _, value = self.model.evaluate_actions(
            self.observations, self.actions, detach_critic_encoder=True)
        (value - 10).square().backward()
        optimizer.step()
        after = self.model.act(self.observations, deterministic=True)[0]
        self.assertTrue(torch.equal(before, after))

    def test_real_ppo_update_uses_isolation_and_remains_finite(self):
        from unittest.mock import patch
        trainer = MAPPOTrainer(self.model, MAPPOConfig(
            rollout_steps=4, update_epochs=1, critic_encoder_mode="detached"))
        env = build_environment(self.model, 2, 4)
        with patch.object(self.model, "evaluate_actions", wraps=self.model.evaluate_actions) as evaluate:
            metrics = trainer.train(env, 1)[0]
        self.assertTrue(all(call.kwargs["detach_critic_encoder"] for call in evaluate.call_args_list))
        self.assertTrue(np.isfinite(metrics.total_loss))

    def test_checkpoint_mismatch_rejected_atomically_and_legacy_shared_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            shared = MAPPOTrainer(self.model)
            shared.save_checkpoint(path)
            detached = MAPPOTrainer(SharedGraphActorCritic(6, 5, 2, hidden_dim=8, aggregation="attention"),
                                    replace(MAPPOConfig(), critic_encoder_mode="detached"))
            before = {key: value.clone() for key, value in detached.model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "梯度模式"):
                detached.load_checkpoint(path)
            self.assertTrue(all(torch.equal(value, detached.model.state_dict()[key])
                                for key, value in before.items()))
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["config"].pop("critic_encoder_mode")
            torch.save(checkpoint, path)
            shared.load_checkpoint(path)
            detached.save_checkpoint(path)
            detached.load_checkpoint(path)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            replace(MAPPOConfig(), critic_encoder_mode="invalid")

    def test_legacy_and_explicit_shared_training_signatures_match(self):
        from tests.test_ablation_reporting import AblationReportingTests
        from evaluation.ablation import _training_signature
        fixture = AblationReportingTests()
        fixture.setUp()
        try:
            fixture._fixture(seeds=(0, 1))
            plan = json.loads((fixture.root / "ablation_plan.json").read_text())
            runs = [fixture.root / "mean" / f"seed_{seed}" for seed in (0, 1)]
            path = runs[0] / "training/run_config.json"
            config = json.loads(path.read_text())
            config["arguments"].pop("critic_encoder_mode", None)
            config["training"].pop("critic_encoder_mode", None)
            path.write_text(json.dumps(config))
            self.assertEqual(_training_signature(runs[0], plan[0]), _training_signature(runs[1], plan[1]))
        finally:
            fixture.doCleanups()

    def test_train_cli_and_matrix_plan_preserve_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for script, extra, destination in (
                ("train_mappo.py", ["--agents", "2"], root / "training"),
                ("run_ablation.py", ["--train-agents", "2", "--aggregations", "attention",
                                     "--seeds", "0", "--agent-counts", "2", "--dry-run"], root / "matrix"),
            ):
                result = subprocess.run([
                    sys.executable, "scripts/" + script, "--critic-encoder-mode", "detached",
                    "--updates", "1", "--rollout-steps", "4", "--update-epochs", "1",
                    "--max-steps", "4", "--hidden-dim", "8", "--output-dir", str(destination), *extra,
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((root / "training/run_config.json").read_text())
            self.assertEqual(config["training"]["critic_encoder_mode"], "detached")
            checkpoint = torch.load(root / "training/mappo_checkpoint.pt", weights_only=True)
            self.assertEqual(checkpoint["config"]["critic_encoder_mode"], "detached")
            train = json.loads((root / "matrix/ablation_plan.json").read_text())[0]["train"]
            self.assertEqual(train[train.index("--critic-encoder-mode") + 1], "detached")
