"""图编码器、共享策略和 MAPPO 最小更新测试。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents import RuleBasedSearchAgent, SharedPolicyAgent
from algorithms import GraphObservationEncoder, SharedGraphActorCritic, batch_observations
from core import AgentAttributes
from core.observation import NeighborObservation, Observation
from environments import Continuous2DConfig, Continuous2DSearchEnv
from training import MAPPOConfig, MAPPOTrainer


class GraphLearningTests(unittest.TestCase):
    @staticmethod
    def _observation(neighbors: tuple[NeighborObservation, ...]) -> Observation:
        return Observation(
            agent_id="A",
            self_features=np.arange(6, dtype=float) / 10,
            neighbors=neighbors,
            task_features=np.array([0.2, 0.8]),
            neighbor_feature_dim=5,
        )

    def test_graph_encoder_is_neighbor_order_invariant(self) -> None:
        first = NeighborObservation("B", np.array([1, 0, 0, 0, 1]), 0.8)
        second = NeighborObservation("C", np.array([0, 1, 0, 0, 1]), 0.4)
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=16)
        batch = batch_observations(
            [self._observation((first, second)), self._observation((second, first))]
        )
        output = encoder(batch)
        self.assertTrue(torch.allclose(output[0], output[1], atol=1e-6))

    def test_graph_encoder_handles_isolated_agent(self) -> None:
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=8)
        output = encoder(batch_observations([self._observation(())]))
        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_shared_policy_action_shape_and_bounds(self) -> None:
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=16)
        actions, log_probs, value = model.act([self._observation(())])
        self.assertEqual(tuple(actions.shape), (1, 2))
        self.assertEqual(tuple(log_probs.shape), (1,))
        self.assertEqual(value.ndim, 0)
        self.assertLessEqual(float(actions.detach().abs().max()), 1.0)

    def test_learning_agent_uses_shared_model(self) -> None:
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8)
        agent = SharedPolicyAgent("A", model)
        action = agent.step(self._observation(()), 0.2)
        self.assertEqual(action.value.shape, (2,))

    def test_attention_is_order_invariant_and_has_trainable_gradients(self) -> None:
        torch.manual_seed(13)
        first = NeighborObservation("B", np.array([1, 0, 0, 0, 1]), 0.8, {"message_age": 0.2})
        second = NeighborObservation("C", np.array([0, 1, 0, 0, 0.4]), 0.4, {"message_age": 0.6})
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=8, aggregation="attention")
        output = encoder(batch_observations([
            self._observation((first, second)), self._observation((second, first)),
        ]))
        self.assertTrue(torch.allclose(output[0], output[1], atol=1e-6))
        output.square().sum().backward()
        gradients = [parameter.grad for parameter in encoder.attention_scorer.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_attention_uses_message_age_and_masks_padding(self) -> None:
        neighbors = (
            NeighborObservation("B", np.ones(5), metadata={"message_age": 0}),
            NeighborObservation("C", np.ones(5), metadata={"message_age": 1}),
        )
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=8, aggregation="attention")
        with torch.no_grad():
            for parameter in encoder.attention_scorer.parameters():
                parameter.zero_()
            encoder.attention_scorer[0].weight[0, -1] = 1
            encoder.attention_scorer[2].weight[0, 0] = 1
        batch = batch_observations([self._observation(neighbors), self._observation(())])
        weights = encoder.aggregation_weights(
            batch, encoder.self_encoder(batch.self_features), encoder.message_encoder(batch.neighbor_features)
        )
        self.assertGreater(float(weights[0, 1].detach()), float(weights[0, 0].detach()))
        self.assertAlmostEqual(float(weights[0].sum().detach()), 1)
        self.assertEqual(float(weights[1].sum().detach()), 0)

    def test_attention_isolated_agent_has_finite_backward_and_padding_is_ignored(self) -> None:
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=8, aggregation="attention")
        batch = batch_observations([self._observation(())])
        output = encoder(batch)
        polluted = replace(
            batch, neighbor_features=torch.full_like(batch.neighbor_features, 999),
            neighbor_weights=torch.full_like(batch.neighbor_weights, 999),
            neighbor_ages=torch.full_like(batch.neighbor_ages, 999),
        )
        self.assertTrue(torch.equal(output, encoder(polluted)))
        output.sum().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in encoder.parameters()
        ))

    def test_no_neighbor_ablation_ignores_neighbor_values(self) -> None:
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=8, aggregation="none")
        batch = batch_observations([
            self._observation(()),
            self._observation((NeighborObservation("B", np.ones(5)),)),
        ])
        output = encoder(batch)
        # float32 批量线性代数可能因行布局产生舍入差异，不要求逐位相等。
        torch.testing.assert_close(output[0], output[1], rtol=1e-6, atol=1e-7)

    def test_checkpoint_loader_supports_attention_and_legacy_mean(self) -> None:
        from scripts.evaluate_policies import _load_model

        with tempfile.TemporaryDirectory() as directory:
            for aggregation in ("mean", "attention", "none"):
                model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8, aggregation=aggregation)
                config = model.architecture_config()
                if aggregation == "mean":
                    del config["aggregation"]  # 旧检查点没有这个字段。
                checkpoint = Path(directory) / f"{aggregation}.pt"
                torch.save({"model": model.state_dict(), "model_config": config}, checkpoint)
                restored = _load_model(checkpoint, hidden_dim=64)
                self.assertEqual(restored.encoder.aggregation, aggregation)
                before = model.act([self._observation(())], deterministic=True)[0]
                after = restored.act([self._observation(())], deterministic=True)[0]
                self.assertTrue(torch.equal(before, after))

    def test_training_seed_reaches_environment_and_fault_does_not_scale_team_reward(self) -> None:
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8, aggregation="attention")
        environment = Continuous2DSearchEnv(Continuous2DConfig(
            width=10, height=10, dt=1, max_steps=2, energy_cost=0,
            target_positions=((0, 0),), collision_penalty=0, connectivity_penalty=0,
        ))
        attributes = AgentAttributes(max_speed=1, sensor_range=0.001, communication_range=20)
        for name in ("A", "B"):
            environment.add_agent(RuleBasedSearchAgent(name, attributes))
        environment.schedule_node_fault("B", start_time=1, duration=1)
        trainer = MAPPOTrainer(model, MAPPOConfig(rollout_steps=4, update_epochs=1), seed=77)
        with patch.object(environment, "reset", wraps=environment.reset) as reset:
            rollout, _, _ = trainer.collect_rollout(environment)
        self.assertEqual(reset.call_args_list[0].kwargs["seed"], 77)
        self.assertEqual(rollout[0].agent_ids, ("A", "B"))
        self.assertEqual(rollout[1].agent_ids, ("A",))
        self.assertAlmostEqual(rollout[0].reward, -0.01)
        self.assertTrue(np.isfinite(trainer.update(rollout)["total_loss"]))

    def test_invalid_aggregation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraphObservationEncoder(6, 5, 2, aggregation="unknown")

    def test_tiny_mappo_update_and_checkpoint(self) -> None:
        torch.manual_seed(3)
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=16)
        environment = Continuous2DSearchEnv(
            Continuous2DConfig(
                width=10,
                height=10,
                dt=1,
                max_steps=3,
                target_positions=((0.0, 0.0),),
                energy_cost=0,
            )
        )
        attributes = AgentAttributes(
            max_speed=1, sensor_range=0.01, communication_range=20
        )
        environment.add_agent(RuleBasedSearchAgent("A", attributes))
        environment.add_agent(RuleBasedSearchAgent("B", attributes))
        # 训练轨迹中在线智能体数从 2 变 1 再恢复，验证算法不依赖固定规模。
        environment.schedule_node_fault("B", start_time=1.0, duration=1.0)
        trainer = MAPPOTrainer(
            model,
            MAPPOConfig(rollout_steps=4, update_epochs=1),
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = trainer.train(environment, updates=1)[0]
        changed = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        self.assertTrue(np.isfinite(metrics.total_loss))
        self.assertEqual(metrics.environment_steps, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save_checkpoint(Path(directory) / "model.pt")
            self.assertTrue(checkpoint.exists())
            saved = torch.load(checkpoint, weights_only=False)
            self.assertIn("model", saved)
            restored = MAPPOTrainer(
                SharedGraphActorCritic(6, 5, 2, hidden_dim=16),
                MAPPOConfig(rollout_steps=4, update_epochs=1),
            )
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.environment_steps, 4)


if __name__ == "__main__":
    unittest.main()
