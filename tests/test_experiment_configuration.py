"""验证训练与评估使用真实的通信配置和可达的故障时间。"""

import argparse
import unittest

from algorithms import SharedGraphActorCritic
from core import Action
from interaction import CommunicationConfig
from scripts.evaluate_policies import _scenarios
from scripts.train_mappo import build_environment


class ExperimentConfigurationTests(unittest.TestCase):
    def test_short_training_run_experiences_node_failure_and_recovery(self):
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8, aggregation="attention")
        for max_steps in (4, 20):
            with self.subTest(max_steps=max_steps):
                env = build_environment(model, 3, max_steps)
                observations = env.reset(seed=7)
                counts = [len(observations)]
                for _ in range(max_steps):
                    result = env.step({name: Action.noop(name) for name in observations})
                    observations = result.observations
                    counts.append(len(observations))
                    if result.info["success"] or result.info["truncated"]:
                        break
                self.assertIn(2, counts)
                self.assertEqual(counts[-1], 3)

    def test_training_communication_parameters_reach_policy_observations(self):
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8)
        env = build_environment(model, 3, 20, communication_config=CommunicationConfig(packet_loss_rate=1))
        observations = env.reset(seed=7)
        self.assertTrue(all(not obs.neighbors for obs in observations.values()))

    def test_evaluation_keeps_nominal_topology_and_communication_faults_separate(self):
        args = argparse.Namespace(
            agent_counts="5,20", episodes=2, seed=1000, max_steps=30,
            packet_loss_rate=0.3, base_delay=0.2, delay_jitter=0.1, noise_std=0.01,
            observation_mode="communicated", neighbor_state_ttl=0.75,
        )
        scenarios = _scenarios(args)
        self.assertEqual(len(scenarios), 6)
        for scenario in scenarios:
            self.assertEqual(scenario.base_seed, 1000)
            self.assertEqual(scenario.neighbor_state_ttl, 0.75)
            if scenario.name == "communication_fault":
                self.assertEqual(scenario.communication_config.packet_loss_rate, 0.3)
                self.assertEqual(scenario.link_faults, ())
                self.assertEqual(scenario.node_faults, ())
            else:
                self.assertEqual(scenario.communication_config, CommunicationConfig())


if __name__ == "__main__":
    unittest.main()
