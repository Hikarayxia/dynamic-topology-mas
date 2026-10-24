"""验证策略观测受实际通信结果约束，而非邻居的实时真值。"""

import unittest

import numpy as np

from agents import RuleBasedSearchAgent
from core import Action, AgentAttributes, Message
from environments import Continuous2DConfig, Continuous2DSearchEnv
from interaction import CommunicationConfig


class CommunicationObservationTests(unittest.TestCase):
    def _environment(self, *, ttl=1.0, mode="communicated", **communication):
        env = Continuous2DSearchEnv(
            Continuous2DConfig(
                width=10, height=10, dt=0.25, max_steps=40,
                target_positions=((9.9, 9.9),), energy_cost=0,
                neighbor_state_ttl=ttl, neighbor_observation_mode=mode,
            ),
            CommunicationConfig(**communication),
        )
        attributes = AgentAttributes(max_speed=1, sensor_range=0.001, communication_range=20)
        for name in ("A", "B"):
            env.add_agent(RuleBasedSearchAgent(name, attributes))
        return env

    @staticmethod
    def _step(env, move_b=False):
        actions = {
            name: Action.noop(name) for name, agent in env.agents.items() if agent.state.active
        }
        if move_b and "B" in actions:
            actions["B"] = Action.continuous("B", np.array([1.0, 0.0]))
        return env.step(actions)

    def test_total_packet_loss_removes_neighbors_but_not_physical_edges(self):
        env = self._environment(packet_loss_rate=1.0)
        observations = env.reset(seed=7)
        self.assertTrue(env.topology.are_directly_connected("A", "B"))
        self.assertEqual(observations["A"].neighbors, ())
        result = self._step(env)
        self.assertEqual(result.observations["A"].neighbors, ())
        self.assertEqual(env.communication.stats.delivered, 0)
        self.assertGreater(env.communication.stats.dropped, 0)

    def test_delay_exposes_sent_snapshot_instead_of_current_neighbor_state(self):
        env = self._environment(base_delay=0.5)
        observations = env.reset(seed=7)
        original_b = env.get_agent("B").state.position.copy()
        original_a = env.get_agent("A").state.position.copy()
        self.assertEqual(observations["A"].neighbors, ())
        self.assertEqual(self._step(env, move_b=True).observations["A"].neighbors, ())
        result = self._step(env, move_b=True)
        neighbor = result.observations["A"].neighbors[0]
        np.testing.assert_allclose(neighbor.features[:2], (original_b - original_a) / 10)
        self.assertAlmostEqual(neighbor.metadata["message_age"], 0.5)
        self.assertFalse(np.allclose(original_b, env.get_agent("B").state.position))

    def test_zero_delay_matches_current_state_and_does_not_fill_protocol_inbox(self):
        env = self._environment()
        env.reset(seed=7)
        result = self._step(env, move_b=True)
        neighbor = result.observations["A"].neighbors[0]
        expected = env.get_agent("B").state.position - env.get_agent("A").state.position
        np.testing.assert_allclose(neighbor.features[:2], expected / 10)
        self.assertEqual(neighbor.metadata["message_age"], 0)
        self.assertEqual(env.get_agent("A").inbox_size, 0)
        env.route_message(Message("B", "A", {"kind": "replacement_gossip"}, env.current_time))
        self._step(env)
        self.assertEqual(env.get_agent("A").pop_messages()[0].payload["kind"], "replacement_gossip")

    def test_live_state_mutation_cannot_change_cached_snapshot(self):
        env = self._environment()
        before = env.reset(seed=7)["A"].neighbors[0].features.copy()
        env.get_agent("B").state.position[:] = (0, 0)
        after = env._build_observations()["A"].neighbors[0].features
        np.testing.assert_array_equal(before, after)

    def test_cached_state_expires_after_disconnect_without_oracle_invalidation(self):
        env = self._environment(ttl=0.5)
        env.reset(seed=7)
        env.schedule_link_fault("A", "B", start_time=0, duration=2)
        first = self._step(env)
        self.assertFalse(env.topology.are_directly_connected("A", "B"))
        self.assertEqual(len(first.observations["A"].neighbors), 1)
        self._step(env)
        self.assertEqual(self._step(env).observations["A"].neighbors, ())

    def test_removed_neighbor_keeps_received_snapshot_until_ttl_expires(self):
        """物理移除不能代替失效通知，接收方只能依靠消息年龄淘汰快照。"""

        env = self._environment(ttl=0.5)
        before = env.reset(seed=7)["A"].neighbors[0].features.copy()
        removed = env.remove_agent("B")
        self.assertNotIn("B", env.agents)
        self.assertFalse(env.topology.has_node("B"))
        # 被移除对象后续的变化不能修改接收方已经收到的状态。
        removed.state.position[:] = (0, 0)
        immediate = env._build_observations()["A"].neighbors
        self.assertEqual(tuple(neighbor.neighbor_id for neighbor in immediate), ("B",))
        np.testing.assert_array_equal(immediate[0].features, before)
        for expected_age in (0.25, 0.5):
            neighbor = self._step(env).observations["A"].neighbors[0]
            np.testing.assert_array_equal(neighbor.features, before)
            self.assertAlmostEqual(neighbor.metadata["message_age"], expected_age)
        self.assertEqual(self._step(env).observations["A"].neighbors, ())

    def test_removed_sender_cannot_refresh_cache_with_in_flight_state(self):
        """只保留已投递快照，移除前在途的状态包仍须通过链路校验。"""

        env = self._environment(ttl=0.5, base_delay=0.25)
        env.reset(seed=7)
        self.assertEqual(len(self._step(env).observations["A"].neighbors), 1)
        env.remove_agent("B")
        neighbors = self._step(env).observations["A"].neighbors
        self.assertEqual(len(neighbors), 1)
        neighbor = neighbors[0]
        self.assertEqual(neighbor.neighbor_id, "B")
        self.assertAlmostEqual(neighbor.metadata["message_age"], 0.5)
        self.assertEqual(self._step(env).observations["A"].neighbors, ())

    def test_oracle_observation_removes_deleted_neighbor_immediately(self):
        """真值对照仍应反映实时拓扑，不能被通信缓存修复改变语义。"""

        env = self._environment(mode="oracle")
        self.assertEqual(len(env.reset(seed=7)["A"].neighbors), 1)
        env.remove_agent("B")
        self.assertEqual(self._step(env).observations["A"].neighbors, ())

    def test_message_expiring_in_transit_never_enters_observation(self):
        env = self._environment(ttl=0.25, base_delay=0.5)
        env.reset(seed=7)
        for _ in range(4):
            self.assertEqual(self._step(env).observations["A"].neighbors, ())

    def test_noise_changes_observation_and_reset_reproduces_it(self):
        clean = self._environment().reset(seed=7)["A"].neighbors[0].features
        env = self._environment(noise_std=0.2)
        first = env.reset(seed=7)["A"].neighbors[0].features
        self.assertFalse(np.allclose(first, clean))
        np.testing.assert_array_equal(first, env.reset(seed=7)["A"].neighbors[0].features)

    def test_out_of_order_packets_never_replace_newer_state(self):
        env = self._environment(base_delay=0.5, delay_jitter=0.5, ttl=2)
        env.reset(seed=7)
        newest = -1.0
        for _ in range(15):
            observation = self._step(env).observations["A"]
            if observation.neighbors:
                sent_at = observation.timestamp - observation.neighbors[0].metadata["message_age"]
                self.assertGreaterEqual(sent_at, newest)
                newest = sent_at
        self.assertGreater(newest, 0)

    def test_reset_discards_old_cached_and_queued_messages(self):
        env = self._environment(base_delay=0.25)
        env.reset(seed=7)
        self.assertEqual(len(self._step(env).observations["A"].neighbors), 1)
        self.assertEqual(env.reset(seed=8)["A"].neighbors, ())

    def test_oracle_ablation_is_explicit_and_ignores_packet_loss(self):
        env = self._environment(mode="oracle", packet_loss_rate=1.0)
        observations = env.reset(seed=7)
        self.assertEqual(len(observations["A"].neighbors), 1)
        self.assertEqual(env.communication.stats.attempted, 0)

    def test_invalid_observation_configuration_is_rejected(self):
        for ttl in (0, -1, float("inf"), float("nan")):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                Continuous2DConfig(neighbor_state_ttl=ttl)
        with self.assertRaises(ValueError):
            Continuous2DConfig(neighbor_observation_mode="unknown")


if __name__ == "__main__":
    unittest.main()
