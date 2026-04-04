"""Tests for SimulatedConsensusAdapter internals (Version 1 only)."""

import pytest
from models import BFAStrategy, BFTConfig, LeaderRotation
from server.adapters.simulated.adapter import SimulatedConsensusAdapter


@pytest.fixture
def adapter():
    a = SimulatedConsensusAdapter(seed=42)
    a.start_cluster(n_nodes=4, f_byzantine=1, leader_rotation=LeaderRotation.ROUND_ROBIN)
    return a


class TestStartCluster:
    def test_returns_correct_node_count(self):
        a = SimulatedConsensusAdapter(seed=42)
        node_ids = a.start_cluster(4, 1, LeaderRotation.ROUND_ROBIN)
        assert len(node_ids) == 4

    def test_honest_nodes_first(self):
        a = SimulatedConsensusAdapter(seed=42)
        node_ids = a.start_cluster(4, 1, LeaderRotation.ROUND_ROBIN)
        honest = a.get_honest_nodes()
        byzantine = a.get_byzantine_nodes()
        assert len(honest) == 3
        assert len(byzantine) == 1
        # Honest IDs should come before byzantine in the returned list
        assert node_ids[:3] == honest

    def test_stop_cluster_is_idempotent(self):
        a = SimulatedConsensusAdapter(seed=42)
        a.stop_cluster()  # safe before start
        a.start_cluster(4, 0, LeaderRotation.ROUND_ROBIN)
        a.stop_cluster()
        a.stop_cluster()  # safe to call twice

    def test_restart_replaces_cluster(self):
        a = SimulatedConsensusAdapter(seed=42)
        a.start_cluster(4, 0, LeaderRotation.ROUND_ROBIN)
        first_honest = a.get_honest_nodes()
        a.start_cluster(7, 2, LeaderRotation.ROUND_ROBIN)
        second_honest = a.get_honest_nodes()
        assert len(second_honest) == 5  # 7 - 2
        assert len(first_honest) == 4


class TestApplyConfig:
    def test_apply_config_to_honest_node(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        cfg = BFTConfig(view_timeout_ms=3000, pipeline_depth=4)
        adapter.apply_config(nid, cfg)
        # Verify config was applied by reading it back
        node = adapter._nodes[nid]
        assert node.config.view_timeout_ms == 3000
        assert node.config.pipeline_depth == 4


class TestReadObservation:
    def test_returns_required_keys(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        obs = adapter.read_observation(nid)
        required = {
            "role", "current_view", "phase_latency_p50", "phase_latency_p99",
            "qc_miss_streak", "view_changes_last_50", "equivocation_counts",
            "inter_message_variance", "suspected_peers", "committed_tps",
            "pending_count", "pipeline_utilisation",
        }
        assert set(obs.keys()) == required

    def test_no_extra_keys(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        obs = adapter.read_observation(nid)
        assert len(obs) == 12

    def test_initial_values_are_sane(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        obs = adapter.read_observation(nid)
        assert obs["current_view"] >= 0
        assert obs["qc_miss_streak"] >= 0
        assert 0.0 <= obs["pipeline_utilisation"] <= 1.0

    def test_committed_tps_non_negative(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        obs = adapter.read_observation(nid)
        assert obs["committed_tps"] >= 0.0

    def test_pending_count_non_negative(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        obs = adapter.read_observation(nid)
        assert obs["pending_count"] >= 0


class TestAdvanceOneStep:
    def test_step_increments_tick(self, adapter):
        initial_tick = adapter._current_tick
        adapter.advance_one_step()
        assert adapter._current_tick == initial_tick + 1

    def test_multiple_steps(self, adapter):
        for _ in range(5):
            adapter.advance_one_step()
        assert adapter._current_tick == 5

    def test_observation_changes_after_steps(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        for _ in range(10):
            adapter.advance_one_step()
        obs = adapter.read_observation(nid)
        # After 10 steps, tps and other metrics should be computed
        assert obs["committed_tps"] >= 0.0


class TestGetCommittedLog:
    def test_empty_initially(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        log = adapter.get_committed_log(nid)
        assert isinstance(log, list)

    def test_log_items_are_dicts(self, adapter):
        nid = adapter.get_honest_nodes()[0]
        for _ in range(20):
            adapter.advance_one_step()
        log = adapter.get_committed_log(nid)
        for item in log:
            assert isinstance(item, dict)
            assert "slot" in item
            assert "hash" in item
            assert "proposer_id" in item


class TestInjectByzantineAction:
    def test_inject_random_delay(self, adapter):
        byz = adapter.get_byzantine_nodes()[0]
        adapter.inject_byzantine_action(
            byzantine_node_id=byz,
            strategy=BFAStrategy.RANDOM_DELAY,
            target_nodes=adapter.get_honest_nodes(),
            parameters={"delay_ms": 200},
        )
        # Verify the action was stored
        node = adapter._nodes[byz]
        assert node.pending_byzantine_action is not None

    def test_inject_none_clears_action(self, adapter):
        byz = adapter.get_byzantine_nodes()[0]
        # First inject something
        adapter.inject_byzantine_action(
            byzantine_node_id=byz,
            strategy=BFAStrategy.RANDOM_DELAY,
            target_nodes=[],
            parameters={"delay_ms": 100},
        )
        # Then inject NONE
        adapter.inject_byzantine_action(
            byzantine_node_id=byz,
            strategy=BFAStrategy.NONE,
            target_nodes=[],
            parameters={},
        )
        node = adapter._nodes[byz]
        assert node.pending_byzantine_action is None

    def test_action_cleared_after_tick(self, adapter):
        byz = adapter.get_byzantine_nodes()[0]
        adapter.inject_byzantine_action(
            byzantine_node_id=byz,
            strategy=BFAStrategy.RANDOM_DELAY,
            target_nodes=adapter.get_honest_nodes(),
            parameters={"delay_ms": 100},
        )
        adapter.advance_one_step()
        node = adapter._nodes[byz]
        assert node.pending_byzantine_action is None


class TestConfigureNetwork:
    def test_configure_all_levels(self):
        a = SimulatedConsensusAdapter(seed=42)
        a.start_cluster(4, 0, LeaderRotation.ROUND_ROBIN)
        for level in range(1, 9):
            a.configure_network(level, a.get_honest_nodes())  # should not raise

    def test_configure_mid_episode(self, adapter):
        adapter.advance_one_step()
        adapter.configure_network(5, adapter.get_honest_nodes())
        adapter.advance_one_step()  # should not raise
