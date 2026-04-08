"""Tests for AccordisEnvironment reset/step with SimulatedConsensusAdapter."""

import pytest
from accordis.models import (
    AccordisAction,
    AccordisObservation,
    BFTConfig,
    LeaderRotation,
    MultiNodeAction,
    MultiNodeObservation,
    STATIC_BASELINE_CONFIG,
)
from accordis.server.adapters.simulated.adapter import SimulatedConsensusAdapter
from accordis.server.accordis_environment import AccordisEnvironment


@pytest.fixture
def env():
    adapter = SimulatedConsensusAdapter(seed=42)
    return AccordisEnvironment(adapter=adapter)


class TestReset:
    def test_reset_returns_multi_node_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, MultiNodeObservation)
        assert len(obs.nodes) > 0

    def test_reset_obs_are_accordis_observations(self, env):
        obs = env.reset()
        for nid, o in obs.nodes.items():
            assert isinstance(o, AccordisObservation)

    def test_reset_state_is_initialised(self, env):
        env.reset()
        state = env.state
        assert state.step == 0
        assert state.episode_id != ""

    def test_reset_with_custom_params(self, env):
        obs = env.reset(
            n_nodes=7,
            f_byzantine=2,
            curriculum_level=3,
            max_steps=300,
        )
        assert len(obs.nodes) == 5  # 7 - 2 honest nodes

    def test_reset_idempotent(self, env):
        env.reset()
        env.reset()  # should not raise
        state = env.state
        assert state.step == 0

    def test_reset_obs_have_step_zero(self, env):
        obs = env.reset()
        for nid, o in obs.nodes.items():
            assert o.step == 0

    def test_reset_no_byzantine(self, env):
        obs = env.reset(n_nodes=4, f_byzantine=0)
        assert len(obs.nodes) == 4


class TestStep:
    def _make_actions(self, env, obs):
        """Create default actions for all honest nodes."""
        return MultiNodeAction(nodes={
            nid: AccordisAction(
                node_id=nid,
                view_timeout_ms=2000,
                pipeline_depth=2,
                replication_batch_size=64,
                equivocation_threshold=5,
                vote_aggregation_timeout_ms=500,
            )
            for nid in obs.nodes.keys()
        })

    def test_step_returns_multi_node_observation(self, env):
        obs = env.reset()
        actions = self._make_actions(env, obs)
        result = env.step(actions)
        assert isinstance(result, MultiNodeObservation)

    def test_step_obs_are_observations(self, env):
        obs = env.reset()
        actions = self._make_actions(env, obs)
        result = env.step(actions)
        for nid, o in result.nodes.items():
            assert isinstance(o, AccordisObservation)

    def test_step_reward_is_float(self, env):
        obs = env.reset()
        actions = self._make_actions(env, obs)
        result = env.step(actions)
        assert isinstance(result.reward, float)

    def test_step_increments_state(self, env):
        obs = env.reset()
        actions = self._make_actions(env, obs)
        env.step(actions)
        assert env.state.step == 1

    def test_step_info_has_required_keys(self, env):
        obs = env.reset()
        actions = self._make_actions(env, obs)
        result = env.step(actions)
        assert "step" in result.metadata
        assert "liveness_rate" in result.metadata
        assert "agreement_ok" in result.metadata
        assert "validity_ok" in result.metadata

    def test_step_done_at_max_steps(self, env):
        obs = env.reset(max_steps=3)
        actions = self._make_actions(env, obs)
        result = None
        for _ in range(3):
            result = env.step(actions)
            actions = self._make_actions(env, result)
        assert result.done is True

    def test_step_without_reset_raises(self):
        adapter = SimulatedConsensusAdapter(seed=42)
        env = AccordisEnvironment(adapter=adapter)
        action = MultiNodeAction(nodes={"node_0": AccordisAction(
            node_id="node_0",
            view_timeout_ms=2000,
            pipeline_depth=2,
            replication_batch_size=64,
            equivocation_threshold=5,
            vote_aggregation_timeout_ms=500,
        )})
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_multiple_steps(self, env):
        obs = env.reset(max_steps=10)
        actions = self._make_actions(env, obs)
        result = None
        for i in range(10):
            result = env.step(actions)
            actions = self._make_actions(env, result)
            if result.done:
                break
        assert result.done is True

    def test_step_snapshots_reuse_large_episode_state(self, env):
        obs = env.reset(pool_size=100)
        actions = self._make_actions(env, obs)

        env.step(actions)

        assert env._episode_log is not None
        snapshot = env._episode_log.steps[0]
        assert snapshot.episode_txn_pool is env.state.episode_txn_pool
        assert snapshot.proposal_registry is env.state.proposal_registry
        assert snapshot.node_states is not env.state.node_states


class TestClampAction:
    def test_clamp_view_timeout_below_min(self, env):
        env.reset()
        action = AccordisAction(
            node_id="node_0",
            view_timeout_ms=50,  # below min 200
            pipeline_depth=2,
            replication_batch_size=64,
            equivocation_threshold=5,
            vote_aggregation_timeout_ms=500,
        )
        clamped = env._clamp_action(action)
        assert clamped.view_timeout_ms == 200

    def test_clamp_view_timeout_above_max(self, env):
        env.reset()
        action = AccordisAction(
            node_id="node_0",
            view_timeout_ms=99999,  # above max 10000
            pipeline_depth=2,
            replication_batch_size=64,
            equivocation_threshold=5,
            vote_aggregation_timeout_ms=500,
        )
        clamped = env._clamp_action(action)
        assert clamped.view_timeout_ms == 10000

    def test_clamp_vat_must_be_less_than_half_vt(self, env):
        env.reset()
        action = AccordisAction(
            node_id="node_0",
            view_timeout_ms=1000,
            pipeline_depth=2,
            replication_batch_size=64,
            equivocation_threshold=5,
            vote_aggregation_timeout_ms=600,  # exceeds vt//2 = 500
        )
        clamped = env._clamp_action(action)
        assert clamped.vote_aggregation_timeout_ms < clamped.view_timeout_ms // 2

    def test_all_fields_in_bounds(self, env):
        from accordis.models import SAFE_BFT_TUNING_BOUNDS
        env.reset()
        action = AccordisAction(
            node_id="node_0",
            view_timeout_ms=9999999,
            pipeline_depth=9999,
            replication_batch_size=9999,
            equivocation_threshold=9999,
            vote_aggregation_timeout_ms=9999,
        )
        clamped = env._clamp_action(action)
        for field, (lo, hi) in SAFE_BFT_TUNING_BOUNDS.items():
            val = getattr(clamped, field)
            assert lo <= val <= hi, f"{field}={val} not in [{lo}, {hi}]"
