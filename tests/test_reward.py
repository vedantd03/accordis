"""Tests for RewardCalculator (adapter-agnostic)."""

import pytest
from accordis.models import (
    AccordisReward,
    AccordisState,
    BaselineComparison,
    BFAStrategy,
    Block,
    NodeState,
    ProposalRegistry,
    Transaction,
    VerificationResult,
    VerifierResults,
)
from accordis.server.rewards.reward_calculator import (
    RewardCalculator,
    LIVENESS_COST_PER_STEP,
    UNNECESSARY_VIEW_CHANGE,
    LIVENESS_STALL,
    BLOCK_COMMIT,
    FAST_LEADER_RECOVERY,
    PIPELINE_EFFICIENCY,
    THROUGHPUT_IMPROVEMENT,
    LATENCY_IMPROVEMENT,
)
from accordis.server.oracle.verifier import CorrectnessOracle


def make_state(step=1, view_change_count=0, n_commits=0, txns=None):
    txns = txns or [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(10)]
    committed_txns = txns[:n_commits]
    blocks = []
    if committed_txns:
        blocks = [Block(slot=1, hash="abc", proposer_id="node_0", transactions=committed_txns)]
    registry = ProposalRegistry(honest_proposals={tx.id: tx for tx in txns})

    return AccordisState(
        episode_id="test",
        step=step,
        n_nodes=4,
        f_byzantine=0,
        node_states={
            "node_0": NodeState(
                node_id="node_0",
                is_byzantine=False,
                committed_log=blocks,
            ),
        },
        view_change_count=view_change_count,
        bfa_strategy=BFAStrategy.NONE,
        proposal_registry=registry,
        episode_txn_pool=txns,
    )


def make_verifier(agreement=True, validity=True):
    return VerifierResults(
        agreement=VerificationResult(passed=agreement, property="Agreement"),
        validity=VerificationResult(passed=validity, property="Validity"),
        agreement_violated=not agreement,
        validity_violated=not validity,
    )


class TestRewardConstants:
    def test_constants_have_correct_signs(self):
        assert LIVENESS_COST_PER_STEP < 0
        assert UNNECESSARY_VIEW_CHANGE < 0
        assert LIVENESS_STALL < 0
        assert BLOCK_COMMIT > 0
        assert FAST_LEADER_RECOVERY > 0
        assert PIPELINE_EFFICIENCY > 0
        assert THROUGHPUT_IMPROVEMENT > 0
        assert LATENCY_IMPROVEMENT > 0

    def test_constant_values(self):
        assert LIVENESS_COST_PER_STEP  == -1.0
        assert UNNECESSARY_VIEW_CHANGE == -150.0
        assert LIVENESS_STALL          == -300.0
        assert BLOCK_COMMIT            == +60.0
        assert FAST_LEADER_RECOVERY    == +120.0
        assert PIPELINE_EFFICIENCY     == +15.0
        assert THROUGHPUT_IMPROVEMENT  == +20.0
        assert LATENCY_IMPROVEMENT     == +10.0


class TestComputeReward:
    def test_total_is_sum_of_components(self):
        calc = RewardCalculator()
        oracle = CorrectnessOracle()
        prev = make_state(step=0, n_commits=0)
        curr = make_state(step=1, n_commits=5)
        verifier = make_verifier()
        liveness = oracle.check_liveness(curr)
        reward = calc.compute(prev, curr, verifier, liveness=liveness)
        expected = (
            reward.liveness_cost
            + reward.unnecessary_view_change
            + reward.liveness_stall
            + reward.block_commit
            + reward.fast_leader_recovery
            + reward.false_positive_avoided
            + reward.pipeline_efficiency
            + reward.throughput_improvement
            + reward.latency_improvement
        )
        assert abs(reward.total - expected) < 1e-9

    def test_liveness_cost_when_pending_txns(self):
        calc = RewardCalculator()
        oracle = CorrectnessOracle()
        prev = make_state(step=0)
        curr = make_state(step=1, n_commits=0)  # pending txns exist
        verifier = make_verifier()
        liveness = oracle.check_liveness(curr)
        reward = calc.compute(prev, curr, verifier, liveness=liveness)
        assert reward.liveness_cost == LIVENESS_COST_PER_STEP

    def test_block_commit_reward(self):
        calc = RewardCalculator()
        oracle = CorrectnessOracle()
        txns = [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(10)]
        prev = make_state(step=0, n_commits=0, txns=txns)
        curr = make_state(step=1, n_commits=5, txns=txns)
        verifier = make_verifier()
        liveness = oracle.check_liveness(curr)
        reward = calc.compute(prev, curr, verifier, liveness=liveness)
        assert reward.block_commit > 0

    def test_episode_end_rewards_only_when_done(self):
        calc = RewardCalculator()
        oracle = CorrectnessOracle()
        prev = make_state(step=0)
        curr = make_state(step=1)
        verifier = make_verifier()
        liveness = oracle.check_liveness(curr)
        baseline = BaselineComparison(relative_tps_improvement=0.5, relative_latency_improvement=0.3)

        # not done
        reward_mid = calc.compute(prev, curr, verifier, baseline=baseline, liveness=liveness, is_done=False)
        assert reward_mid.throughput_improvement == 0.0
        assert reward_mid.latency_improvement == 0.0

        # done
        reward_end = calc.compute(prev, curr, verifier, baseline=baseline, liveness=liveness, is_done=True)
        assert reward_end.throughput_improvement > 0 or reward_end.latency_improvement > 0

    def test_no_none_state_raises(self):
        calc = RewardCalculator()
        oracle = CorrectnessOracle()
        curr = make_state(step=1)
        verifier = make_verifier()
        liveness = oracle.check_liveness(curr)
        # prev=None should not raise
        reward = calc.compute(None, curr, verifier, liveness=liveness)
        assert isinstance(reward, AccordisReward)

    def test_reward_is_accordis_reward(self):
        calc = RewardCalculator()
        oracle = CorrectnessOracle()
        curr = make_state(step=1)
        verifier = make_verifier()
        liveness = oracle.check_liveness(curr)
        reward = calc.compute(None, curr, verifier, liveness=liveness)
        assert isinstance(reward, AccordisReward)
