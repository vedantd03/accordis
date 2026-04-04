"""Tests for CorrectnessOracle (adapter-agnostic)."""

import pytest
from accordis.models import (
    AccordisState,
    BFAStrategy,
    Block,
    EpisodeLog,
    LeaderRotation,
    NodeState,
    ProposalRegistry,
    Transaction,
    VerificationResult,
)
from accordis.server.oracle.verifier import CorrectnessOracle


def make_state(honest_logs=None, byzantine_ids=None, txn_pool=None, view_change_count=0):
    """Helper: build a minimal AccordisState for testing."""
    txn_pool = txn_pool or [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(10)]
    registry = ProposalRegistry(honest_proposals={tx.id: tx for tx in txn_pool})

    node_states = {}
    if honest_logs:
        for nid, log in honest_logs.items():
            node_states[nid] = NodeState(node_id=nid, is_byzantine=False, committed_log=log)
    for byz in (byzantine_ids or []):
        node_states[byz] = NodeState(node_id=byz, is_byzantine=True, committed_log=[])

    return AccordisState(
        episode_id="test",
        n_nodes=len(node_states) or 4,
        f_byzantine=len(byzantine_ids or []),
        node_states=node_states,
        view_change_count=view_change_count,
        bfa_strategy=BFAStrategy.NONE,
        proposal_registry=registry,
        episode_txn_pool=txn_pool,
    )


class TestVerifyAgreement:
    def test_agreement_empty_logs(self):
        oracle = CorrectnessOracle()
        state = make_state({"node_0": [], "node_1": []})
        result = oracle.verify_agreement(state)
        assert result.passed is True

    def test_agreement_same_hash(self):
        oracle = CorrectnessOracle()
        tx = Transaction(id="tx_0", submitted_at=0)
        block = Block(slot=1, hash="abc", proposer_id="node_0", transactions=[tx])
        state = make_state({
            "node_0": [block],
            "node_1": [block],
        })
        result = oracle.verify_agreement(state)
        assert result.passed is True

    def test_agreement_violated_different_hashes(self):
        oracle = CorrectnessOracle()
        tx = Transaction(id="tx_0", submitted_at=0)
        block_a = Block(slot=1, hash="hash_A", proposer_id="node_0", transactions=[tx])
        block_b = Block(slot=1, hash="hash_B", proposer_id="node_0", transactions=[tx])
        state = make_state({
            "node_0": [block_a],
            "node_1": [block_b],
        })
        result = oracle.verify_agreement(state)
        assert result.passed is False
        assert "hash_A" in result.evidence or "hash_B" in result.evidence

    def test_agreement_byzantine_nodes_ignored(self):
        oracle = CorrectnessOracle()
        tx = Transaction(id="tx_0", submitted_at=0)
        block_a = Block(slot=1, hash="hash_A", proposer_id="node_0", transactions=[tx])
        block_b = Block(slot=1, hash="hash_B", proposer_id="node_0", transactions=[tx])
        state = make_state(
            honest_logs={"node_0": [block_a], "node_1": [block_a]},
            byzantine_ids=["node_3"],
        )
        # Add different block to byzantine node's state manually
        state.node_states["node_3"].committed_log = [block_b]
        result = oracle.verify_agreement(state)
        assert result.passed is True


class TestVerifyValidity:
    def test_validity_all_in_registry(self):
        oracle = CorrectnessOracle()
        tx = Transaction(id="tx_0", submitted_at=0)
        block = Block(slot=1, hash="abc", proposer_id="node_0", transactions=[tx])
        state = make_state(
            honest_logs={"node_0": [block]},
            txn_pool=[tx],
        )
        result = oracle.verify_validity(state)
        assert result.passed is True

    def test_validity_unknown_transaction(self):
        oracle = CorrectnessOracle()
        unknown_tx = Transaction(id="tx_unknown_99", submitted_at=0)
        block = Block(slot=1, hash="abc", proposer_id="node_0", transactions=[unknown_tx])
        # txn_pool does NOT contain unknown_tx
        txn_pool = [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(5)]
        state = make_state(
            honest_logs={"node_0": [block]},
            txn_pool=txn_pool,
        )
        result = oracle.verify_validity(state)
        assert result.passed is False
        assert "tx_unknown_99" in result.evidence

    def test_validity_empty_logs(self):
        oracle = CorrectnessOracle()
        state = make_state({"node_0": []})
        result = oracle.verify_validity(state)
        assert result.passed is True


class TestCheckLiveness:
    def test_liveness_no_commits(self):
        oracle = CorrectnessOracle()
        state = make_state({"node_0": [], "node_1": []})
        result = oracle.check_liveness(state)
        assert result.liveness_rate == 0.0
        assert result.committed_count == 0
        assert result.pending_count == len(state.episode_txn_pool)

    def test_liveness_with_commits(self):
        oracle = CorrectnessOracle()
        txns = [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(10)]
        block = Block(slot=1, hash="abc", proposer_id="node_0", transactions=txns[:5])
        state = make_state(
            honest_logs={"node_0": [block], "node_1": [block]},
            txn_pool=txns,
        )
        result = oracle.check_liveness(state)
        assert result.committed_count == 5
        assert result.liveness_rate == 0.5
        assert result.pending_count == 5

    def test_liveness_full_commitment(self):
        oracle = CorrectnessOracle()
        txns = [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(5)]
        block = Block(slot=1, hash="abc", proposer_id="node_0", transactions=txns)
        state = make_state(
            honest_logs={"node_0": [block]},
            txn_pool=txns,
        )
        result = oracle.check_liveness(state)
        assert result.liveness_rate == 1.0


class TestRunAll:
    def test_run_all_both_pass(self):
        oracle = CorrectnessOracle()
        state = make_state({"node_0": [], "node_1": []})
        results = oracle.run_all(state)
        assert results.agreement.passed is True
        assert results.validity.passed is True
        assert results.agreement_violated is False
        assert results.validity_violated is False

    def test_run_all_flags_agreement_violation(self):
        oracle = CorrectnessOracle()
        tx = Transaction(id="tx_0", submitted_at=0)
        block_a = Block(slot=1, hash="A", proposer_id="node_0", transactions=[tx])
        block_b = Block(slot=1, hash="B", proposer_id="node_0", transactions=[tx])
        state = make_state(
            honest_logs={"node_0": [block_a], "node_1": [block_b]},
            txn_pool=[tx],
        )
        results = oracle.run_all(state)
        assert results.agreement_violated is True
