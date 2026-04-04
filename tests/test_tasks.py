"""Tests for task grading (adapter-agnostic)."""

import pytest
from accordis.models import (
    AccordisReward,
    AccordisState,
    BFAStrategy,
    Block,
    EpisodeLog,
    NodeState,
    ProposalRegistry,
    Transaction,
)
from accordis.server.tasks.task_easy import EasyTask
from accordis.server.tasks.task_medium import MediumTask
from accordis.server.tasks.task_hard import HardTask


def make_episode_log(liveness_rate=0.9, view_change_count=2, agreement_ok=True, n_commits=9):
    """Build a minimal EpisodeLog for task grading tests."""
    txns = [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(10)]
    committed_txns = txns[:n_commits]
    block = Block(slot=1, hash="abc", proposer_id="node_0", transactions=committed_txns)
    registry = ProposalRegistry(honest_proposals={tx.id: tx for tx in txns})

    state = AccordisState(
        episode_id="test",
        n_nodes=4,
        f_byzantine=0,
        node_states={
            "node_0": NodeState(
                node_id="node_0",
                is_byzantine=False,
                committed_log=[block],
            ),
            "node_1": NodeState(
                node_id="node_1",
                is_byzantine=False,
                committed_log=[block],
            ),
        },
        view_change_count=view_change_count,
        bfa_strategy=BFAStrategy.NONE,
        proposal_registry=registry,
        episode_txn_pool=txns,
    )

    rewards = [AccordisReward(total=50.0)]
    log = EpisodeLog(
        episode_id="test",
        curriculum_level=1,
        bfa_strategy=BFAStrategy.NONE,
        bfa_strategy_seed=42,
        steps=[state],
        rewards=rewards,
        final_score=0.0,
    )
    return log


class TestEasyTask:
    def test_grade_high_performance(self):
        task = EasyTask(curriculum_level=1)
        log = make_episode_log(n_commits=10, view_change_count=0)
        score = task.grade(log)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # good liveness + no view changes

    def test_grade_poor_performance(self):
        task = EasyTask(curriculum_level=1)
        log = make_episode_log(n_commits=0, view_change_count=10)
        score = task.grade(log)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_grade_empty_log(self):
        task = EasyTask()
        log = EpisodeLog(
            episode_id="x", curriculum_level=1,
            bfa_strategy=BFAStrategy.NONE, bfa_strategy_seed=42,
        )
        score = task.grade(log)
        assert score == 0.0

    def test_get_initial_conditions_level1(self):
        task = EasyTask(curriculum_level=1)
        cond = task.get_initial_conditions()
        assert cond["n_nodes"] == 4
        assert cond["f_byzantine"] == 0
        assert cond["max_steps"] == 200

    def test_get_initial_conditions_level2(self):
        task = EasyTask(curriculum_level=2)
        cond = task.get_initial_conditions()
        assert cond["f_byzantine"] == 1


class TestMediumTask:
    def test_grade_range(self):
        task = MediumTask(curriculum_level=3)
        log = make_episode_log(n_commits=8, view_change_count=3)
        score = task.grade(log)
        assert 0.0 <= score <= 1.0

    def test_get_initial_conditions(self):
        task = MediumTask(curriculum_level=4)
        cond = task.get_initial_conditions()
        assert cond["n_nodes"] == 7
        assert cond["f_byzantine"] == 2
        assert cond["max_steps"] == 300

    def test_recovery_bonus_when_no_recovery(self):
        task = MediumTask(curriculum_level=3)
        log = make_episode_log(n_commits=8)
        # No fast_leader_recovery in rewards
        score_no_recovery = task.grade(log)

        # Add a reward with fast_leader_recovery
        log.rewards.append(AccordisReward(fast_leader_recovery=120.0, total=120.0))
        score_with_recovery = task.grade(log)

        assert score_with_recovery >= score_no_recovery

    def test_empty_log_returns_zero(self):
        task = MediumTask()
        log = EpisodeLog(
            episode_id="x", curriculum_level=3,
            bfa_strategy=BFAStrategy.SELECTIVE_DELAY, bfa_strategy_seed=42,
        )
        assert task.grade(log) == 0.0


class TestHardTask:
    def test_grade_range(self):
        task = HardTask(curriculum_level=6)
        log = make_episode_log(n_commits=7, view_change_count=5)
        score = task.grade(log)
        assert 0.0 <= score <= 1.0

    def test_get_initial_conditions_level6(self):
        task = HardTask(curriculum_level=6)
        cond = task.get_initial_conditions()
        assert cond["n_nodes"] == 10
        assert cond["f_byzantine"] == 3
        assert cond["max_steps"] == 500
        from accordis.models import LeaderRotation
        assert cond["leader_rotation"] == LeaderRotation.ROUND_ROBIN

    def test_get_initial_conditions_level7_vrf(self):
        from accordis.models import LeaderRotation
        task = HardTask(curriculum_level=7)
        cond = task.get_initial_conditions()
        assert cond["leader_rotation"] == LeaderRotation.VRF

    def test_get_initial_conditions_level8_reputation(self):
        from accordis.models import LeaderRotation
        task = HardTask(curriculum_level=8)
        cond = task.get_initial_conditions()
        assert cond["leader_rotation"] == LeaderRotation.REPUTATION_WEIGHTED

    def test_empty_log_returns_zero(self):
        task = HardTask()
        log = EpisodeLog(
            episode_id="x", curriculum_level=6,
            bfa_strategy=BFAStrategy.CASCADE_TIMING, bfa_strategy_seed=42,
        )
        assert task.grade(log) == 0.0
