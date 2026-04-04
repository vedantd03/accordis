"""Tests for models.py — all Pydantic model validation."""

import pytest
from accordis.models import (
    AccordisAction,
    AccordisObservation,
    AccordisRubric,
    MultiNodeObservation,
    AccordisReward,
    AccordisState,
    AccordisTransform,
    BFAStrategy,
    BFTConfig,
    Block,
    EpisodeLog,
    LeaderRotation,
    NodeRole,
    NodeState,
    Phase,
    ProposalRegistry,
    SAFE_BFT_TUNING_BOUNDS,
    STATIC_BASELINE_CONFIG,
    Transaction,
    VerificationResult,
    VerifierResults,
)


class TestBFTConfig:
    def test_defaults(self):
        cfg = BFTConfig()
        assert cfg.view_timeout_ms == 2000
        assert cfg.pipeline_depth == 2
        assert cfg.replication_batch_size == 64
        assert cfg.equivocation_threshold == 5
        assert cfg.vote_aggregation_timeout_ms == 500

    def test_custom_values(self):
        cfg = BFTConfig(view_timeout_ms=3000, pipeline_depth=4)
        assert cfg.view_timeout_ms == 3000
        assert cfg.pipeline_depth == 4

    def test_static_baseline_is_bftconfig(self):
        assert isinstance(STATIC_BASELINE_CONFIG, BFTConfig)
        assert STATIC_BASELINE_CONFIG.view_timeout_ms == 2000

    def test_bounds_all_present(self):
        expected = {
            "view_timeout_ms", "pipeline_depth", "replication_batch_size",
            "equivocation_threshold", "vote_aggregation_timeout_ms",
        }
        assert set(SAFE_BFT_TUNING_BOUNDS.keys()) == expected


class TestAccordisAction:
    def test_valid_action(self):
        action = AccordisAction(
            node_id="node_0",
            view_timeout_ms=2000,
            pipeline_depth=2,
            replication_batch_size=64,
            equivocation_threshold=5,
            vote_aggregation_timeout_ms=500,
        )
        assert action.node_id == "node_0"
        assert action.suspect_node is None
        assert action.clear_suspicion is None

    def test_with_suspicion_fields(self):
        action = AccordisAction(
            node_id="node_0",
            view_timeout_ms=2000,
            pipeline_depth=2,
            replication_batch_size=64,
            equivocation_threshold=5,
            vote_aggregation_timeout_ms=500,
            suspect_node="node_3",
            clear_suspicion="node_2",
        )
        assert action.suspect_node == "node_3"
        assert action.clear_suspicion == "node_2"


class TestAccordisObservation:
    def test_minimal_observation(self):
        obs = AccordisObservation(
            node_id="node_0",
            current_role=NodeRole.REPLICA,
            current_view=0,
        )
        assert obs.step == 0
        assert obs.commit_throughput_tps == 0.0
        assert obs.per_phase_latency_p50 == {}

    def test_full_observation(self):
        obs = AccordisObservation(
            node_id="node_1",
            current_role=NodeRole.LEADER,
            current_view=5,
            per_phase_latency_p50={"prepare": 2.0},
            per_phase_latency_p99={"prepare": 5.0},
            qc_formation_miss_streak=2,
            view_change_count_recent=1,
            commit_throughput_tps=10.5,
            pending_txn_count=50,
            pipeline_utilisation=0.75,
            step=10,
        )
        assert obs.current_role == NodeRole.LEADER
        assert obs.per_phase_latency_p50["prepare"] == 2.0
        assert obs.step == 10


class TestAccordisReward:
    def test_defaults(self):
        r = AccordisReward()
        assert r.total == 0.0
        assert r.liveness_cost == 0.0
        assert r.block_commit == 0.0

    def test_total_computed(self):
        r = AccordisReward(
            liveness_cost=-1.0,
            block_commit=60.0,
            total=59.0,
        )
        assert r.total == 59.0


class TestAccordisTransform:
    def test_transform_creates_observation(self):
        t = AccordisTransform()
        raw = {
            "role": "leader",
            "current_view": 3,
            "phase_latency_p50": {"prepare": 2.0},
            "phase_latency_p99": {"prepare": 5.0},
            "qc_miss_streak": 0,
            "view_changes_last_50": 1,
            "equivocation_counts": {},
            "inter_message_variance": {},
            "suspected_peers": {},
            "committed_tps": 5.0,
            "pending_count": 100,
            "pipeline_utilisation": 0.8,
        }
        obs = t.transform(raw, "node_0", step=3, current_config=STATIC_BASELINE_CONFIG)
        assert isinstance(obs, AccordisObservation)
        assert obs.current_role == NodeRole.LEADER
        assert obs.current_view == 3
        assert obs.commit_throughput_tps == 5.0

    def test_transform_fallback_role(self):
        t = AccordisTransform()
        raw = {
            "role": "unknown_role",
            "current_view": 0,
            "phase_latency_p50": {},
            "phase_latency_p99": {},
            "qc_miss_streak": 0,
            "view_changes_last_50": 0,
            "equivocation_counts": {},
            "inter_message_variance": {},
            "suspected_peers": {},
            "committed_tps": 0.0,
            "pending_count": 0,
            "pipeline_utilisation": 0.0,
        }
        obs = t.transform(raw, "node_0", step=0, current_config=STATIC_BASELINE_CONFIG)
        assert obs.current_role == NodeRole.REPLICA

    def test_call_wraps_into_multi_node_observation(self):
        t = AccordisTransform()
        raw = {
            "role": "replica", "current_view": 0, "phase_latency_p50": {},
            "phase_latency_p99": {}, "qc_miss_streak": 0, "view_changes_last_50": 0,
            "equivocation_counts": {}, "inter_message_variance": {},
            "suspected_peers": {}, "committed_tps": 0.0,
            "pending_count": 0, "pipeline_utilisation": 0.0,
        }
        node_obs = t.transform(raw, "node_0", step=0, current_config=STATIC_BASELINE_CONFIG)
        obs_dict = {"node_0": node_obs}
        result = t(obs_dict)
        assert isinstance(result, MultiNodeObservation)
        assert result.nodes == obs_dict


class TestAccordisRubric:
    def test_grade_zero_on_violation(self):
        rubric = AccordisRubric()
        rewards = [AccordisReward(total=500.0)]
        results = VerifierResults(
            agreement=VerificationResult(passed=False, property="Agreement"),
            validity=VerificationResult(passed=True, property="Validity"),
            agreement_violated=True,
        )
        score = rubric.grade(rewards, results)
        assert score == 0.0

    def test_grade_normalised(self):
        rubric = AccordisRubric(max_possible_reward=1000.0, min_possible_reward=-1000.0)
        rewards = [AccordisReward(total=500.0)]
        results = VerifierResults(
            agreement=VerificationResult(passed=True, property="Agreement"),
            validity=VerificationResult(passed=True, property="Validity"),
        )
        score = rubric.grade(rewards, results)
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.75) < 0.01  # (500 - (-1000)) / (1000 - (-1000)) = 0.75


class TestEnums:
    def test_phase_enum(self):
        assert Phase.PREPARE.value == "prepare"
        assert Phase.PRE_COMMIT.value == "pre_commit"
        assert Phase.COMMIT.value == "commit"
        assert Phase.DECIDE.value == "decide"

    def test_node_role_enum(self):
        assert NodeRole.LEADER.value == "leader"
        assert NodeRole.REPLICA.value == "replica"
        assert NodeRole.CANDIDATE.value == "candidate"

    def test_bfa_strategy_enum(self):
        assert BFAStrategy.NONE.value == "none"
        assert BFAStrategy.EQUIVOCATION.value == "equivocation"
        assert BFAStrategy.ADAPTIVE_MIRROR.value == "adaptive_mirror"
        assert len(list(BFAStrategy)) == 8

    def test_leader_rotation_enum(self):
        assert LeaderRotation.ROUND_ROBIN.value == "round_robin"
        assert LeaderRotation.VRF.value == "vrf"
        assert LeaderRotation.REPUTATION_WEIGHTED.value == "reputation_weighted"


class TestStateModels:
    def test_transaction(self):
        tx = Transaction(id="tx_0", submitted_at=0)
        assert tx.id == "tx_0"

    def test_block(self):
        tx = Transaction(id="tx_0", submitted_at=0)
        block = Block(slot=1, hash="abc123", proposer_id="node_0", transactions=[tx])
        assert block.slot == 1
        assert len(block.transactions) == 1

    def test_node_state(self):
        ns = NodeState(node_id="node_0", is_byzantine=False)
        assert ns.current_view == 0
        assert ns.is_byzantine is False
        assert ns.committed_log == []

    def test_accordis_state(self):
        state = AccordisState(
            episode_id="ep_1",
            n_nodes=4,
            f_byzantine=1,
        )
        assert state.step == 0
        assert state.curriculum_level == 1

    def test_episode_log(self):
        log = EpisodeLog(
            episode_id="ep_1",
            curriculum_level=1,
            bfa_strategy=BFAStrategy.NONE,
            bfa_strategy_seed=42,
        )
        assert log.final_score == 0.0
        assert log.steps == []
