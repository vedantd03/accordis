"""Typed models for the Accordis synchronous distributed-consensus environment.

The key API decision in this module is intentional:

- Observations are agent-scoped. A caller receives exactly one observer's view.
- Actions are round-scoped. A single environment step represents one synchronous
  consensus round and therefore accepts one joint action containing all node
  decisions for that round, plus an optional adversary intervention.

This keeps the environment faithful to synchronous distributed consensus while
still exposing a partially observed interface to each training agent.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Any

from openenv.core.env_server.types import Action, Observation
from openenv.core.env_server.interfaces import Transform
from openenv.core.rubrics import Rubric
from pydantic import BaseModel, Field

NodeID = str


class Phase(str, Enum):
    PREPARE    = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT     = "commit"
    DECIDE     = "decide"


class NodeRole(str, Enum):
    LEADER    = "leader"
    REPLICA   = "replica"
    CANDIDATE = "candidate"


class LeaderRotation(str, Enum):
    ROUND_ROBIN         = "round_robin"
    VRF                 = "vrf"
    REPUTATION_WEIGHTED = "reputation_weighted"


class BFAStrategy(str, Enum):
    NONE            = "none"
    RANDOM_DELAY    = "random_delay"
    SELECTIVE_DELAY = "selective_delay"
    EQUIVOCATION    = "equivocation"
    LEADER_SUPPRESS = "leader_suppression"
    CASCADE_TIMING  = "cascade_timing"
    RECOVERY_DELAY  = "recovery_delay"
    ADAPTIVE_MIRROR = "adaptive_mirror"


class BFTConfig(BaseModel):
    """
    Tunable configuration for a single honest node.
    quorum_size is enforced by the adapter — NOT a tunable field.
    All bounds enforced by AccordisEnvironment._clamp_action() before reaching adapter.
    """
    view_timeout_ms:             int = 2000
    pipeline_depth:              int = 2
    replication_batch_size:      int = 64
    equivocation_threshold:      int = 5
    vote_aggregation_timeout_ms: int = 500


SAFE_BFT_TUNING_BOUNDS: Dict[str, Tuple[int, int]] = {
    "view_timeout_ms":             (200, 10000),
    "pipeline_depth":              (1, 8),
    "replication_batch_size":      (1, 512),
    "equivocation_threshold":      (1, 15),
    "vote_aggregation_timeout_ms": (50, 1000),
}

STATIC_BASELINE_CONFIG = BFTConfig()


class AccordisObservation(Observation):
    """
    Per-node partial observation. Each honest node sees only its own metrics.
    Agent cannot observe Byzantine node internals or other nodes' observations.
    """
    node_id:      NodeID
    current_role: NodeRole
    current_view: int

    per_phase_latency_p50:    Dict[str, float] = Field(default_factory=dict)
    per_phase_latency_p99:    Dict[str, float] = Field(default_factory=dict)
    qc_formation_miss_streak: int = 0
    view_change_count_recent: int = 0

    equivocation_miss_streak:  Dict[NodeID, int]   = Field(default_factory=dict)
    message_arrival_variance:  Dict[NodeID, float] = Field(default_factory=dict)
    suspected_byzantine:       Dict[NodeID, bool]  = Field(default_factory=dict)

    commit_throughput_tps:  float = 0.0
    pending_txn_count:      int   = 0
    pipeline_utilisation:   float = 0.0

    current_config: BFTConfig = Field(default_factory=BFTConfig)
    step:           int = 0


class AccordisAction(Action):
    """
    Configuration assignment for a single honest node per step.
    All values are clamped to SAFE_BFT_TUNING_BOUNDS by AccordisEnvironment before
    being passed to the adapter. The adapter always receives valid, in-bounds values.
    """
    node_id:                     NodeID
    view_timeout_ms:             int
    pipeline_depth:              int
    replication_batch_size:      int
    equivocation_threshold:      int
    vote_aggregation_timeout_ms: int
    suspect_node:                Optional[NodeID] = None
    clear_suspicion:             Optional[NodeID] = None


MultiNodeAction = Dict[NodeID, AccordisAction]


class AccordisReward(BaseModel):
    """
    Per-step reward breakdown. total is the sum used for policy gradient.
    Final episode reward normalised to [0.0, 1.0] by AccordisRubric.
    """
    liveness_cost:           float = 0.0
    unnecessary_view_change: float = 0.0
    liveness_stall:          float = 0.0
    block_commit:            float = 0.0
    fast_leader_recovery:    float = 0.0
    false_positive_avoided:  float = 0.0
    pipeline_efficiency:     float = 0.0
    throughput_improvement:  float = 0.0
    latency_improvement:     float = 0.0
    total:                   float = 0.0


class Transaction(BaseModel):
    id:           str
    submitted_at: int


class Block(BaseModel):
    slot:         int
    hash:         str
    proposer_id:  NodeID
    transactions: List[Transaction] = Field(default_factory=list)


class NodeState(BaseModel):
    node_id:       NodeID
    is_byzantine:  bool
    committed_log: List[Block] = Field(default_factory=list)
    current_view:  int = 0
    current_role:  NodeRole = NodeRole.REPLICA
    config:        BFTConfig = Field(default_factory=BFTConfig)


class ProposalRegistry(BaseModel):
    honest_proposals: Dict[str, Transaction] = Field(default_factory=dict)


class AccordisState(BaseModel):
    """Full true state — accessible to oracle and environment only. Never exposed to agent."""
    episode_id:        str
    step:              int = 0
    curriculum_level:  int = 1
    n_nodes:           int = 4
    f_byzantine:       int = 0
    leader_rotation:   LeaderRotation = LeaderRotation.ROUND_ROBIN
    node_states:       Dict[NodeID, NodeState] = Field(default_factory=dict)
    view_change_count: int = 0
    bfa_strategy:      BFAStrategy = BFAStrategy.NONE
    proposal_registry: ProposalRegistry = Field(default_factory=ProposalRegistry)
    episode_txn_pool:  List[Transaction] = Field(default_factory=list)


class VerificationResult(BaseModel):
    passed:   bool
    property: Literal["Agreement", "Validity", "Liveness"]
    evidence: str = ""


class LivenessResult(BaseModel):
    committed_count:      int = 0
    pending_count:        int = 0
    liveness_rate:        float = 0.0
    view_change_overhead: int = 0


class VerifierResults(BaseModel):
    agreement:          VerificationResult
    validity:           VerificationResult
    liveness:           Optional[LivenessResult] = None
    agreement_violated: bool = False
    validity_violated:  bool = False


class BaselineComparison(BaseModel):
    baseline_throughput_tps:      float = 0.0
    baseline_view_change_count:   int = 0
    baseline_commit_latency_p99:  float = 0.0
    relative_tps_improvement:     float = 0.0
    relative_latency_improvement: float = 0.0


class EpisodeLog(BaseModel):
    episode_id:        str
    curriculum_level:  int
    bfa_strategy:      BFAStrategy
    bfa_strategy_seed: int
    steps:             List[AccordisState] = Field(default_factory=list)
    rewards:           List[AccordisReward] = Field(default_factory=list)
    final_score:       float = 0.0


class AccordisTransform(Transform):
    """
    Converts a raw metrics dict (from BaseConsensusAdapter.read_observation()) into
    a typed AccordisObservation. Version-agnostic: the dict schema is identical
    from both SimulatedConsensusAdapter and LibraBFTAdapter.
    """
    smoothing_window:     int   = 10
    normalise_latency_ms: float = 10000.0

    def transform(
        self, raw_metrics: dict, node_id: NodeID, step: int, current_config: BFTConfig
    ) -> AccordisObservation:
        role_str = raw_metrics.get("role", "replica")
        try:
            role = NodeRole(role_str)
        except ValueError:
            role = NodeRole.REPLICA

        p50_raw = raw_metrics.get("phase_latency_p50", {})
        p99_raw = raw_metrics.get("phase_latency_p99", {})

        return AccordisObservation(
            node_id=node_id,
            current_role=role,
            current_view=raw_metrics.get("current_view", 0),
            per_phase_latency_p50={k: float(v) for k, v in p50_raw.items()},
            per_phase_latency_p99={k: float(v) for k, v in p99_raw.items()},
            qc_formation_miss_streak=raw_metrics.get("qc_miss_streak", 0),
            view_change_count_recent=raw_metrics.get("view_changes_last_50", 0),
            equivocation_miss_streak={k: int(v) for k, v in raw_metrics.get("equivocation_counts", {}).items()},
            message_arrival_variance={k: float(v) for k, v in raw_metrics.get("inter_message_variance", {}).items()},
            suspected_byzantine={k: bool(v) for k, v in raw_metrics.get("suspected_peers", {}).items()},
            commit_throughput_tps=float(raw_metrics.get("committed_tps", 0.0)),
            pending_txn_count=int(raw_metrics.get("pending_count", 0)),
            pipeline_utilisation=float(raw_metrics.get("pipeline_utilisation", 0.0)),
            current_config=current_config,
            step=step,
        )

    def __call__(self, observation: AccordisObservation) -> AccordisObservation:
        return observation


class AccordisRubric(Rubric):
    """
    Episode-level grader. Normalises cumulative reward to [0.0, 1.0].
    Used by openenv validate and all task graders.
    """
    task_id:             str   = ""
    curriculum_level:    int   = 1
    max_possible_reward: float = 1000.0
    min_possible_reward: float = -1000.0

    def __init__(
        self,
        task_id: str = "",
        curriculum_level: int = 1,
        max_possible_reward: float = 1000.0,
        min_possible_reward: float = -1000.0,
    ):
        super().__init__()
        self.task_id = task_id
        self.curriculum_level = curriculum_level
        self.max_possible_reward = max_possible_reward
        self.min_possible_reward = min_possible_reward

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def grade(
        self,
        episode_rewards: List[AccordisReward],
        verifier_results: VerifierResults,
    ) -> float:
        if verifier_results.agreement_violated or verifier_results.validity_violated:
            return 0.0
        total = sum(r.total for r in episode_rewards)
        denom = self.max_possible_reward - self.min_possible_reward
        if denom == 0:
            return 0.0
        score = (total - self.min_possible_reward) / denom
        return max(0.0, min(1.0, score))
