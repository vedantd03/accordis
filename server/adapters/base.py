"""BaseConsensusAdapter — the only seam between the environment logic layer
and the version-specific adapter implementation.

Layer 2 (AccordisEnvironment, oracle, reward, curriculum) imports ONLY from this file.
No Layer 2 file may import from server/adapters/simulated/ or server/adapters/librabft/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from accordis.models import NodeID, BFTConfig, BFAStrategy, LeaderRotation


class BaseConsensusAdapter(ABC):
    """Interface contract between environment logic and consensus engine implementation."""

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @abstractmethod
    def start_cluster(
        self,
        n_nodes: int,
        f_byzantine: int,
        leader_rotation: LeaderRotation,
        pool_size: int = 1000,
    ) -> List[NodeID]:
        """Start a fresh cluster of n_nodes. Designate f_byzantine nodes as Byzantine.
        Return the ordered list of all node IDs (honest first, Byzantine last).
        Must be callable multiple times — each call fully replaces the prior cluster.
        pool_size controls the number of transactions in the episode transaction pool.
        """
        ...

    @abstractmethod
    def stop_cluster(self) -> None:
        """Tear down the running cluster and release all resources.
        Must be idempotent — safe to call when no cluster is running.
        """
        ...

    # ── Configuration ─────────────────────────────────────────────────────────

    @abstractmethod
    def apply_config(self, node_id: NodeID, config: BFTConfig) -> None:
        """Write a complete BFTConfig to a single honest node.
        Takes effect before the next call to advance_one_step().
        Byzantine node IDs must never be passed to this method.
        """
        ...

    @abstractmethod
    def configure_network(self, curriculum_level: int, node_ids: List[NodeID]) -> None:
        """Apply the FaultProfile for the given curriculum level to all inter-node links.
        Called once per reset(). May also be called mid-episode for regime-switch levels.
        """
        ...

    # ── Execution ─────────────────────────────────────────────────────────────

    @abstractmethod
    def advance_one_step(self) -> None:
        """Tick the cluster forward by exactly one logical timestep.
        On return, the following are guaranteed:
          - All pending messages with delivery_tick <= current_tick are processed.
          - View timeouts have fired for any node that exceeded view_timeout_ms.
          - QC formation has been attempted. Accepted on 2f+1 votes (no crypto check).
          - Byzantine nodes have executed any injected actions for this tick.
          - Network fault model has been applied to all messages in transit.
        This method is synchronous. Returns only when the full tick is complete.
        """
        ...

    # ── Observation ──────────────────────────────────────────────────────────

    @abstractmethod
    def read_observation(self, node_id: NodeID) -> dict:
        """Return raw metrics for a single honest node as a plain Python dict.
        The dict must contain exactly these keys:
          role, current_view, phase_latency_p50, phase_latency_p99,
          qc_miss_streak, view_changes_last_50, equivocation_counts,
          inter_message_variance, suspected_peers, committed_tps,
          pending_count, pipeline_utilisation
        Byzantine node IDs must never be passed to this method.
        """
        ...

    @abstractmethod
    def get_committed_log(self, node_id: NodeID) -> List[dict]:
        """Return the full committed block list for node_id as a list of plain dicts.
        Each dict must be deserializable to a Block model.
        Used exclusively by CorrectnessOracle.
        """
        ...

    @abstractmethod
    def get_honest_nodes(self) -> List[NodeID]:
        """Return the current list of non-Byzantine node IDs."""
        ...

    @abstractmethod
    def get_finalized_txn_count(self) -> int:
        """Return the number of transactions for which a QC has been formed.
        This is the authoritative finality signal — independent of per-node
        propagation lag. A transaction counts the moment 2f+1 votes produced
        a QC, before replicas have necessarily delivered it.
        """
        ...

    @abstractmethod
    def get_byzantine_nodes(self) -> List[NodeID]:
        """Return the current list of Byzantine node IDs."""
        ...

    # ── Byzantine Injection ──────────────────────────────────────────────────

    @abstractmethod
    def inject_byzantine_action(
        self,
        byzantine_node_id: NodeID,
        strategy: BFAStrategy,
        target_nodes: List[NodeID],
        parameters: dict,
    ) -> None:
        """Instruct a Byzantine node to execute a disruption strategy on the immediately
        following advance_one_step() call only. The effect does not persist across steps.

        Strategy-to-parameter mapping:
          NONE              → parameters: {}
          RANDOM_DELAY      → parameters: {delay_ms: int}
          SELECTIVE_DELAY   → parameters: {delay_ms: int, targets: List[NodeID]}
          EQUIVOCATION      → parameters: {targets_A: List[NodeID], targets_B: List[NodeID]}
          LEADER_SUPPRESS   → parameters: {targets: List[NodeID]}
          CASCADE_TIMING    → parameters: {delay_ms: int}
          RECOVERY_DELAY    → parameters: {delay_ms: int}
          ADAPTIVE_MIRROR   → parameters: {view_timeout_ms: int, delta_ms: int}
        """
        ...
