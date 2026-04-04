"""SimulatedConsensusAdapter — Version 1 implementation of BaseConsensusAdapter.

Coordinates hotstuff_sim, network_sim, and bfa_sim to provide a fully in-memory,
zero-dependency implementation of the consensus adapter interface.
"""

from __future__ import annotations

import random
import statistics
from collections import deque
from typing import Any, Dict, List, Optional

from accordis.models import (
    BFAStrategy,
    BFTConfig,
    Block,
    LeaderRotation,
    NodeID,
    NodeRole,
    Phase,
    Transaction,
)
from accordis.server.adapters.base import BaseConsensusAdapter
from accordis.server.adapters.simulated.hotstuff_sim import HotStuffSimulator, SimulatedNode
from accordis.server.adapters.simulated.network_sim import NetworkSimulator
from accordis.server.adapters.simulated.bfa_sim import ByzantineInjector
from accordis.server.network.fault_profiles import get_fault_profile


class SimulatedConsensusAdapter(BaseConsensusAdapter):
    """Version 1 adapter: fully in-process, deterministic, zero external dependencies."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._network_sim = NetworkSimulator(seed=seed)
        self._hotstuff = HotStuffSimulator(rng=self._rng)
        self._injector = ByzantineInjector()

        self._honest_nodes: List[NodeID] = []
        self._byzantine_nodes: List[NodeID] = []
        self._nodes: Dict[NodeID, SimulatedNode] = {}

        self._current_tick: int = 0
        self._episode_txn_pool: List[Transaction] = []
        self._txn_counter: int = 0
        self._is_running: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start_cluster(
        self,
        n_nodes: int,
        f_byzantine: int,
        leader_rotation: LeaderRotation,
    ) -> List[NodeID]:
        """Start a fresh cluster. Returns node IDs: honest first, Byzantine last."""
        self._is_running = True
        self._current_tick = 0
        self._rng = random.Random(self._seed)
        self._network_sim = NetworkSimulator(seed=self._seed)

        # Generate transaction pool
        self._txn_counter = 0
        self._episode_txn_pool = [
            Transaction(id=f"tx_{i}", submitted_at=0)
            for i in range(1000)
        ]
        self._txn_counter = 1000

        # Create nodes
        honest_count = n_nodes - f_byzantine
        honest_ids = [f"node_{i}" for i in range(honest_count)]
        byzantine_ids = [f"node_{i}" for i in range(honest_count, n_nodes)]

        default_config = BFTConfig()
        self._nodes = {}
        for nid in honest_ids:
            self._nodes[nid] = SimulatedNode(nid, is_byzantine=False, config=default_config)
        for nid in byzantine_ids:
            self._nodes[nid] = SimulatedNode(nid, is_byzantine=True, config=default_config)

        self._honest_nodes = honest_ids
        self._byzantine_nodes = byzantine_ids

        # Setup HotStuff simulator
        self._hotstuff.setup(
            nodes=self._nodes,
            honest_nodes=honest_ids,
            byzantine_nodes=byzantine_ids,
            f_byzantine=f_byzantine,
            leader_rotation=leader_rotation,
            network_sim=self._network_sim,
            txn_pool=self._episode_txn_pool,
        )

        return honest_ids + byzantine_ids

    def stop_cluster(self) -> None:
        """Idempotent teardown."""
        self._is_running = False
        self._nodes = {}
        self._honest_nodes = []
        self._byzantine_nodes = []

    # ── Configuration ─────────────────────────────────────────────────────────

    def apply_config(self, node_id: NodeID, config: BFTConfig) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].config = config

    def configure_network(self, curriculum_level: int, node_ids: List[NodeID]) -> None:
        profile = get_fault_profile(curriculum_level)
        self._network_sim.set_profile(profile)

    # ── Execution ─────────────────────────────────────────────────────────────

    def advance_one_step(self) -> None:
        """Tick the simulation forward by one logical timestep."""
        # Flush messages due this tick into node inbound queues
        delivered = self._network_sim.flush(self._current_tick)

        # Run one HotStuff round
        self._hotstuff.tick(delivered)

        self._current_tick += 1

    # ── Observation ──────────────────────────────────────────────────────────

    def read_observation(self, node_id: NodeID) -> dict:
        """Return raw metrics dict with exactly the 12 required keys."""
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        # Phase latency p50 and p99
        phase_latency_p50: Dict[str, float] = {}
        phase_latency_p99: Dict[str, float] = {}

        for phase_name, history in node.phase_latency_history.items():
            if len(history) == 0:
                phase_latency_p50[phase_name] = 0.0
                phase_latency_p99[phase_name] = 0.0
            elif len(history) == 1:
                val = float(history[0])
                phase_latency_p50[phase_name] = val
                phase_latency_p99[phase_name] = val
            else:
                sorted_vals = sorted(history)
                n = len(sorted_vals)
                p50_idx = max(0, int(n * 0.50) - 1)
                p99_idx = max(0, int(n * 0.99) - 1)
                phase_latency_p50[phase_name] = float(sorted_vals[p50_idx])
                phase_latency_p99[phase_name] = float(sorted_vals[p99_idx])

        # View changes in last 50 steps
        view_changes_last_50 = len(node.view_change_history)

        # Inter-message variance per sender
        inter_message_variance: Dict[NodeID, float] = {}
        for sender, times in node.message_arrival_times.items():
            if len(times) >= 2:
                diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
                try:
                    inter_message_variance[sender] = statistics.variance(diffs)
                except statistics.StatisticsError:
                    inter_message_variance[sender] = 0.0
            else:
                inter_message_variance[sender] = 0.0

        # TPS computation: average over the rolling window (last 10 ticks)
        window = node.recent_commit_counts
        committed_tps = sum(window) / max(1, len(window))

        # Pending count: total submitted - committed
        committed_ids: set = set()
        for block in node.committed_log:
            for tx in block.transactions:
                committed_ids.add(tx.id)
        pending_count = len(self._episode_txn_pool) - len(committed_ids)

        return {
            "role":                   node.current_role.value,
            "current_view":           node.current_view,
            "phase_latency_p50":      phase_latency_p50,
            "phase_latency_p99":      phase_latency_p99,
            "qc_miss_streak":         node.qc_miss_streak,
            "view_changes_last_50":   view_changes_last_50,
            "equivocation_counts":    dict(node.equivocation_counts),
            "inter_message_variance": inter_message_variance,
            "suspected_peers":        dict(node.suspected_peers),
            "committed_tps":          committed_tps,
            "pending_count":          max(0, pending_count),
            "pipeline_utilisation":   node.pipeline_utilisation,
        }

    def get_committed_log(self, node_id: NodeID) -> List[dict]:
        """Return committed log as list of dicts deserializable to Block."""
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [block.model_dump() for block in node.committed_log]

    def get_honest_nodes(self) -> List[NodeID]:
        return list(self._honest_nodes)

    def get_byzantine_nodes(self) -> List[NodeID]:
        return list(self._byzantine_nodes)

    def get_finalized_txn_count(self) -> int:
        return len(self._hotstuff.get_committed_txn_ids())

    # ── Byzantine Injection ──────────────────────────────────────────────────

    def inject_byzantine_action(
        self,
        byzantine_node_id: NodeID,
        strategy: BFAStrategy,
        target_nodes: List[NodeID],
        parameters: dict,
    ) -> None:
        node = self._nodes.get(byzantine_node_id)
        if node is None or not node.is_byzantine:
            return

        action = self._injector.build_action(strategy, target_nodes, parameters)
        node.pending_byzantine_action = action
