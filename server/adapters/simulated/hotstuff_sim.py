"""HotStuffSimulator — in-memory HotStuff phase engine.

Implements PREPARE → PRE-COMMIT → COMMIT → DECIDE phases for all nodes
in a single synchronous tick per advance_one_step() call.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from accordis.models import (
    BFTConfig,
    BFAStrategy,
    Block,
    LeaderRotation,
    NodeID,
    NodeRole,
    Phase,
    Transaction,
)


# ── Message types ──────────────────────────────────────────────────────────────

PROPOSAL  = "PROPOSAL"
VOTE      = "VOTE"
NEW_VIEW  = "NEW_VIEW"
QC        = "QC"


@dataclass
class Message:
    msg_type:   str
    sender_id:  NodeID
    target_id:  NodeID
    payload:    Dict[str, Any]
    extra_delay_ticks: int = 0


# ── Simulated Node ─────────────────────────────────────────────────────────────

class SimulatedNode:
    def __init__(self, node_id: NodeID, is_byzantine: bool, config: BFTConfig) -> None:
        self.node_id      = node_id
        self.is_byzantine = is_byzantine
        self.config       = config

        self.current_view: int = 0
        self.current_role: NodeRole = NodeRole.REPLICA
        self.committed_log: List[Block] = []

        # Inbound message queue (delivered by NetworkSimulator)
        self.inbound_queue: List[Message] = []

        # Rolling history for latency metrics (per phase)
        self.phase_latency_history: Dict[str, Deque[float]] = {
            Phase.PREPARE.value:    deque(maxlen=50),
            Phase.PRE_COMMIT.value: deque(maxlen=50),
            Phase.COMMIT.value:     deque(maxlen=50),
            Phase.DECIDE.value:     deque(maxlen=50),
        }

        self.view_change_count: int = 0
        self.view_change_history: Deque[int] = deque(maxlen=50)  # views where changes happened
        self.no_commit_streak: int = 0

        # Set by inject_byzantine_action, cleared after each tick
        self.pending_byzantine_action: Optional[Dict[str, Any]] = None

        # Equivocation tracking: {peer_id: count of inconsistent messages observed}
        self.equivocation_counts: Dict[NodeID, int] = {}
        # Inter-message arrival times for variance computation
        self.message_arrival_times: Dict[NodeID, Deque[int]] = {}
        # Suspicion flags set by the agent
        self.suspected_peers: Dict[NodeID, bool] = {}

        # For TPS computation
        self.total_committed_txns: int = 0
        self.ticks_elapsed: int = 0

        # Pipeline tracking
        self.pipeline_utilisation: float = 0.0
        self.in_flight_slots: int = 0

        # QC miss streak (consecutive rounds without valid QC)
        self.qc_miss_streak: int = 0

        # Track votes received per view for QC formation
        self.votes_received: Dict[int, Dict[str, List[NodeID]]] = {}  # view → {hash → [voters]}
        # Track NEW_VIEW messages received
        self.new_views_received: Dict[int, List[NodeID]] = {}

        # Last tick a proposal was received
        self.last_proposal_tick: Optional[int] = None
        # Pending block for current view (waiting for QC)
        self.pending_block: Optional[Block] = None


# ── HotStuff Simulator ─────────────────────────────────────────────────────────

class HotStuffSimulator:
    """Coordinates all SimulatedNode objects through the HotStuff phase sequence."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._nodes: Dict[NodeID, SimulatedNode] = {}
        self._honest_nodes: List[NodeID] = []
        self._byzantine_nodes: List[NodeID] = []
        self._all_nodes: List[NodeID] = []
        self._current_tick: int = 0
        self._f_byzantine: int = 0
        self._quorum_size: int = 3  # 2f+1
        self._leader_rotation: LeaderRotation = LeaderRotation.ROUND_ROBIN
        self._current_view: int = 0
        self._txn_counter: int = 0
        self._network_sim = None  # set by adapter

        # Reputation tracking for REPUTATION_WEIGHTED rotation
        self._node_reputation: Dict[NodeID, float] = {}

        # Episode transaction pool (shared reference from adapter)
        self._episode_txn_pool: List[Transaction] = []
        self._committed_txn_ids: set = set()

    def setup(
        self,
        nodes: Dict[NodeID, SimulatedNode],
        honest_nodes: List[NodeID],
        byzantine_nodes: List[NodeID],
        f_byzantine: int,
        leader_rotation: LeaderRotation,
        network_sim: Any,
        txn_pool: List[Transaction],
    ) -> None:
        self._nodes = nodes
        self._honest_nodes = honest_nodes
        self._byzantine_nodes = byzantine_nodes
        self._all_nodes = honest_nodes + byzantine_nodes
        self._f_byzantine = f_byzantine
        self._quorum_size = 2 * f_byzantine + 1
        self._leader_rotation = leader_rotation
        self._network_sim = network_sim
        self._current_view = 0
        self._current_tick = 0
        self._episode_txn_pool = txn_pool
        self._committed_txn_ids = set()
        self._node_reputation = {nid: 1.0 for nid in honest_nodes}

    def _get_leader(self, view: int) -> NodeID:
        honest = self._honest_nodes
        if not honest:
            return self._all_nodes[0] if self._all_nodes else "node_0"

        if self._leader_rotation == LeaderRotation.ROUND_ROBIN:
            return honest[view % len(honest)]

        elif self._leader_rotation == LeaderRotation.VRF:
            # Deterministic hash-based selection
            h = int(hashlib.sha256(f"{self._rng.random()}{view}".encode()).hexdigest(), 16)
            return honest[h % len(honest)]

        elif self._leader_rotation == LeaderRotation.REPUTATION_WEIGHTED:
            weights = [max(0.01, self._node_reputation.get(nid, 1.0)) for nid in honest]
            total = sum(weights)
            norm = [w / total for w in weights]
            r = self._rng.random()
            cumulative = 0.0
            for nid, w in zip(honest, norm):
                cumulative += w
                if r <= cumulative:
                    return nid
            return honest[-1]

        return honest[view % len(honest)]

    def _create_block(self, leader_id: NodeID, view: int) -> Block:
        """Create a new block with transactions from the pool."""
        config = self._nodes[leader_id].config
        batch_size = config.replication_batch_size

        # Grab uncommitted transactions
        uncommitted = [
            tx for tx in self._episode_txn_pool
            if tx.id not in self._committed_txn_ids
        ][:batch_size]

        block_hash = hashlib.sha256(
            f"{view}:{leader_id}:{[tx.id for tx in uncommitted]}".encode()
        ).hexdigest()[:16]

        return Block(
            slot=view,
            hash=block_hash,
            proposer_id=leader_id,
            transactions=uncommitted,
        )

    def _send_message(
        self,
        sender_id: NodeID,
        target_id: NodeID,
        msg_type: str,
        payload: Dict[str, Any],
        extra_delay: int = 0,
    ) -> None:
        msg = Message(
            msg_type=msg_type,
            sender_id=sender_id,
            target_id=target_id,
            payload=payload,
            extra_delay_ticks=extra_delay,
        )
        if self._network_sim is not None:
            self._network_sim.dispatch(
                target_node_id=target_id,
                message=msg,
                current_tick=self._current_tick,
                extra_delay_ticks=extra_delay,
            )

    def _broadcast(
        self,
        sender_id: NodeID,
        msg_type: str,
        payload: Dict[str, Any],
        extra_delay: int = 0,
        exclude: Optional[List[NodeID]] = None,
    ) -> None:
        for nid in self._all_nodes:
            if nid == sender_id:
                continue
            if exclude and nid in exclude:
                continue
            self._send_message(sender_id, nid, msg_type, payload, extra_delay)

    def _get_extra_delay(self, node: SimulatedNode, target_id: NodeID) -> int:
        """Determine extra delay from pending_byzantine_action."""
        action = node.pending_byzantine_action
        if action is None:
            return 0
        strategy = action.get("strategy", "")
        if strategy == "random_delay":
            return action.get("extra_delay_ticks", 0)
        if strategy == "selective_delay":
            targets = action.get("targets")
            if targets is None or target_id in targets:
                return action.get("extra_delay_ticks", 0)
        if strategy == "recovery_delay":
            # Applied only to NEW_VIEW messages — handled separately
            return 0
        return 0

    def _should_suppress(self, node: SimulatedNode, target_id: NodeID) -> bool:
        """Check if LEADER_SUPPRESS action suppresses a message to target_id."""
        action = node.pending_byzantine_action
        if action is None:
            return False
        if action.get("strategy") == "leader_suppress":
            targets = action.get("targets", [])
            return target_id in targets
        return False

    def tick(self, delivered_messages: Dict[str, List[Message]]) -> None:
        """Process one full HotStuff round for all nodes."""
        # Deliver inbound messages
        for nid, messages in delivered_messages.items():
            if nid in self._nodes:
                node = self._nodes[nid]
                for msg in messages:
                    node.inbound_queue.append(msg)
                    # Track arrival times for variance
                    sender = msg.sender_id
                    if sender not in node.message_arrival_times:
                        node.message_arrival_times[sender] = deque(maxlen=20)
                    node.message_arrival_times[sender].append(self._current_tick)

        # Determine leader for current view
        leader_id = self._get_leader(self._current_view)

        # Update node roles
        for nid, node in self._nodes.items():
            node.current_view = self._current_view
            if nid == leader_id:
                node.current_role = NodeRole.LEADER
            else:
                node.current_role = NodeRole.REPLICA

        # ── PREPARE: Leader sends PROPOSAL ────────────────────────────────────
        if leader_id in self._nodes:
            leader_node = self._nodes[leader_id]
            block = self._create_block(leader_id, self._current_view)
            leader_node.pending_block = block

            for target_id in self._all_nodes:
                if target_id == leader_id:
                    continue
                if leader_node.is_byzantine and self._should_suppress(leader_node, target_id):
                    continue
                extra = self._get_extra_delay(leader_node, target_id) if leader_node.is_byzantine else 0
                self._send_message(
                    sender_id=leader_id,
                    target_id=target_id,
                    msg_type=PROPOSAL,
                    payload={"view": self._current_view, "block": block, "proposer_id": leader_id},
                    extra_delay=extra,
                )

            # Leader also processes the proposal locally
            if not leader_node.is_byzantine:
                leader_node.phase_latency_history[Phase.PREPARE.value].append(float(self._current_tick))

        # ── PRE-COMMIT: Replicas vote on received proposals ───────────────────
        votes_this_round: Dict[str, List[NodeID]] = {}  # block_hash → [voters]

        for nid, node in self._nodes.items():
            if node.is_byzantine:
                action = node.pending_byzantine_action
                if action and action.get("strategy") == "equivocation":
                    # Send two conflicting votes
                    targets_A = action.get("targets_A", [])
                    targets_B = action.get("targets_B", [])
                    hash_A = f"fake_hash_A_{self._current_view}"
                    hash_B = f"fake_hash_B_{self._current_view}"
                    for tgt in targets_A:
                        self._send_message(nid, tgt, VOTE, {
                            "view": self._current_view,
                            "block_hash": hash_A,
                            "voter_id": nid,
                        })
                    for tgt in targets_B:
                        self._send_message(nid, tgt, VOTE, {
                            "view": self._current_view,
                            "block_hash": hash_B,
                            "voter_id": nid,
                        })
                continue  # byzantine nodes don't vote honestly

            # Check if we received a valid proposal this tick
            proposal_received = None
            for msg in node.inbound_queue:
                if msg.msg_type == PROPOSAL and msg.payload.get("view") == self._current_view:
                    proposal_received = msg.payload
                    break

            if proposal_received or nid == leader_id:
                # Vote for the block
                block = proposal_received["block"] if proposal_received else node.pending_block
                if block is None:
                    continue
                extra = self._get_extra_delay(node, leader_id)
                self._send_message(
                    sender_id=nid,
                    target_id=leader_id,
                    msg_type=VOTE,
                    payload={
                        "view": self._current_view,
                        "block_hash": block.hash,
                        "voter_id": nid,
                    },
                    extra_delay=extra,
                )
                if block.hash not in votes_this_round:
                    votes_this_round[block.hash] = []
                votes_this_round[block.hash].append(nid)
                node.phase_latency_history[Phase.PRE_COMMIT.value].append(float(self._current_tick))

        # ── COMMIT: Leader collects votes and forms QC ────────────────────────
        qc_formed = False
        committed_block: Optional[Block] = None
        view = self._current_view

        if leader_id in self._nodes:
            leader_node = self._nodes[leader_id]
            if view not in leader_node.votes_received:
                leader_node.votes_received[view] = {}

            # Collect votes from inbound queue
            for msg in leader_node.inbound_queue:
                if msg.msg_type == VOTE and msg.payload.get("view") == view:
                    bh = msg.payload["block_hash"]
                    voter = msg.payload["voter_id"]
                    if bh not in leader_node.votes_received[view]:
                        leader_node.votes_received[view][bh] = []
                    if voter not in leader_node.votes_received[view][bh]:
                        leader_node.votes_received[view][bh].append(voter)
                    # Detect equivocation (same voter, different hashes)
                    for other_hash, voters in leader_node.votes_received[view].items():
                        if other_hash != bh and voter in voters:
                            leader_node.equivocation_counts[voter] = leader_node.equivocation_counts.get(voter, 0) + 1

            # Also include this-round votes
            for bh, voters in votes_this_round.items():
                if bh not in leader_node.votes_received[view]:
                    leader_node.votes_received[view][bh] = []
                for v in voters:
                    if v not in leader_node.votes_received[view][bh]:
                        leader_node.votes_received[view][bh].append(v)

            # Check if we have 2f+1 votes for any hash
            best_hash = None
            best_count = 0
            for bh, voters in leader_node.votes_received.get(view, {}).items():
                if len(voters) > best_count:
                    best_count = len(voters)
                    best_hash = bh

            config = leader_node.config
            if best_count >= self._quorum_size and best_hash is not None:
                qc_formed = True
                committed_block = leader_node.pending_block
                if committed_block and committed_block.hash == best_hash:
                    leader_node.phase_latency_history[Phase.COMMIT.value].append(float(self._current_tick))
                    # Broadcast QC
                    self._broadcast(
                        sender_id=leader_id,
                        msg_type=QC,
                        payload={"view": view, "block_hash": best_hash, "block": committed_block},
                    )
            else:
                # Check for view timeout
                timeout_ticks = config.view_timeout_ms
                if self._current_tick > 0 and (self._current_tick % max(1, timeout_ticks // 10)) == 0:
                    # Trigger view change
                    self._trigger_view_change(leader_id)

        # ── DECIDE: Commit the block on all nodes that received QC ────────────
        committed_any = False
        for nid, node in self._nodes.items():
            if node.is_byzantine:
                continue

            for msg in node.inbound_queue:
                if msg.msg_type == QC and msg.payload.get("view") == view:
                    block_in_qc = msg.payload.get("block")
                    if block_in_qc and isinstance(block_in_qc, Block):
                        # Check not already committed
                        existing_slots = {b.slot for b in node.committed_log}
                        if block_in_qc.slot not in existing_slots:
                            node.committed_log.append(block_in_qc)
                            for tx in block_in_qc.transactions:
                                self._committed_txn_ids.add(tx.id)
                            node.total_committed_txns += len(block_in_qc.transactions)
                            committed_any = True
                    node.phase_latency_history[Phase.DECIDE.value].append(float(self._current_tick))
                    break

            # Leader also commits its own block if QC formed
            if nid == leader_id and qc_formed and committed_block:
                existing_slots = {b.slot for b in node.committed_log}
                if committed_block.slot not in existing_slots:
                    node.committed_log.append(committed_block)
                    for tx in committed_block.transactions:
                        self._committed_txn_ids.add(tx.id)
                    node.total_committed_txns += len(committed_block.transactions)
                    committed_any = True

        # ── Metrics Update ────────────────────────────────────────────────────
        for nid, node in self._nodes.items():
            # Always clear byzantine action after tick (spec: effect consumed after one tick)
            node.pending_byzantine_action = None
            node.inbound_queue = []

            if node.is_byzantine:
                continue
            node.ticks_elapsed += 1

            if committed_any:
                node.no_commit_streak = 0
                node.qc_miss_streak = 0
            else:
                node.no_commit_streak += 1
                node.qc_miss_streak += 1

            # Pipeline utilisation
            pipeline_depth = node.config.pipeline_depth
            # Approximate: in-flight = min(pipeline_depth, view - last_committed_view)
            node.pipeline_utilisation = min(
                1.0,
                max(0.0, 1.0 - (node.no_commit_streak / max(1, pipeline_depth * 5)))
            )

        # Advance view if QC formed
        if qc_formed:
            self._current_view += 1
            # Update reputation for the leader
            if leader_id in self._node_reputation:
                self._node_reputation[leader_id] = min(
                    2.0, self._node_reputation[leader_id] + 0.1
                )

        self._current_tick += 1

    def _trigger_view_change(self, initiator_id: NodeID) -> None:
        """Broadcast NEW_VIEW messages and elect a new leader."""
        new_view = self._current_view + 1
        for nid, node in self._nodes.items():
            if node.is_byzantine:
                continue
            node.view_change_count += 1
            node.view_change_history.append(self._current_view)
            if self._node_reputation.get(self._get_leader(self._current_view)):
                leader = self._get_leader(self._current_view)
                self._node_reputation[leader] = max(
                    0.01, self._node_reputation.get(leader, 1.0) - 0.2
                )

        # Send NEW_VIEW to all
        for target_id in self._honest_nodes:
            extra = 0
            sender_node = self._nodes.get(initiator_id)
            if sender_node and sender_node.is_byzantine:
                action = sender_node.pending_byzantine_action
                if action and action.get("strategy") == "recovery_delay":
                    extra = action.get("extra_delay_ticks", 0)
            self._send_message(
                sender_id=initiator_id,
                target_id=target_id,
                msg_type=NEW_VIEW,
                payload={"from_view": self._current_view, "new_view": new_view, "sender_id": initiator_id},
                extra_delay=extra,
            )
        self._current_view = new_view

    def get_node(self, node_id: NodeID) -> Optional[SimulatedNode]:
        return self._nodes.get(node_id)

    def get_current_view(self) -> int:
        return self._current_view

    def get_current_tick(self) -> int:
        return self._current_tick

    def get_committed_txn_ids(self) -> set:
        return self._committed_txn_ids
