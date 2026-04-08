"""HotStuffSimulator — Chained HotStuff in-memory consensus engine.

Implements proper Chained HotStuff with:
- QC-based pipelining (each block carries a justify QC for the previous block)
- 3-chain commit rule
- Safe node predicate (locked_qc)
- Per-node pacemaker (view timeout + NEW_VIEW messages)
- Sync protocol for lagging nodes
- Full Byzantine strategy support (fork, equivocation, delays, suppression)
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

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
SYNC_REQ  = "SYNC_REQ"
SYNC_RESP = "SYNC_RESP"


# ── Constants ──────────────────────────────────────────────────────────────────

VIEW_TICK_MS: int = 50        # ms per tick — divide view_timeout_ms by this
SYNC_LAG_THRESHOLD: int = 3   # views behind before requesting sync


# ── Core data structures ───────────────────────────────────────────────────────

@dataclass
class Message:
    msg_type:          str
    sender_id:         NodeID
    target_id:         NodeID
    payload:           Dict[str, Any]
    extra_delay_ticks: int = 0


@dataclass
class QuorumCert:
    view:       int
    block_hash: str
    block:      Optional["SimBlock"] = None
    signers:    List[NodeID] = field(default_factory=list)


@dataclass
class SimBlock:
    slot:         int          # view number when proposed
    hash:         str
    parent_hash:  str          # "genesis" for the first block
    proposer_id:  NodeID
    transactions: List[Transaction]
    justify:      Optional[QuorumCert]  # QC leader used to propose this block

    def to_pydantic(self) -> Block:
        return Block(
            slot=self.slot,
            hash=self.hash,
            parent_hash=self.parent_hash,
            proposer_id=self.proposer_id,
            transactions=list(self.transactions),
        )


# Bootstrap / genesis state (created at module level so imports always work)
GENESIS_BLOCK = SimBlock(
    slot=-1,
    hash="genesis",
    parent_hash="genesis",
    proposer_id="genesis",
    transactions=[],
    justify=None,
)
GENESIS_QC = QuorumCert(
    view=-1,
    block_hash="genesis",
    block=GENESIS_BLOCK,
    signers=[],
)
# Circular reference is intentional — they mutually reference each other as bootstrap state
GENESIS_BLOCK.justify = GENESIS_QC


# ── Simulated Node ─────────────────────────────────────────────────────────────

class SimulatedNode:
    def __init__(self, node_id: NodeID, is_byzantine: bool, config: BFTConfig) -> None:
        self.node_id      = node_id
        self.is_byzantine = is_byzantine
        self.config       = config

        # ── Protocol state ────────────────────────────────────────────────────
        self.current_view:    int      = 0
        self.current_role:    NodeRole = NodeRole.REPLICA
        self.high_qc:         QuorumCert = GENESIS_QC
        self.locked_qc:       QuorumCert = GENESIS_QC
        self.voted_view:      int      = -1
        self.view_start_tick: int      = 0
        self.new_view_sent:   Set[int] = set()

        # ── Block store ───────────────────────────────────────────────────────
        self.block_store: Dict[str, SimBlock] = {"genesis": GENESIS_BLOCK}

        # ── Committed chain ───────────────────────────────────────────────────
        self.committed_log:    List[SimBlock] = []
        self.committed_hashes: Set[str]       = set()

        # ── Inbound message queue ─────────────────────────────────────────────
        self.inbound_queue: List[Message] = []

        # ── Leader collections ────────────────────────────────────────────────
        # new_views_for[view][sender_id] = high_qc they sent
        self.new_views_for:  Dict[int, Dict[NodeID, QuorumCert]] = {}
        # votes_for[view][block_hash] = [voter_ids]
        self.votes_for:      Dict[int, Dict[str, List[NodeID]]]  = {}
        # formed_qcs[view] = QuorumCert formed from votes for that view
        self.formed_qcs:     Dict[int, QuorumCert]               = {}
        # proposed_views: set of views for which this node has sent a PROPOSAL
        self.proposed_views: Set[int]                            = set()

        # ── Byzantine ─────────────────────────────────────────────────────────
        self.pending_byzantine_action: Optional[Dict[str, Any]] = None

        # ── Metrics ───────────────────────────────────────────────────────────
        self.phase_latency_history: Dict[str, Deque[float]] = {
            Phase.PREPARE.value:    deque(maxlen=50),
            Phase.PRE_COMMIT.value: deque(maxlen=50),
            Phase.COMMIT.value:     deque(maxlen=50),
            Phase.DECIDE.value:     deque(maxlen=50),
        }
        self.view_change_count:   int           = 0
        self.view_change_history: Deque[int]    = deque(maxlen=50)
        self.no_commit_streak:    int           = 0
        self.qc_miss_streak:      int           = 0
        self.total_committed_txns: int          = 0
        self.ticks_elapsed:        int          = 0
        self.recent_commit_counts: Deque[int]   = deque(maxlen=10)
        self.pipeline_utilisation: float        = 0.0

        self.equivocation_counts:   Dict[NodeID, int]        = {}
        self.message_arrival_times: Dict[NodeID, Deque[int]] = {}
        self.suspected_peers:       Dict[NodeID, bool]       = {}


# ── HotStuff Simulator ─────────────────────────────────────────────────────────

class HotStuffSimulator:
    """Coordinates all SimulatedNode objects through Chained HotStuff."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._nodes:           Dict[NodeID, SimulatedNode] = {}
        self._honest_nodes:    List[NodeID] = []
        self._byzantine_nodes: List[NodeID] = []
        self._all_nodes:       List[NodeID] = []
        self._current_tick:    int          = 0
        self._f_byzantine:     int          = 0
        self._quorum_size:     int          = 3
        self._leader_rotation: LeaderRotation = LeaderRotation.ROUND_ROBIN
        self._network_sim      = None  # set by adapter

        self._node_reputation:    Dict[NodeID, float]  = {}
        self._episode_txn_pool:   List[Transaction]    = []
        self._committed_txn_ids:  set                  = set()

    # ── Setup ──────────────────────────────────────────────────────────────────

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
        self._nodes           = nodes
        self._honest_nodes    = honest_nodes
        self._byzantine_nodes = byzantine_nodes
        self._all_nodes       = honest_nodes + byzantine_nodes
        self._f_byzantine     = f_byzantine
        n_total               = len(self._all_nodes)
        # quorum = max(ceil((n+1)/2), 2f+1)  — ensures both majority and BFT threshold
        self._quorum_size     = max(
            math.ceil((n_total + 1) / 2),
            2 * f_byzantine + 1,
        )
        self._leader_rotation = leader_rotation
        self._network_sim     = network_sim
        self._current_tick    = 0
        self._episode_txn_pool  = txn_pool
        self._committed_txn_ids = set()

        # Reputation: honest start at 1.0, byzantine at 0.01
        self._node_reputation = {nid: 1.0 for nid in honest_nodes}
        for nid in byzantine_nodes:
            self._node_reputation[nid] = 0.01

        # Bootstrap: pre-populate leader_0's new_views so it can propose at tick 0
        leader_0 = self._get_leader(0)
        if leader_0 in self._nodes:
            node_0 = self._nodes[leader_0]
            node_0.new_views_for[0] = {
                nid: GENESIS_QC for nid in self._all_nodes
            }

    # ── Leader Selection ───────────────────────────────────────────────────────

    def _get_leader(self, view: int) -> NodeID:
        pool = self._all_nodes  # Byzantine nodes are included in the rotation
        if not pool:
            return "node_0"

        if self._leader_rotation == LeaderRotation.ROUND_ROBIN:
            return pool[view % len(pool)]

        elif self._leader_rotation == LeaderRotation.VRF:
            h = int(hashlib.sha256(f"{view}".encode()).hexdigest(), 16)
            return pool[h % len(pool)]

        elif self._leader_rotation == LeaderRotation.REPUTATION_WEIGHTED:
            weights = [max(0.01, self._node_reputation.get(nid, 1.0)) for nid in pool]
            total   = sum(weights)
            norm    = [w / total for w in weights]
            r       = self._rng.random()
            cum     = 0.0
            for nid, w in zip(pool, norm):
                cum += w
                if r <= cum:
                    return nid
            return pool[-1]

        return pool[view % len(pool)]

    # ── Block creation ─────────────────────────────────────────────────────────

    def _create_block(
        self, leader_id: NodeID, view: int, justify_qc: QuorumCert
    ) -> SimBlock:
        config      = self._nodes[leader_id].config
        batch_size  = config.replication_batch_size
        uncommitted = [
            tx for tx in self._episode_txn_pool
            if tx.id not in self._committed_txn_ids
        ][:batch_size]

        parent_hash = justify_qc.block_hash
        tx_ids      = [tx.id for tx in uncommitted]
        block_hash  = hashlib.sha256(
            f"{view}:{leader_id}:{parent_hash}:{tx_ids}".encode()
        ).hexdigest()[:16]

        return SimBlock(
            slot=view,
            hash=block_hash,
            parent_hash=parent_hash,
            proposer_id=leader_id,
            transactions=uncommitted,
            justify=justify_qc,
        )

    def _create_alt_block(
        self, leader_id: NodeID, view: int, justify_qc: QuorumCert
    ) -> SimBlock:
        base_block = self._create_block(leader_id, view, justify_qc)
        alt_hash   = hashlib.sha256(f"alt:{base_block.hash}".encode()).hexdigest()[:16]
        return SimBlock(
            slot=view,
            hash=alt_hash,
            parent_hash=base_block.parent_hash,
            proposer_id=leader_id,
            transactions=base_block.transactions,
            justify=justify_qc,
        )

    # ── Messaging helpers ──────────────────────────────────────────────────────

    def _send_message(
        self,
        sender_id:   NodeID,
        target_id:   NodeID,
        msg_type:    str,
        payload:     Dict[str, Any],
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
        sender_id:   NodeID,
        msg_type:    str,
        payload:     Dict[str, Any],
        extra_delay: int = 0,
        exclude:     Optional[List[NodeID]] = None,
    ) -> None:
        for nid in self._all_nodes:
            if nid == sender_id:
                continue
            if exclude and nid in exclude:
                continue
            self._send_message(sender_id, nid, msg_type, payload, extra_delay)

    def _get_extra_delay(self, node: SimulatedNode, target_id: NodeID) -> int:
        """Return extra delay ticks from a node's pending byzantine action."""
        action = node.pending_byzantine_action
        if action is None:
            return 0
        strategy = action.get("strategy", "")
        if strategy == "random_delay":
            return action.get("extra_delay_ticks", 0)
        if strategy in ("selective_delay", "cascade_timing", "adaptive_mirror"):
            targets = action.get("targets")
            if targets is None or target_id in targets:
                return action.get("extra_delay_ticks", 0)
        return 0

    def _should_suppress(self, node: SimulatedNode, target_id: NodeID) -> bool:
        action = node.pending_byzantine_action
        if action is None:
            return False
        if action.get("strategy") in ("leader_suppress", "stall"):
            targets = action.get("targets", [])
            return target_id in targets
        return False

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self, delivered_messages: Dict[str, List[Message]]) -> None:
        """Process one full HotStuff tick for all nodes."""
        # 1. Deliver messages to inbound queues, track arrival times
        for nid, messages in delivered_messages.items():
            if nid in self._nodes:
                node = self._nodes[nid]
                for msg in messages:
                    node.inbound_queue.append(msg)
                    sender = msg.sender_id
                    if sender not in node.message_arrival_times:
                        node.message_arrival_times[sender] = deque(maxlen=20)
                    node.message_arrival_times[sender].append(self._current_tick)

        # 2. Process each node
        for nid, node in self._nodes.items():
            if node.is_byzantine:
                self._process_byzantine(nid, node)
            else:
                self._process_honest(nid, node)

        # 3. Update roles, clear queues, update metrics
        for nid, node in self._nodes.items():
            view = node.current_view
            leader = self._get_leader(view)
            node.current_role = NodeRole.LEADER if nid == leader else NodeRole.REPLICA

            node.pending_byzantine_action = None
            node.inbound_queue = []

            if node.is_byzantine:
                continue

            node.ticks_elapsed += 1
            # Per-tick commit count for TPS window (0 if no commit this tick)
            # We track whether any commit happened via no_commit_streak logic below.
            # Append 0 as default; _commit_chain_up_to will have updated total_committed_txns.
            # We compute delta by looking at committed count changes.

        # Advance global tick counter
        self._current_tick += 1

        # Per honest node: append to recent_commit_counts (delta since last tick)
        for nid, node in self._nodes.items():
            if node.is_byzantine:
                continue
            # Use ticks_elapsed as a proxy: recent_commit_counts was appended inside
            # _commit_chain_up_to incrementally; here we ensure one entry per tick
            # by appending 0 if the deque hasn't grown since the last tick.
            # Simpler: track committed count at start of tick and append delta.
            # We do this via _tick_commit_delta which is set in _commit_chain_up_to.
            delta = getattr(node, "_tick_commit_delta", 0)
            node.recent_commit_counts.append(delta)
            node._tick_commit_delta = 0  # reset for next tick

            # no_commit_streak and qc_miss_streak
            if delta > 0:
                node.no_commit_streak = 0
                node.qc_miss_streak   = 0
            else:
                node.no_commit_streak += 1
                node.qc_miss_streak   += 1

            # Pipeline utilisation
            pipeline_depth = node.config.pipeline_depth
            node.pipeline_utilisation = min(
                1.0,
                max(0.0, 1.0 - (node.no_commit_streak / max(1, pipeline_depth * 5)))
            )

            # Suspicion
            threshold = node.config.equivocation_threshold
            for peer_id, count in node.equivocation_counts.items():
                node.suspected_peers[peer_id] = count >= threshold

    # ── Honest node processing ────────────────────────────────────────────────

    def _process_honest(self, nid: NodeID, node: SimulatedNode) -> None:
        # 1. Pacemaker — check for view timeout
        self._check_view_timeout(nid, node)

        # 2. Process inbound messages
        for msg in list(node.inbound_queue):
            if msg.msg_type == PROPOSAL:
                self._on_proposal(nid, node, msg)
            elif msg.msg_type == VOTE:
                self._on_vote(nid, node, msg)
            elif msg.msg_type == NEW_VIEW:
                self._on_new_view(nid, node, msg)
            elif msg.msg_type == SYNC_REQ:
                self._on_sync_req(nid, node, msg)
            elif msg.msg_type == SYNC_RESP:
                self._on_sync_resp(nid, node, msg)

        # 3. Try to propose if we are the leader for the current view
        self._try_propose(nid, node)

    # ── Pacemaker ─────────────────────────────────────────────────────────────

    def _check_view_timeout(self, nid: NodeID, node: SimulatedNode) -> None:
        timeout_ticks = max(1, node.config.view_timeout_ms // VIEW_TICK_MS)
        elapsed       = self._current_tick - node.view_start_tick
        if elapsed >= timeout_ticks and node.current_view not in node.new_view_sent:
            self._do_view_change(nid, node, node.current_view)

    def _do_view_change(
        self, nid: NodeID, node: SimulatedNode, from_view: int
    ) -> None:
        new_view    = from_view + 1
        next_leader = self._get_leader(new_view)

        # Send NEW_VIEW carrying our high_qc to the next leader
        self._send_message(
            sender_id=nid,
            target_id=next_leader,
            msg_type=NEW_VIEW,
            payload={
                "target_view": new_view,
                "sender_id":   nid,
                "high_qc_view": node.high_qc.view,
                "high_qc_block_hash": node.high_qc.block_hash,
                "high_qc": node.high_qc,
            },
        )

        node.new_view_sent.add(from_view)
        node.view_change_count += 1
        node.view_change_history.append(from_view)

        # Penalise the failing leader's reputation
        failing_leader = self._get_leader(from_view)
        self._node_reputation[failing_leader] = max(
            0.01,
            self._node_reputation.get(failing_leader, 1.0) - 0.2,
        )

        # Advance local view
        node.current_view    = new_view
        node.view_start_tick = self._current_tick

    # ── Proposal handling ─────────────────────────────────────────────────────

    def _on_proposal(self, nid: NodeID, node: SimulatedNode, msg: Message) -> None:
        payload    = msg.payload
        prop_view  = payload.get("view", -1)
        block_hash = payload.get("block_hash", "")
        sim_block: Optional[SimBlock] = payload.get("sim_block")
        justify_qc: Optional[QuorumCert] = payload.get("justify_qc")

        # Ignore stale proposals
        if prop_view < node.current_view:
            return

        # Verify sender is the expected leader for this view
        expected_leader = self._get_leader(prop_view)
        if msg.sender_id != expected_leader:
            node.equivocation_counts[msg.sender_id] = (
                node.equivocation_counts.get(msg.sender_id, 0) + 1
            )
            return

        # Detect fork: same view, different block already stored
        if prop_view in [b.slot for b in node.block_store.values() if b.hash != block_hash]:
            existing = next(
                (b for b in node.block_store.values() if b.slot == prop_view and b.hash != block_hash),
                None,
            )
            if existing is not None:
                # A different block for this view already exists → equivocation by leader
                node.equivocation_counts[msg.sender_id] = (
                    node.equivocation_counts.get(msg.sender_id, 0) + 1
                )
                return

        # Sync lag check
        if prop_view > node.current_view + SYNC_LAG_THRESHOLD:
            self._send_message(
                sender_id=nid,
                target_id=msg.sender_id,
                msg_type=SYNC_REQ,
                payload={"since_view": node.current_view, "requester": nid},
            )

        # Advance to proposal view if needed
        if prop_view > node.current_view:
            node.current_view    = prop_view
            node.view_start_tick = self._current_tick

        # Store block
        if sim_block is not None:
            node.block_store[sim_block.hash] = sim_block
            # Also store the justify block if present
            if justify_qc is not None and justify_qc.block is not None:
                node.block_store[justify_qc.block_hash] = justify_qc.block

            # If votes arrived before the proposal (network reordering), a pending
            # vote batch may already have hit quorum but deferred QC formation because
            # the block object was missing. Retry now that we have the block.
            if sim_block.hash not in {qc.block_hash for qc in node.formed_qcs.values()}:
                pending_voters = (
                    node.votes_for.get(prop_view, {}).get(sim_block.hash, [])
                )
                if len(pending_voters) >= self._quorum_size and prop_view not in node.formed_qcs:
                    qc = QuorumCert(
                        view=prop_view,
                        block_hash=sim_block.hash,
                        block=sim_block,
                        signers=list(pending_voters),
                    )
                    node.formed_qcs[prop_view] = qc
                    if qc.view > node.high_qc.view:
                        node.high_qc = qc
                    # Responsive advance for the next leader
                    next_view = prop_view + 1
                    if self._get_leader(next_view) == nid and node.current_view <= prop_view:
                        node.current_view    = next_view
                        node.view_start_tick = self._current_tick

        # Update high_qc
        if justify_qc is not None and justify_qc.view > node.high_qc.view:
            node.high_qc = justify_qc

        if sim_block is None:
            return

        # Safety checks
        if justify_qc is None:
            justify_qc = GENESIS_QC

        if not self._safe_node(node, sim_block, justify_qc):
            return

        # 3-chain commit rule check
        self._check_commit_rule(nid, node, sim_block)
        # 2-chain locked_qc update
        self._update_locked_qc(node, sim_block)

        # Vote if we haven't voted in this view yet
        if node.voted_view < prop_view:
            next_leader = self._get_leader(prop_view + 1)
            extra_delay = self._get_extra_delay(node, next_leader)
            self._send_message(
                sender_id=nid,
                target_id=next_leader,
                msg_type=VOTE,
                payload={
                    "vote_view":  prop_view,
                    "block_hash": sim_block.hash,
                    "voter_id":   nid,
                },
                extra_delay=extra_delay,
            )
            node.voted_view = prop_view
            # PRE_COMMIT latency = ticks from view start to vote
            node.phase_latency_history[Phase.PRE_COMMIT.value].append(
                float(self._current_tick - node.view_start_tick)
            )

        # Got valid proposal: reset view timer
        node.view_start_tick = self._current_tick

    # ── Vote handling ──────────────────────────────────────────────────────────

    def _on_vote(self, nid: NodeID, node: SimulatedNode, msg: Message) -> None:
        payload    = msg.payload
        vote_view  = payload.get("vote_view", -1)
        block_hash = payload.get("block_hash", "")
        voter_id   = payload.get("voter_id", msg.sender_id)

        # Only collect if nid is the leader for vote_view + 1
        if nid != self._get_leader(vote_view + 1):
            return

        # Skip if QC already formed for this view
        if vote_view in node.formed_qcs:
            return

        # Equivocation detection: same voter, different hash in same vote_view
        if vote_view in node.votes_for:
            for other_hash, voters in node.votes_for[vote_view].items():
                if other_hash != block_hash and voter_id in voters:
                    node.equivocation_counts[voter_id] = (
                        node.equivocation_counts.get(voter_id, 0) + 1
                    )

        # Record vote
        if vote_view not in node.votes_for:
            node.votes_for[vote_view] = {}
        if block_hash not in node.votes_for[vote_view]:
            node.votes_for[vote_view][block_hash] = []
        if voter_id not in node.votes_for[vote_view][block_hash]:
            node.votes_for[vote_view][block_hash].append(voter_id)

        voters = node.votes_for[vote_view][block_hash]
        if len(voters) >= self._quorum_size:
            # Form QC — require the block object so commit/lock rules can traverse
            # the ancestor chain. If the proposal hasn't arrived yet, skip for now;
            # _on_proposal will re-trigger QC formation when the block is stored.
            voted_block = node.block_store.get(block_hash)
            if voted_block is None:
                return
            qc = QuorumCert(
                view=vote_view,
                block_hash=block_hash,
                block=voted_block,
                signers=list(voters),
            )
            node.formed_qcs[vote_view] = qc

            # Update high_qc
            if qc.view > node.high_qc.view:
                node.high_qc = qc

            # Transactions are finalized only on 3-chain commit (_commit_chain_up_to),
            # not here at QC formation — prevents overstating progress.

            # COMMIT latency = ticks from view start
            node.phase_latency_history[Phase.COMMIT.value].append(
                float(self._current_tick - node.view_start_tick)
            )

            # Reward leader reputation
            self._node_reputation[nid] = min(
                2.0, self._node_reputation.get(nid, 1.0) + 0.1
            )

            # Responsive view advance: if this node is the leader for vote_view + 1,
            # advance immediately so _try_propose fires in the same tick rather than
            # waiting for a view timeout.
            next_view = vote_view + 1
            if self._get_leader(next_view) == nid and node.current_view <= vote_view:
                node.current_view    = next_view
                node.view_start_tick = self._current_tick

    # ── NEW_VIEW handling ──────────────────────────────────────────────────────

    def _on_new_view(self, nid: NodeID, node: SimulatedNode, msg: Message) -> None:
        payload     = msg.payload
        target_view = payload.get("target_view", -1)
        sender_id   = payload.get("sender_id", msg.sender_id)
        sender_hqc: Optional[QuorumCert] = payload.get("high_qc")

        # Only collect if nid is the leader for target_view
        if nid != self._get_leader(target_view):
            return

        if sender_hqc is None:
            sender_hqc = GENESIS_QC

        if target_view not in node.new_views_for:
            node.new_views_for[target_view] = {}
        node.new_views_for[target_view][sender_id] = sender_hqc

        # Update own high_qc if sender's is better
        if sender_hqc.view > node.high_qc.view:
            node.high_qc = sender_hqc

    # ── Sync handling ──────────────────────────────────────────────────────────

    def _on_sync_req(self, nid: NodeID, node: SimulatedNode, msg: Message) -> None:
        since_view = msg.payload.get("since_view", 0)
        requester  = msg.payload.get("requester", msg.sender_id)

        # Collect committed blocks since since_view
        blocks_to_send = [
            b for b in node.committed_log
            if b.slot >= since_view
        ]

        self._send_message(
            sender_id=nid,
            target_id=requester,
            msg_type=SYNC_RESP,
            payload={
                "blocks":  blocks_to_send,
                "high_qc": node.high_qc,
            },
        )

    def _on_sync_resp(self, nid: NodeID, node: SimulatedNode, msg: Message) -> None:
        blocks:   List[SimBlock]       = msg.payload.get("blocks", [])
        resp_hqc: Optional[QuorumCert] = msg.payload.get("high_qc")

        for sim_block in blocks:
            if not isinstance(sim_block, SimBlock):
                continue
            node.block_store[sim_block.hash] = sim_block
            if sim_block.hash not in node.committed_hashes:
                node.committed_hashes.add(sim_block.hash)
                node.committed_log.append(sim_block)
                for tx in sim_block.transactions:
                    self._committed_txn_ids.add(tx.id)
                node.total_committed_txns += len(sim_block.transactions)

        if resp_hqc is not None and resp_hqc.view > node.high_qc.view:
            node.high_qc = resp_hqc

        # Advance current_view
        if node.committed_log:
            max_slot    = max(b.slot for b in node.committed_log)
            node.current_view = max(max_slot + 1, node.current_view)

    # ── Try to Propose ─────────────────────────────────────────────────────────

    def _try_propose(self, nid: NodeID, node: SimulatedNode) -> None:
        view = node.current_view

        # Only the leader for this view proposes
        if self._get_leader(view) != nid:
            return

        # Don't propose twice for the same view
        if view in node.proposed_views:
            return

        best_qc: Optional[QuorumCert] = None

        # Path A: QC formed from votes for view-1
        prev_qc = node.formed_qcs.get(view - 1)
        if prev_qc is not None:
            best_qc = prev_qc

        # Path B: 2f+1 NEW_VIEW messages for this view
        if best_qc is None:
            nv_map = node.new_views_for.get(view, {})
            if len(nv_map) >= self._quorum_size:
                # Use the highest high_qc from senders
                best_nv_qc = max(nv_map.values(), key=lambda q: q.view)
                best_qc    = best_nv_qc

        if best_qc is None:
            return

        # Create and broadcast proposal
        sim_block = self._create_block(nid, view, best_qc)
        node.block_store[sim_block.hash] = sim_block
        node.proposed_views.add(view)

        # PREPARE latency
        node.phase_latency_history[Phase.PREPARE.value].append(
            float(self._current_tick - node.view_start_tick)
        )

        # Broadcast proposal to all other nodes
        proposal_payload = {
            "view":       view,
            "block_hash": sim_block.hash,
            "sim_block":  sim_block,
            "justify_qc": best_qc,
            "proposer_id": nid,
        }
        for target in self._all_nodes:
            if target == nid:
                continue
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=PROPOSAL,
                payload=proposal_payload,
            )

        # Leader also votes for its own proposal → sends VOTE to leader(view+1)
        next_leader = self._get_leader(view + 1)
        if next_leader != nid:
            self._send_message(
                sender_id=nid,
                target_id=next_leader,
                msg_type=VOTE,
                payload={
                    "vote_view":  view,
                    "block_hash": sim_block.hash,
                    "voter_id":   nid,
                },
            )
        else:
            # Leader is also leader(view+1) — record vote locally
            if view not in node.votes_for:
                node.votes_for[view] = {}
            if sim_block.hash not in node.votes_for[view]:
                node.votes_for[view][sim_block.hash] = []
            if nid not in node.votes_for[view][sim_block.hash]:
                node.votes_for[view][sim_block.hash].append(nid)

        node.voted_view = view

        # Apply commit/lock rules to own proposal
        self._check_commit_rule(nid, node, sim_block)
        self._update_locked_qc(node, sim_block)

        # Reward reputation for proposing
        self._node_reputation[nid] = min(
            2.0, self._node_reputation.get(nid, 1.0) + 0.1
        )

    # ── Safety predicates ──────────────────────────────────────────────────────

    def _safe_node(
        self, node: SimulatedNode, block: SimBlock, qc: QuorumCert
    ) -> bool:
        """
        Safe node predicate (HotStuff Lemma 2):
        A block is safe if its justify QC extends beyond locked_qc, OR
        if it directly extends the locked_qc block.
        """
        return (
            qc.view > node.locked_qc.view
            or block.parent_hash == node.locked_qc.block_hash
        )

    def _update_locked_qc(self, node: SimulatedNode, block: SimBlock) -> None:
        """
        2-chain locked_qc update (standard Chained HotStuff PRE-COMMIT phase):
        When block (at view V) directly extends its parent (at view V-1, i.e. consecutive),
        lock on the parent's QC — the QC that certified the parent (block.justify).
        This prevents voting for conflicting chains unless they extend past the locked block.
        """
        justify = block.justify  # QC that certifies the parent block
        if justify is None or justify.block is None:
            return
        parent_block = justify.block
        if block.slot != parent_block.slot + 1:
            return
        # Lock on the parent's QC (one level up, not two)
        if justify.view > node.locked_qc.view:
            node.locked_qc = justify

    def _check_commit_rule(
        self, nid: NodeID, node: SimulatedNode, block: SimBlock
    ) -> None:
        """
        3-chain commit rule: b3 → b2 → b1 must be consecutive views.
        If so, commit everything up to and including b1.
        """
        b3 = block
        qc3 = b3.justify
        if qc3 is None or qc3.block is None:
            return
        b2 = qc3.block
        qc2 = b2.justify
        if qc2 is None or qc2.block is None:
            return
        b1 = qc2.block

        # Check 3-chain consecutiveness
        if b3.slot == b2.slot + 1 and b2.slot == b1.slot + 1:
            self._commit_chain_up_to(nid, node, b1)

    def _commit_chain_up_to(
        self, nid: NodeID, node: SimulatedNode, tip: SimBlock
    ) -> None:
        """
        Walk ancestors back through parent_hash until hitting genesis or an
        already-committed block, then commit oldest-first.
        """
        chain: List[SimBlock] = []
        current = tip

        while current is not None and current.hash != "genesis":
            if current.hash in node.committed_hashes:
                break
            chain.append(current)
            parent = node.block_store.get(current.parent_hash)
            if parent is None or parent.hash == current.hash:
                break
            current = parent

        # Commit oldest first
        for b in reversed(chain):
            if b.hash in node.committed_hashes:
                continue
            node.committed_hashes.add(b.hash)
            node.committed_log.append(b)
            n_txns = len(b.transactions)
            node.total_committed_txns += n_txns
            # Accumulate delta for this tick
            node._tick_commit_delta = getattr(node, "_tick_commit_delta", 0) + n_txns
            # Mark txns globally committed
            for tx in b.transactions:
                self._committed_txn_ids.add(tx.id)
            # DECIDE latency
            node.phase_latency_history[Phase.DECIDE.value].append(
                float(self._current_tick - node.view_start_tick)
            )

    # ── Byzantine processing ───────────────────────────────────────────────────

    def _process_byzantine(self, nid: NodeID, node: SimulatedNode) -> None:
        # Sync Byzantine node's view to the maximum across all nodes
        max_view = max(
            (n.current_view for n in self._nodes.values()), default=0
        )
        node.current_view = max_view

        # Update role
        leader = self._get_leader(node.current_view)
        node.current_role = NodeRole.LEADER if nid == leader else NodeRole.REPLICA

        action = node.pending_byzantine_action
        strategy = action.get("strategy", "") if action else ""
        is_leader = (nid == leader)

        # No action: participate honestly
        if not action or strategy == "":
            if is_leader:
                self._byz_propose_honest(nid, node)
            else:
                self._byz_vote_honest(nid, node)
            return

        if strategy in ("leader_suppress", "stall"):
            if is_leader:
                # Suppress — do not send any proposals
                return
            else:
                # As replica, still vote to avoid being too obviously detectable
                self._byz_vote_honest(nid, node)

        elif strategy == "fork":
            if is_leader:
                self._byz_fork_proposal(nid, node, action)
            else:
                self._byz_vote_honest(nid, node)

        elif strategy == "equivocation":
            self._byz_equivocate_vote(nid, node, action)

        elif strategy in ("random_delay", "selective_delay", "cascade_timing", "adaptive_mirror"):
            if is_leader:
                self._byz_propose_with_delay(nid, node, action)
            else:
                self._byz_vote_with_delay(nid, node, action)

        elif strategy == "recovery_delay":
            self._byz_delay_new_view(nid, node, action)

        else:
            # Unknown strategy: behave honestly
            if is_leader:
                self._byz_propose_honest(nid, node)
            else:
                self._byz_vote_honest(nid, node)

    def _byz_best_qc(self, node: SimulatedNode) -> QuorumCert:
        """Return the best QC available to a Byzantine node."""
        view = node.current_view
        # Try formed QC for view-1
        prev_qc = node.formed_qcs.get(view - 1)
        if prev_qc is not None:
            return prev_qc
        # Fall back to high_qc
        return node.high_qc if node.high_qc is not None else GENESIS_QC

    def _byz_propose_honest(self, nid: NodeID, node: SimulatedNode) -> None:
        view = node.current_view
        if view in node.proposed_views:
            return
        best_qc   = self._byz_best_qc(node)
        sim_block = self._create_block(nid, view, best_qc)
        node.block_store[sim_block.hash] = sim_block
        node.proposed_views.add(view)
        for target in self._all_nodes:
            if target == nid:
                continue
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=PROPOSAL,
                payload={
                    "view":       view,
                    "block_hash": sim_block.hash,
                    "sim_block":  sim_block,
                    "justify_qc": best_qc,
                    "proposer_id": nid,
                },
            )

    def _byz_vote_honest(self, nid: NodeID, node: SimulatedNode) -> None:
        for msg in node.inbound_queue:
            if msg.msg_type == PROPOSAL:
                prop_view  = msg.payload.get("view", -1)
                block_hash = msg.payload.get("block_hash", "")
                if prop_view >= node.current_view:
                    next_leader = self._get_leader(prop_view + 1)
                    self._send_message(
                        sender_id=nid,
                        target_id=next_leader,
                        msg_type=VOTE,
                        payload={
                            "vote_view":  prop_view,
                            "block_hash": block_hash,
                            "voter_id":   nid,
                        },
                    )
                    break

    def _byz_fork_proposal(
        self, nid: NodeID, node: SimulatedNode, action: Dict[str, Any]
    ) -> None:
        """Send conflicting blocks to two partitions of honest nodes."""
        view = node.current_view
        if view in node.proposed_views:
            return

        best_qc  = self._byz_best_qc(node)
        block_a  = self._create_block(nid, view, best_qc)
        block_b  = self._create_alt_block(nid, view, best_qc)
        node.block_store[block_a.hash] = block_a
        node.block_store[block_b.hash] = block_b
        node.proposed_views.add(view)

        partition_a: List[NodeID] = action.get("partition_A", [])
        partition_b: List[NodeID] = action.get("partition_B", [])

        # Default: split honest nodes half-and-half
        if not partition_a and not partition_b:
            honest = [n for n in self._all_nodes if not self._nodes[n].is_byzantine]
            mid    = max(1, len(honest) // 2)
            partition_a = honest[:mid]
            partition_b = honest[mid:]

        for target in partition_a:
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=PROPOSAL,
                payload={
                    "view":       view,
                    "block_hash": block_a.hash,
                    "sim_block":  block_a,
                    "justify_qc": best_qc,
                    "proposer_id": nid,
                },
            )
        for target in partition_b:
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=PROPOSAL,
                payload={
                    "view":       view,
                    "block_hash": block_b.hash,
                    "sim_block":  block_b,
                    "justify_qc": best_qc,
                    "proposer_id": nid,
                },
            )

    def _byz_equivocate_vote(
        self, nid: NodeID, node: SimulatedNode, action: Dict[str, Any]
    ) -> None:
        """Send conflicting votes to two groups of targets."""
        targets_a: List[NodeID] = action.get("targets_A", [])
        targets_b: List[NodeID] = action.get("targets_B", [])

        # Find a proposal to vote on
        view = node.current_view
        proposal = next(
            (msg for msg in node.inbound_queue if msg.msg_type == PROPOSAL),
            None,
        )
        if proposal is None:
            return

        real_hash = proposal.payload.get("block_hash", "")
        fake_hash = hashlib.sha256(f"fake:{real_hash}".encode()).hexdigest()[:16]
        prop_view = proposal.payload.get("view", view)

        for target in targets_a:
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=VOTE,
                payload={
                    "vote_view":  prop_view,
                    "block_hash": real_hash,
                    "voter_id":   nid,
                },
            )
        for target in targets_b:
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=VOTE,
                payload={
                    "vote_view":  prop_view,
                    "block_hash": fake_hash,
                    "voter_id":   nid,
                },
            )

    def _byz_propose_with_delay(
        self, nid: NodeID, node: SimulatedNode, action: Dict[str, Any]
    ) -> None:
        view = node.current_view
        if view in node.proposed_views:
            return

        best_qc   = self._byz_best_qc(node)
        sim_block = self._create_block(nid, view, best_qc)
        node.block_store[sim_block.hash] = sim_block
        node.proposed_views.add(view)

        strategy  = action.get("strategy", "")
        base_delay = action.get("extra_delay_ticks", 0)
        stagger    = action.get("stagger_ticks", 1)

        for i, target in enumerate(t for t in self._all_nodes if t != nid):
            if strategy == "cascade_timing":
                delay = base_delay + i * stagger
            else:
                delay = self._get_extra_delay(node, target)
            if self._should_suppress(node, target):
                continue
            self._send_message(
                sender_id=nid,
                target_id=target,
                msg_type=PROPOSAL,
                payload={
                    "view":       view,
                    "block_hash": sim_block.hash,
                    "sim_block":  sim_block,
                    "justify_qc": best_qc,
                    "proposer_id": nid,
                },
                extra_delay=delay,
            )

    def _byz_vote_with_delay(
        self, nid: NodeID, node: SimulatedNode, action: Dict[str, Any]
    ) -> None:
        view       = node.current_view
        strategy   = action.get("strategy", "")
        base_delay = action.get("extra_delay_ticks", 0)

        for msg in node.inbound_queue:
            if msg.msg_type == PROPOSAL:
                prop_view  = msg.payload.get("view", -1)
                block_hash = msg.payload.get("block_hash", "")
                if prop_view >= view:
                    next_leader = self._get_leader(prop_view + 1)
                    if strategy == "adaptive_mirror":
                        # base_delay already encodes (vat_ms + delta_ms) // VIEW_TICK_MS ticks
                        # (converted in bfa_sim). Do NOT re-add raw vat_ms here.
                        delay = base_delay
                    else:
                        delay = base_delay
                    self._send_message(
                        sender_id=nid,
                        target_id=next_leader,
                        msg_type=VOTE,
                        payload={
                            "vote_view":  prop_view,
                            "block_hash": block_hash,
                            "voter_id":   nid,
                        },
                        extra_delay=delay,
                    )
                    break

    def _byz_delay_new_view(
        self, nid: NodeID, node: SimulatedNode, action: Dict[str, Any]
    ) -> None:
        extra = action.get("extra_delay_ticks", 0)
        view  = node.current_view
        next_leader = self._get_leader(view + 1)
        self._send_message(
            sender_id=nid,
            target_id=next_leader,
            msg_type=NEW_VIEW,
            payload={
                "target_view": view + 1,
                "sender_id":   nid,
                "high_qc":     node.high_qc,
            },
            extra_delay=extra,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_node(self, node_id: NodeID) -> Optional[SimulatedNode]:
        return self._nodes.get(node_id)

    def get_current_view(self) -> int:
        """Return median view across honest nodes."""
        if not self._honest_nodes:
            return 0
        views = sorted(
            self._nodes[nid].current_view
            for nid in self._honest_nodes
            if nid in self._nodes
        )
        if not views:
            return 0
        mid = len(views) // 2
        return views[mid]

    def get_current_tick(self) -> int:
        return self._current_tick

    def get_committed_txn_ids(self) -> set:
        return self._committed_txn_ids
