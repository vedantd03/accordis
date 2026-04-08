# HotStuff Simulator — Correctness Status

This document tracks the protocol-correctness gaps that were identified
between the original `HotStuffSimulator` and the HotStuff paper, and the
state of each fix as of the current branch.

The simulator now implements **Chained HotStuff** with per-node pacemaker,
3-chain commit, fork detection, and a sync protocol. All timing-sensitive
parameters are converted between ms and ticks via a single shared constant
`VIEW_TICK_MS = 50` defined in [hotstuff_sim.py](../../../server/adapters/simulated/hotstuff_sim.py).

---

## 1. Protocol Correctness — Status

### A. Single-round commit (no 3-chain rule) — **FIXED**
`_check_commit_rule` walks `b3 → b2 → b1` via each block's `justify` QC and
commits the chain only when all three are at consecutive views.
`_commit_chain_up_to` walks ancestors via `parent_hash`, commits oldest
first, and updates `total_committed_txns` plus the global
`_committed_txn_ids` set. Transactions are no longer counted at QC
formation — only at 3-chain commit, aligning with the oracle's view of
finality.

### B. No `highQC` / `lockedQC` on each node — **FIXED**
`SimulatedNode.high_qc` and `SimulatedNode.locked_qc` are first-class
fields, both initialised to `GENESIS_QC`. `_update_locked_qc` advances
`locked_qc` when a block directly extends its parent (the standard
2-chain Chained-HotStuff PRE-COMMIT progression).

### C. Leader proposes every tick regardless — **FIXED**
`_try_propose` only fires when one of two conditions holds:
1. A QC was formed for `view-1` from collected votes (path A).
2. The leader has received `≥ 2f+1` `NEW_VIEW` messages and uses the
   highest `high_qc` among them (path B).
A leader for the bootstrap view (view 0) is pre-populated with synthetic
`NEW_VIEW`s in `setup()` so the very first proposal can fire.

### D. No block chaining / parent hash — **FIXED**
`SimBlock` carries `parent_hash` and `justify: QuorumCert`. `Block`
(Pydantic) also exposes `parent_hash` for round-tripping into the oracle.

### I. Voting is unconditional — **FIXED**
`_safe_node` enforces the HotStuff Lemma 2 predicate: a vote is cast only
if `qc.view > locked_qc.view` OR the proposed block directly extends
`locked_qc.block_hash`. `voted_view` prevents double-voting per view.

---

## 2. Leader Election & View Change — Status

### E. Byzantine nodes excluded from leader pool — **FIXED**
`_get_leader` draws from `self._all_nodes`, so Byzantine nodes can be
elected and the LEADER_SUPPRESS / FORK / stall attacks are now reachable.

### F. View change is leader-triggered and centralised — **FIXED**
Per-node pacemaker (`_check_view_timeout`) runs every tick on every honest
node. When `current_tick - view_start_tick >= view_timeout_ms // VIEW_TICK_MS`,
the node sends `NEW_VIEW` independently. There is no central view counter
on the simulator anymore.

### G. NEW_VIEW doesn't carry highQC — **FIXED**
`_do_view_change` packs the sender's `high_qc` into the NEW_VIEW payload.
`_on_new_view` updates the recipient leader's `high_qc` if the sender's
is fresher.

### H. No 2f+1 NEW_VIEW collection before leader acts — **FIXED**
Path B in `_try_propose` requires `len(nv_map) >= self._quorum_size` and
selects the highest QC across the collected NEW_VIEWs as `best_qc`.

---

## 3. Safety Model — Status

### J. View synchronisation is global and centralised — **FIXED**
Each `SimulatedNode` carries its own `current_view`. The simulator-wide
`get_current_view()` returns the median across honest nodes for observation
purposes only — there is no longer a single source of truth that all
nodes advance from in lockstep.

### K. No message authentication model — **NOT FIXED (intentional)**
Messages are still unsigned. Equivocation detection is now bidirectional
(both proposal-equivocation and vote-equivocation are tracked) and is
strong enough for the agent's `equivocation_threshold` field to work.
Threshold signatures are out of scope for the simulator.

---

## 4. Byzantine Fault Model — Status

### L. Byzantine leader behaviour unmodelled — **FIXED**
- **Pure stall** — `_byz_propose_with_delay` with strategy `stall` /
  `leader_suppress` simply does not emit a proposal. Honest pacemaker times
  out on its own.
- **Partition fork** — `_byz_fork_proposal` builds a base block plus an
  `_create_alt_block` (different hash, same view) and sends each to a
  configurable partition of honest nodes. Default split is honest // 2.
- **Invalid ancestor** — already covered by the safety predicate; honest
  nodes refuse to vote when `_safe_node` returns False.

### M. No coordinated multi-Byzantine attacks — **PARTIAL**
`CASCADE_TIMING` now staggers delays across Byzantine nodes via the
`byz_index` parameter threaded from the adapter through `bfa_sim.py` into
`hotstuff_sim.py`. Multi-node fork coordination (two Byzantine nodes
splitting the honest pool simultaneously) is still single-actor; building
true cross-node BFA coordination remains future work.

### N. CASCADE_TIMING and ADAPTIVE_MIRROR are identical — **FIXED**
- `CASCADE_TIMING`: stagger = `base + i * stagger_ticks` per Byzantine node.
- `ADAPTIVE_MIRROR`: delay = `(vote_aggregation_timeout_ms + delta_ms) / VIEW_TICK_MS`,
  computed once in `bfa_sim.py` so the simulator doesn't double-add the
  raw vat_ms inside `_byz_vote_with_delay`.

### O. Equivocation only in voting — **PARTIAL**
- Vote equivocation: `_on_vote` flags any voter that signs two different
  hashes in the same view.
- Proposal equivocation: `_on_proposal` flags a leader that produces two
  different blocks for the same view (used by the FORK attack).
- Replayed historical messages: still not modelled.

---

## 5. Simulation Realism — Status

### P. All 4 phases happen in 1 tick — **FIXED**
Phases now span multiple ticks because every message is enqueued through
`NetworkSimulator` and dispatched on its `delivery_tick`. PROPOSAL → VOTE
→ QC formation → next-view PROPOSAL is multi-tick by construction; the
3-chain commit rule then requires three consecutive views' worth of ticks
before any block finalises.

### Q. Phase latency values are tick timestamps, not durations — **FIXED**
All four phase latency appends use `float(self._current_tick - node.view_start_tick)` —
a duration anchored at the start of the current view.

### R. Pipelined HotStuff not actually pipelined — **PARTIAL**
A proposal can fire in the same tick a QC for `view-1` is formed
(responsive view advance in `_on_vote` / `_on_proposal`). This gives one
level of pipelining but does not yet model `pipeline_depth > 1` as
true overlapping in-flight blocks. Higher pipeline depths still produce
the expected reward signal because `pipeline_utilisation` is derived from
`no_commit_streak` decay, but the deeper-pipeline speedup is not literal.

### S. No replica state transfer / catch-up — **FIXED**
`SYNC_REQ` / `SYNC_RESP` messages are exchanged when a replica observes a
proposal more than `SYNC_LAG_THRESHOLD = 3` views ahead. The responder
sends every committed block since `since_view`. The requester admits the
new blocks, advances its `current_view`, and may bump its `high_qc`.

---

## 6. Tick / Timing Model

A **single explicit tick = `VIEW_TICK_MS = 50` ms** of simulated wall
clock. The constant lives in `hotstuff_sim.py` and is mirrored (with the
same value) in `network_sim.py` and `bfa_sim.py` so that:

| Where                        | What gets converted              | Direction        |
|------------------------------|----------------------------------|------------------|
| `_check_view_timeout`        | `view_timeout_ms` → ticks        | floor div by 50  |
| `NetworkSimulator.sample_*`  | sampled latency (ms) → ticks     | round / 50       |
| `bfa_sim.build_action`       | every `*_ms` parameter → ticks   | floor div by 50  |
| `read_observation`           | view-stuck ticks → `view_stuck_ms` | multiply by 50 |

This is the single seam between the agent's ms-denominated knobs and the
internal tick scheduler. Any future change to tick granularity must
update **all five** call sites.

### Episode budget vs. timeout (the Medium-Task failure mode)

Tasks have explicit step budgets:

| Task   | max_steps | wall-clock budget |
|--------|-----------|-------------------|
| Easy   | 50        | 2500 ms           |
| Medium | 100       | 5000 ms           |
| Hard   | 100       | 5000 ms           |

If `view_timeout_ms` is set to a value ≥ the episode budget, no
view-change can fire inside the episode and a Byzantine leader stall
deadlocks the cluster (this was the Gemini Flash-Lite Medium-Task
failure: `view_timeout_ms = 5000`, `qc_miss_streak ≈ 71`, view stuck at
5).

This is now structurally prevented by:

1. **`SAFE_BFT_TUNING_BOUNDS["view_timeout_ms"] = (200, 3000)`** in
   [models.py](../../../models.py). The agent literally cannot configure
   a timeout ≥ 60 ticks; at least one view change is always reachable
   inside a 100-step budget.
2. **`view_stuck_ms` observation field** ([models.py](../../../models.py),
   surfaced from [adapter.py](../../../server/adapters/simulated/adapter.py)).
   Reports the ms-equivalent time the node has been waiting in its
   current view, in the **same unit** as `view_timeout_ms` so the agent
   can compare the two without knowing about ticks. Plumbed into the
   per-node JSON summary in [inference.py](../../../inference.py).
3. **SYSTEM_PROMPT** ([constants.py](../../../server/utils/constants.py))
   now spells out: "1 step ≈ 50 ms", episode budgets per task, and the
   rule "if `view_stuck_ms` rises with `qc_miss_streak`, lower
   `view_timeout_ms` immediately".

### Why the same fix protects Hard

Hard task uses `n=10`, `f=3`, `LEADER_SUPPRESS` + `CASCADE_TIMING`, same
100-step budget. With 30% Byzantine in round-robin rotation, ~3 of every
10 leaders stall, so the agent **must** allow several view changes per
episode for the pool to drain. The same `view_timeout_ms = 5000ms` policy
that broke Medium would break Hard even harder — capping the bound at
3000ms and exposing `view_stuck_ms` are the same fix.

---

## 7. Reward / Environment Wiring

- `RewardCalculator` ([reward_calculator.py](../../../server/rewards/reward_calculator.py))
  derives `block_commit` from `finalized_txn_count` delta (3-chain) and
  splits view changes into `unnecessary_view_change` (no attack) vs
  `fast_leader_recovery` (attack present) — mutually exclusive by attack
  state.
- `AccordisEnvironment` ([accordis_environment.py](../../../server/accordis_environment.py))
  pulls `_state.view_change_count` from `adapter.get_cumulative_view_changes()`
  so the reward delta never saturates at the 50-deep history window.
- `BaseConsensusAdapter` ([base.py](../../../server/adapters/base.py))
  mandates `view_stuck_ms` in the observation dict and `get_cumulative_view_changes()`
  on the adapter contract.

---

## 8. Remaining Gaps (Not Yet Fixed)

| Gap | Description                                            | Why deferred                           |
|-----|--------------------------------------------------------|----------------------------------------|
| K   | Threshold-signature message authentication             | Out of scope for in-process simulator  |
| M   | Cross-node Byzantine collusion (true coordinated fork) | Requires shared BFA state across nodes |
| O.3 | Replayed-message attacks                               | Low expected reward signal             |
| R.2 | True overlapping pipeline at `pipeline_depth > 1`      | Reward proxy already drives policy     |
