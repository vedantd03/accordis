"""RewardCalculator — reward computation aligned with Chained HotStuff semantics.

Per-step reward signals:
  liveness_cost           -1.0   every step while pending txns exist (pressure to commit)
  block_commit            +60    per block's worth of txns finalized (QC formed)
  unnecessary_view_change -150   per view change with NO Byzantine activity (spurious timeout)
  liveness_stall          -30   per step where no new txns finalized AND step > STALL_THRESHOLD
  fast_leader_recovery    +120   per view change UNDER Byzantine attack, when liveness recovers
  false_positive_avoided  +40    Byzantine attack contained without agreement violation
  pipeline_efficiency     +15    episode-end: avg pipeline utilisation > threshold
  throughput_improvement  +20    episode-end: episode TPS beats static-config baseline
  latency_improvement     +10    episode-end: view-change overhead below baseline

Design notes
────────────
- UNNECESSARY_VIEW_CHANGE and FAST_LEADER_RECOVERY are mutually exclusive by
  attack state: if f_byzantine > 0 and bfa != NONE, a view change is a necessary
  response to a Byzantine leader stall or disruption.  If the cluster is clean,
  it means the agent set view_timeout_ms too aggressively.

- block_commit is driven by finalized_txn_count delta (3-chain commit = finalized).
  A transaction is counted only when a block containing it is committed by the
  3-chain rule inside _commit_chain_up_to(), not at QC formation. This prevents
  overstating progress relative to the oracle's committed_log view.

- liveness_stall fires per step (not once per episode) to give the agent a
  continuous gradient.  STALL_THRESHOLD is set to 80 steps to account for the
  ~60 ticks needed for the first 3-chain commit after bootstrap.

- false_positive_avoided is awarded when the cluster is under a Byzantine attack
  (equivocation or fork) but the episode did NOT end with an agreement or validity
  violation — meaning the agent's equivocation_threshold and view_timeout_ms were
  tuned well enough to contain the attack without letting it cause safety failures.
"""

from __future__ import annotations

from typing import Optional

from accordis.models import (
    AccordisReward,
    AccordisState,
    BaselineComparison,
    BFAStrategy,
    LivenessResult,
    VerifierResults,
)

# ── Reward constants ───────────────────────────────────────────────────────────

LIVENESS_COST_PER_STEP  = -1.0
BLOCK_COMMIT_PER_BLOCK  = +60.0
UNNECESSARY_VIEW_CHANGE = -150.0
LIVENESS_STALL          = -30.0
FAST_LEADER_RECOVERY    = +120.0
ATTACK_CONTAINED        = +40.0    # replaces false_positive_avoided stub
PIPELINE_EFFICIENCY     = +15.0    # episode-end only
THROUGHPUT_IMPROVEMENT  = +20.0    # episode-end only
LATENCY_IMPROVEMENT     = +10.0    # episode-end only

# First commit requires 3 consecutive views; each view takes network_delay ticks.
# Allow 10 steps before declaring a stall.
STALL_THRESHOLD = 10

# Default batch size used to convert finalized_txn_count deltas → block count
DEFAULT_BATCH_SIZE = 64

# Strategies that constitute active Byzantine disruption
_DISRUPTIVE = {
    BFAStrategy.RANDOM_DELAY,
    BFAStrategy.SELECTIVE_DELAY,
    BFAStrategy.EQUIVOCATION,
    BFAStrategy.LEADER_SUPPRESS,
    BFAStrategy.CASCADE_TIMING,
    BFAStrategy.RECOVERY_DELAY,
    BFAStrategy.ADAPTIVE_MIRROR,
    BFAStrategy.FORK,
}


class RewardCalculator:
    """Computes per-step AccordisReward from AccordisState transitions and oracle output."""

    def compute(
        self,
        prev_state:       Optional[AccordisState],
        current_state:    AccordisState,
        verifier_results: VerifierResults,
        baseline:         Optional[BaselineComparison] = None,
        liveness:         Optional[LivenessResult]     = None,
        is_done:          bool                         = False,
    ) -> AccordisReward:
        reward = AccordisReward()

        under_attack = (
            current_state.f_byzantine > 0
            and current_state.bfa_strategy in _DISRUPTIVE
        )

        # ── LIVENESS_COST: pressure to commit pending transactions ─────────────
        if liveness and liveness.pending_count > 0:
            reward.liveness_cost = LIVENESS_COST_PER_STEP

        # ── finalized_txn_count delta: authoritative commit signal ────────────
        prev_finalized = prev_state.finalized_txn_count if prev_state else 0
        finalized_delta = max(0, current_state.finalized_txn_count - prev_finalized)

        # ── BLOCK_COMMIT: +60 per block's-worth of txns finalized ─────────────
        # Convert txn delta → approximate block count using each honest node's
        # configured replication_batch_size (take the max, not the average, to
        # avoid awarding fractional blocks when the batch size was just increased).
        if finalized_delta > 0:
            max_batch = max(
                (
                    ns.config.replication_batch_size
                    for ns in current_state.node_states.values()
                    if not ns.is_byzantine
                ),
                default=DEFAULT_BATCH_SIZE,
            )
            new_blocks = max(1, round(finalized_delta / max(1, max_batch)))
            reward.block_commit = BLOCK_COMMIT_PER_BLOCK * new_blocks

        # ── VIEW CHANGE HANDLING ───────────────────────────────────────────────
        # UNNECESSARY vs FAST_LEADER_RECOVERY are mutually exclusive.
        vc_delta = 0
        if prev_state is not None:
            vc_delta = max(0, current_state.view_change_count - prev_state.view_change_count)

        if vc_delta > 0:
            if under_attack:
                # View change is a NECESSARY response to Byzantine leader stall / disruption.
                # Reward it proportionally to liveness recovery: the agent's view_timeout_ms
                # was calibrated well if liveness_rate recovered after the change.
                if liveness and liveness.liveness_rate > 0.0:
                    reward.fast_leader_recovery = FAST_LEADER_RECOVERY * vc_delta
                else:
                    # Attack ongoing and liveness still zero — partial credit
                    reward.fast_leader_recovery = (FAST_LEADER_RECOVERY * 0.3) * vc_delta
            else:
                # No Byzantine activity; the view change was spurious.
                # Cause: view_timeout_ms set too aggressively short.
                # Only penalise if agreement was maintained (the node wasn't
                # protecting itself from an actual safety threat).
                if verifier_results.agreement.passed:
                    reward.unnecessary_view_change = UNNECESSARY_VIEW_CHANGE * vc_delta

        # ── LIVENESS_STALL: per-step penalty for zero progress ────────────────
        # Fire only when all three conditions hold:
        #   1. Past bootstrap window (first 3-chain commit takes ~5 steps now)
        #   2. No new txns finalized this step
        #   3. There are still pending transactions — avoids penalising post-drain states
        has_pending = liveness is not None and liveness.pending_count > 0
        if current_state.step > STALL_THRESHOLD and finalized_delta == 0 and has_pending:
            reward.liveness_stall = LIVENESS_STALL

        # ── Episode-end components ─────────────────────────────────────────────
        if is_done:
            # ── ATTACK_CONTAINED (false_positive_avoided field) ───────────────────
            # Award when the cluster was under a disruptive Byzantine attack AND:
            #   1. Agreement was maintained (no fork succeeded)
            #   2. Validity was maintained (no invalid txn committed)
            # Both safety dimensions must hold; one violated means the attack succeeded.
            if (
                under_attack
                and verifier_results.agreement.passed
                and verifier_results.validity.passed
            ):
                reward.false_positive_avoided = ATTACK_CONTAINED

            # PIPELINE_EFFICIENCY: reward high average pipeline utilisation
            honest_count = sum(
                1 for ns in current_state.node_states.values() if not ns.is_byzantine
            )
            if honest_count > 0:
                # Use liveness_rate as a proxy: high rate → pipeline kept busy
                rate = liveness.liveness_rate if liveness else 0.0
                if rate > 0.8:
                    reward.pipeline_efficiency = PIPELINE_EFFICIENCY
                elif rate > 0.5:
                    reward.pipeline_efficiency = PIPELINE_EFFICIENCY * 0.5

            # THROUGHPUT_IMPROVEMENT: episode TPS > static-config baseline
            if baseline and baseline.relative_tps_improvement > 0:
                reward.throughput_improvement = THROUGHPUT_IMPROVEMENT

            # LATENCY_IMPROVEMENT: view-change overhead below baseline
            if baseline and baseline.relative_latency_improvement > 0:
                reward.latency_improvement = LATENCY_IMPROVEMENT

        # ── Total ──────────────────────────────────────────────────────────────
        reward.total = (
            reward.liveness_cost
            + reward.block_commit
            + reward.unnecessary_view_change
            + reward.liveness_stall
            + reward.fast_leader_recovery
            + reward.false_positive_avoided
            + reward.pipeline_efficiency
            + reward.throughput_improvement
            + reward.latency_improvement
        )

        return reward
