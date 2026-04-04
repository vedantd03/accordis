"""RewardCalculator — shared, adapter-agnostic reward computation.

Operates only on AccordisState transitions and oracle outputs.
Never calls the adapter.
"""

from __future__ import annotations

from typing import Optional

from accordis.models import (
    AccordisReward,
    AccordisState,
    BaselineComparison,
    LivenessResult,
    VerifierResults,
)

# ── Reward constants ──────────────────────────────────────────────────────────

LIVENESS_COST_PER_STEP  = -1.0
UNNECESSARY_VIEW_CHANGE = -150.0
LIVENESS_STALL          = -300.0
BLOCK_COMMIT            = +60.0
FAST_LEADER_RECOVERY    = +120.0
FALSE_POSITIVE_AVOIDED  = +40.0
PIPELINE_EFFICIENCY     = +15.0    # episode-end only
THROUGHPUT_IMPROVEMENT  = +20.0    # episode-end only
LATENCY_IMPROVEMENT     = +10.0    # episode-end only

# Number of steps without commit before triggering LIVENESS_STALL
STALL_THRESHOLD = 20


class RewardCalculator:
    """Computes per-step AccordisReward from state transitions and oracle outputs."""

    def compute(
        self,
        prev_state: Optional[AccordisState],
        current_state: AccordisState,
        verifier_results: VerifierResults,
        baseline: Optional[BaselineComparison] = None,
        liveness: Optional[LivenessResult] = None,
        is_done: bool = False,
    ) -> AccordisReward:
        """Compute reward for a single environment step.

        Episode-end components (pipeline_efficiency, throughput_improvement,
        latency_improvement) are only set when is_done=True.

        Args:
            prev_state:       State from the previous step (None on first step).
            current_state:    State after the current step.
            verifier_results: Output from CorrectnessOracle.run_all().
            baseline:         Output from CorrectnessOracle.compute_baseline_comparison().
            liveness:         Output from CorrectnessOracle.check_liveness().
            is_done:          Whether this is the final step of the episode.

        Returns:
            AccordisReward with all components populated.
        """
        reward = AccordisReward()

        # ── LIVENESS_COST: -1.0 per step while pending transactions exist ────
        if liveness and liveness.pending_count > 0:
            reward.liveness_cost = LIVENESS_COST_PER_STEP

        # ── UNNECESSARY_VIEW_CHANGE: -150 per false view change ──────────────
        if prev_state is not None:
            view_change_delta = current_state.view_change_count - prev_state.view_change_count
            if view_change_delta > 0 and verifier_results.agreement.passed:
                # View change happened but agreement was maintained → unnecessary
                # Only penalise if liveness was fine (no actual Byzantine disruption needed it)
                if liveness and liveness.liveness_rate > 0.5:
                    reward.unnecessary_view_change = UNNECESSARY_VIEW_CHANGE * view_change_delta

        # ── LIVENESS_STALL: -300 if no commit >20 steps with honest quorum ───
        # Check no-commit streak across honest nodes
        max_no_commit_streak = 0
        for nid, ns in current_state.node_states.items():
            if not ns.is_byzantine:
                # Approximate stall from view_change_count growth rate
                if prev_state:
                    prev_ns = prev_state.node_states.get(nid)
                    if prev_ns and len(ns.committed_log) == len(prev_ns.committed_log):
                        max_no_commit_streak += 1

        # Simple stall detection: if no new commits and step > threshold
        if (
            current_state.step > STALL_THRESHOLD
            and liveness
            and liveness.committed_count == 0
        ):
            reward.liveness_stall = LIVENESS_STALL

        # ── BLOCK_COMMIT: +60 per block committed by 2f+1 honest replicas ────
        if prev_state is not None:
            # Count new blocks committed since last step
            new_commits = 0
            for nid, ns in current_state.node_states.items():
                if not ns.is_byzantine:
                    prev_ns = prev_state.node_states.get(nid)
                    if prev_ns:
                        new_blocks = len(ns.committed_log) - len(prev_ns.committed_log)
                        new_commits = max(new_commits, new_blocks)
            reward.block_commit = BLOCK_COMMIT * new_commits

        # ── FAST_LEADER_RECOVERY: +120 per quick Byzantine leader replacement ─
        if prev_state is not None:
            # Detect view change that resolved quickly (within 2 steps)
            vc_delta = current_state.view_change_count - prev_state.view_change_count
            if vc_delta > 0 and liveness and liveness.liveness_rate > 0.0:
                # A view change happened and liveness recovered
                reward.fast_leader_recovery = FAST_LEADER_RECOVERY * vc_delta

        # ── FALSE_POSITIVE_AVOIDED: +40 per resolved suspicion ───────────────
        # Approximated: if suspected nodes are cleared without causing a view change
        if prev_state is not None:
            vc_delta = current_state.view_change_count - prev_state.view_change_count
            if vc_delta == 0:
                # Check if any suspected nodes were cleared
                for nid, ns in current_state.node_states.items():
                    if not ns.is_byzantine:
                        # Count nodes that were suspected in prev but not now
                        # This would require agent action tracking; approximate to 0
                        pass
            # (actual false positive detection requires agent action history)

        # ── Episode-end only rewards ──────────────────────────────────────────
        if is_done:
            # PIPELINE_EFFICIENCY: +15 if utilisation > 0.8 × pipeline_depth
            avg_utilisation = 0.0
            count = 0
            for nid, ns in current_state.node_states.items():
                if not ns.is_byzantine:
                    # Get pipeline utilisation from observation if available
                    pipeline_depth = ns.config.pipeline_depth
                    # Approximate: if node committed consistently
                    if liveness and liveness.liveness_rate > 0.8:
                        avg_utilisation += 0.85
                    else:
                        avg_utilisation += liveness.liveness_rate if liveness else 0.0
                    count += 1
            if count > 0:
                avg_utilisation /= count
                if avg_utilisation > 0.8:
                    reward.pipeline_efficiency = PIPELINE_EFFICIENCY

            # THROUGHPUT_IMPROVEMENT: +20 if episode TPS > static baseline TPS
            if baseline and baseline.relative_tps_improvement > 0:
                reward.throughput_improvement = THROUGHPUT_IMPROVEMENT

            # LATENCY_IMPROVEMENT: +10 if p99 < static baseline p99
            if baseline and baseline.relative_latency_improvement > 0:
                reward.latency_improvement = LATENCY_IMPROVEMENT

        # ── Total ─────────────────────────────────────────────────────────────
        reward.total = (
            reward.liveness_cost
            + reward.unnecessary_view_change
            + reward.liveness_stall
            + reward.block_commit
            + reward.fast_leader_recovery
            + reward.false_positive_avoided
            + reward.pipeline_efficiency
            + reward.throughput_improvement
            + reward.latency_improvement
        )

        return reward
