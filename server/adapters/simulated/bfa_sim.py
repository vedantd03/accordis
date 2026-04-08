"""ByzantineInjector — translates BFAStrategy values into node-level pending_byzantine_action dicts.

Called by SimulatedConsensusAdapter.inject_byzantine_action() and stores the instruction
on the target node's pending_byzantine_action field. The instruction is consumed and
cleared during the next advance_one_step() tick.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from accordis.models import BFAStrategy, NodeID

# Must match VIEW_TICK_MS in hotstuff_sim — used to convert ms delay parameters to ticks
VIEW_TICK_MS: int = 50


class ByzantineInjector:
    """Translates BFAStrategy enum values into pending_byzantine_action dicts.

    Each strategy maps to a dict that the HotStuffSimulator reads during advance_one_step()
    to alter the Byzantine node's message-sending behavior for exactly that tick.
    """

    def build_action(
        self,
        strategy: BFAStrategy,
        target_nodes: List[NodeID],
        parameters: dict,
        byz_index: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Return a pending_byzantine_action dict for the given strategy, or None for NONE.

        Args:
            strategy:     The BFAStrategy to apply.
            target_nodes: Honest node IDs eligible to be targeted.
            parameters:   Strategy-specific parameters from ByzantineFailureAgent.
            byz_index:    Index of this Byzantine node in the Byzantine node list
                          (used by CASCADE_TIMING to stagger delays across nodes).
        """
        if strategy == BFAStrategy.NONE:
            return None

        if strategy == BFAStrategy.RANDOM_DELAY:
            delay_ms = parameters.get("delay_ms", 100)
            return {
                "strategy":          "random_delay",
                "extra_delay_ticks": max(1, delay_ms // VIEW_TICK_MS),
                "targets":           None,  # applies to all outbound messages
            }

        if strategy == BFAStrategy.SELECTIVE_DELAY:
            delay_ms = parameters.get("delay_ms", 100)
            targets  = parameters.get("targets", target_nodes)
            return {
                "strategy":          "selective_delay",
                "extra_delay_ticks": max(1, delay_ms // VIEW_TICK_MS),
                "targets":           targets,
            }

        if strategy == BFAStrategy.EQUIVOCATION:
            mid       = max(1, len(target_nodes) // 2)
            targets_a = parameters.get("targets_A", target_nodes[:mid])
            targets_b = parameters.get("targets_B", target_nodes[mid:])
            return {
                "strategy":  "equivocation",
                "targets_A": targets_a,
                "targets_B": targets_b,
            }

        if strategy == BFAStrategy.LEADER_SUPPRESS:
            targets = parameters.get("targets", target_nodes)
            return {
                "strategy": "leader_suppress",
                "targets":  targets,
            }

        if strategy == BFAStrategy.CASCADE_TIMING:
            # Each Byzantine node i delays by base + i*stagger_ticks,
            # creating a coordinated cascade effect across Byzantine nodes.
            base_ms  = parameters.get("delay_ms", 200)
            stagger  = max(1, parameters.get("stagger_ticks", 10) // VIEW_TICK_MS)
            base     = max(1, base_ms // VIEW_TICK_MS)
            return {
                "strategy":          "cascade_timing",
                "extra_delay_ticks": base + byz_index * stagger,
                "stagger_ticks":     stagger,
                "targets":           None,  # applies to all outbound messages
            }

        if strategy == BFAStrategy.RECOVERY_DELAY:
            delay_ms = parameters.get("delay_ms", 150)
            return {
                "strategy":          "recovery_delay",
                "extra_delay_ticks": max(1, delay_ms // VIEW_TICK_MS),
            }

        if strategy == BFAStrategy.ADAPTIVE_MIRROR:
            # Time delays to land just after the honest node's vote aggregation window closes.
            # delay = vote_aggregation_timeout_ms + delta_ms, converted to ticks
            vat_ms   = parameters.get("vote_aggregation_timeout_ms", 2000)
            delta_ms = parameters.get("delta_ms", 50)
            return {
                "strategy":          "adaptive_mirror",
                "extra_delay_ticks": max(1, (vat_ms + delta_ms) // VIEW_TICK_MS),
                "targets":           None,  # applies to all outbound messages
            }

        if strategy == BFAStrategy.FORK:
            # Byzantine leader sends different blocks to each partition of honest nodes.
            mid         = max(1, len(target_nodes) // 2)
            partition_a = parameters.get("partition_A", target_nodes[:mid])
            partition_b = parameters.get("partition_B", target_nodes[mid:])
            return {
                "strategy":    "fork",
                "partition_A": partition_a,
                "partition_B": partition_b,
            }

        return None
