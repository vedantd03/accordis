"""ByzantineInjector — translates BFAStrategy values into node-level pending_byzantine_action dicts.

Called by SimulatedConsensusAdapter.inject_byzantine_action() and stores the instruction
on the target node's pending_byzantine_action field. The instruction is consumed and
cleared during the next advance_one_step() tick.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models import BFAStrategy, NodeID


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
    ) -> Optional[Dict[str, Any]]:
        """Return a pending_byzantine_action dict for the given strategy, or None for NONE."""
        if strategy == BFAStrategy.NONE:
            return None

        if strategy == BFAStrategy.RANDOM_DELAY:
            return {
                "strategy": "random_delay",
                "extra_delay_ticks": parameters.get("delay_ms", 100),
                "targets": None,  # applies to all outbound messages
            }

        if strategy == BFAStrategy.SELECTIVE_DELAY:
            return {
                "strategy": "selective_delay",
                "extra_delay_ticks": parameters.get("delay_ms", 100),
                "targets": parameters.get("targets", target_nodes),
            }

        if strategy == BFAStrategy.EQUIVOCATION:
            return {
                "strategy": "equivocation",
                "targets_A": parameters.get("targets_A", []),
                "targets_B": parameters.get("targets_B", []),
            }

        if strategy == BFAStrategy.LEADER_SUPPRESS:
            return {
                "strategy": "leader_suppress",
                "targets": parameters.get("targets", target_nodes),
            }

        if strategy == BFAStrategy.CASCADE_TIMING:
            # Functionally identical to SELECTIVE_DELAY applied to all Byzantine nodes
            return {
                "strategy": "selective_delay",
                "extra_delay_ticks": parameters.get("delay_ms", 200),
                "targets": None,  # applies to all outbound messages from this node
            }

        if strategy == BFAStrategy.RECOVERY_DELAY:
            return {
                "strategy": "recovery_delay",
                "extra_delay_ticks": parameters.get("delay_ms", 150),
            }

        if strategy == BFAStrategy.ADAPTIVE_MIRROR:
            vt = parameters.get("view_timeout_ms", 2000)
            delta = parameters.get("delta_ms", 50)
            actual_delay = max(1, vt - delta)
            return {
                "strategy": "selective_delay",
                "extra_delay_ticks": actual_delay,
                "targets": None,  # all non-byzantine, handled by the sim
            }

        return None
