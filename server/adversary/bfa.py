"""ByzantineFailureAgent — shared, adapter-agnostic adversary component.

Operates entirely on BFAStrategy enums and BFTConfig values.
Has zero knowledge of how strategies are implemented at the adapter level.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import BFAStrategy, BFTConfig, NodeID


STRATEGY_POOL: Dict[int, List[BFAStrategy]] = {
    1: [BFAStrategy.NONE],
    2: [BFAStrategy.RANDOM_DELAY],
    3: [BFAStrategy.SELECTIVE_DELAY],
    4: [BFAStrategy.EQUIVOCATION],
    5: [BFAStrategy.ADAPTIVE_MIRROR],
    6: [BFAStrategy.LEADER_SUPPRESS, BFAStrategy.SELECTIVE_DELAY],
    7: [BFAStrategy.CASCADE_TIMING, BFAStrategy.EQUIVOCATION],
    8: list(BFAStrategy),
}


class ByzantineFailureAgent:
    """Selects Byzantine disruption strategies per step.

    Shared between both adapter versions. The strategy selection is deterministic
    given the same seed, ensuring reproducible episodes.
    """

    def __init__(self) -> None:
        self._rng = random.Random(42)

    def select_strategy(
        self,
        curriculum_level: int,
        step: int = 0,
        agent_configs: Optional[Dict[NodeID, BFTConfig]] = None,
        seed: Optional[int] = None,
    ) -> BFAStrategy:
        """Sample a strategy from STRATEGY_POOL[curriculum_level] using seeded PRNG.

        Args:
            curriculum_level: Current difficulty level (1–8).
            step:             Current episode step (used to vary strategy over time).
            agent_configs:    Current BFTConfig per node (for ADAPTIVE_MIRROR delta).
            seed:             Optional seed override. If provided, reseeds the PRNG.

        Returns:
            BFAStrategy to use for this step.
        """
        if seed is not None:
            self._rng = random.Random(seed + step)

        level = max(1, min(8, curriculum_level))
        pool = STRATEGY_POOL.get(level, [BFAStrategy.NONE])
        return self._rng.choice(pool)

    def get_disruption_parameters(
        self,
        strategy: BFAStrategy,
        agent_configs: Optional[Dict[NodeID, BFTConfig]] = None,
        target_nodes: Optional[List[NodeID]] = None,
    ) -> dict:
        """Return the parameters dict for inject_byzantine_action().

        The returned dict matches the strategy-to-parameter mapping in BaseConsensusAdapter.

        Args:
            strategy:     The BFAStrategy to parameterise.
            agent_configs: Current BFTConfig per honest node (used for ADAPTIVE_MIRROR).
            target_nodes:  List of honest node IDs to target (used for SELECTIVE_DELAY, etc.).

        Returns:
            parameters dict ready to pass to inject_byzantine_action().
        """
        targets = target_nodes or []
        configs = agent_configs or {}

        if strategy == BFAStrategy.NONE:
            return {}

        if strategy == BFAStrategy.RANDOM_DELAY:
            return {"delay_ms": self._rng.randint(100, 500)}

        if strategy == BFAStrategy.SELECTIVE_DELAY:
            selected = targets[:max(1, len(targets) // 2)] if targets else []
            return {"delay_ms": self._rng.randint(200, 800), "targets": selected}

        if strategy == BFAStrategy.EQUIVOCATION:
            mid = max(1, len(targets) // 2)
            return {
                "targets_A": targets[:mid],
                "targets_B": targets[mid:],
            }

        if strategy == BFAStrategy.LEADER_SUPPRESS:
            return {"targets": targets}

        if strategy == BFAStrategy.CASCADE_TIMING:
            return {"delay_ms": self._rng.randint(300, 1000)}

        if strategy == BFAStrategy.RECOVERY_DELAY:
            return {"delay_ms": self._rng.randint(500, 2000)}

        if strategy == BFAStrategy.ADAPTIVE_MIRROR:
            # Compute delta from min view_timeout_ms across honest nodes
            if configs:
                min_vt = min(c.view_timeout_ms for c in configs.values())
            else:
                min_vt = 2000
            delta = self._rng.randint(50, 200)
            return {"view_timeout_ms": min_vt, "delta_ms": delta}

        return {}
