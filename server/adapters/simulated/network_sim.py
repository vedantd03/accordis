"""NetworkSimulator — synthetic Pareto latency model for the HotStuff simulation.

All random sampling uses a seeded PRNG initialised at reset().
The same seed always produces the same delivery sequence for the same fault profile.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from accordis.server.network.fault_profiles import FaultProfile, get_fault_profile

# Must match VIEW_TICK_MS in hotstuff_sim — 1 logical tick = this many ms
VIEW_TICK_MS: int = 50


@dataclass
class PendingMessage:
    delivery_tick: int
    target_node_id: str
    message: Any

    def __lt__(self, other: "PendingMessage") -> bool:
        return self.delivery_tick < other.delivery_tick


class NetworkSimulator:
    """Applies a FaultProfile to every message dispatched in the simulation.

    Does not use real sockets or OS networking. Latency is a per-message integer
    representing the delivery tick, not a wall-clock measurement.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._profile: FaultProfile = get_fault_profile(1)
        self._pending: List[PendingMessage] = []
        self._current_tick: int = 0

    def set_profile(self, profile: FaultProfile) -> None:
        """Replace the active FaultProfile. Already-queued messages retain their delivery_tick."""
        self._profile = profile

    def reset(self, seed: int, profile: FaultProfile) -> None:
        """Full reset — new seed, new profile, empty queue."""
        self._rng = random.Random(seed)
        self._profile = profile
        self._pending = []
        self._current_tick = 0

    def _pareto_delay_ticks(self) -> int:
        """Compute base_delay_ticks using Pareto distribution.

        We derive shape α from the p50/p99 ratio:
          For a Pareto distribution: P(X > x) = (x_m / x)^α
          p50 → (x_m / p50)^α = 0.50  → α = log(0.50) / log(x_m / p50)
          p99 → (x_m / p99)^α = 0.01

        We use a simplified Pareto variate: x_m / U^(1/α) where U ~ Uniform(0,1).
        """
        p50 = max(1.0, self._profile.latency_p50_ms)
        p99 = max(p50 + 1.0, self._profile.latency_p99_ms)

        # Estimate x_m (scale) and α (shape) from p50 and p99
        # p50: (x_m/p50)^α = 0.50  → x_m = p50 * 0.50^(1/α)
        # p99: (x_m/p99)^α = 0.01
        # Dividing: (p99/p50)^α = 0.01/0.50 = 0.02
        # α = log(0.02) / log(p50/p99)
        ratio = p50 / p99
        if ratio >= 1.0:
            ratio = 0.9999
        alpha = math.log(0.02) / math.log(ratio)
        alpha = max(0.5, alpha)

        # x_m such that P(X > p50) = 0.50
        x_m = p50 * (0.5 ** (1.0 / alpha))
        x_m = max(1.0, x_m)

        # Pareto variate
        u = self._rng.random()
        u = max(1e-9, u)  # avoid division by zero
        sample = x_m / (u ** (1.0 / alpha))

        # Jitter
        jitter = self._rng.uniform(-self._profile.jitter_ms / 2.0, self._profile.jitter_ms / 2.0)
        total_ms = sample + jitter
        total_ms = max(1.0, total_ms)

        # Convert ms → ticks so that delivery_tick aligns with pacemaker timing
        return max(1, int(round(total_ms / VIEW_TICK_MS)))

    def dispatch(
        self,
        target_node_id: str,
        message: Any,
        current_tick: int,
        extra_delay_ticks: int = 0,
    ) -> bool:
        """Schedule a message for delivery. Returns False if the message was dropped (packet loss)."""
        loss_pct = self._profile.packet_loss_pct
        if loss_pct > 0.0 and self._rng.random() < loss_pct / 100.0:
            return False  # dropped

        delay = self._pareto_delay_ticks() + extra_delay_ticks
        delivery_tick = current_tick + delay

        self._pending.append(PendingMessage(
            delivery_tick=delivery_tick,
            target_node_id=target_node_id,
            message=message,
        ))
        return True

    def flush(self, current_tick: int) -> Dict[str, List[Any]]:
        """Deliver all messages with delivery_tick <= current_tick.
        Returns a dict mapping node_id → list of delivered messages.
        """
        delivered: Dict[str, List[Any]] = {}
        remaining: List[PendingMessage] = []

        for pm in self._pending:
            if pm.delivery_tick <= current_tick:
                if pm.target_node_id not in delivered:
                    delivered[pm.target_node_id] = []
                delivered[pm.target_node_id].append(pm.message)
            else:
                remaining.append(pm)

        self._pending = remaining
        self._current_tick = current_tick
        return delivered

    def pending_count(self) -> int:
        return len(self._pending)
