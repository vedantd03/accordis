"""Adapter factory — create_adapter() returns the appropriate BaseConsensusAdapter."""

from __future__ import annotations

import os
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.adapters.base import BaseConsensusAdapter


def create_adapter(adapter_type: Optional[str] = None, **kwargs) -> BaseConsensusAdapter:
    """Factory function that returns the appropriate adapter instance.

    Args:
        adapter_type: "simulated" (default) or "librabft".
                      Falls back to ACCORDIS_ADAPTER env var, then "simulated".
        **kwargs:     Passed to the adapter constructor.

    Returns:
        A BaseConsensusAdapter instance ready to use.
    """
    if adapter_type is None:
        adapter_type = os.environ.get("ACCORDIS_ADAPTER", "simulated")

    adapter_type = adapter_type.lower()

    if adapter_type == "simulated":
        from server.adapters.simulated.adapter import SimulatedConsensusAdapter
        seed = kwargs.get("seed", 42)
        return SimulatedConsensusAdapter(seed=seed)

    if adapter_type == "librabft":
        raise NotImplementedError(
            "LibraBFT adapter (Version 2) is not implemented in this release. "
            "Use adapter_type='simulated' or set ACCORDIS_ADAPTER=simulated."
        )

    raise ValueError(
        f"Unknown adapter type: {adapter_type!r}. "
        "Valid options: 'simulated', 'librabft'."
    )


__all__ = ["create_adapter", "BaseConsensusAdapter"]
