"""Adversary package — Byzantine failure injection."""

from server.adversary.bfa import ByzantineFailureAgent, STRATEGY_POOL

__all__ = ["ByzantineFailureAgent", "STRATEGY_POOL"]
