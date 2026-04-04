"""FaultProfile definitions — shared by both Version 1 (simulated) and Version 2 (LibraBFT).

Both adapters read from get_fault_profile() and apply it in their own way.
This file is never modified when switching adapter versions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaultProfile:
    """Network fault parameters applied to all inter-node links."""
    latency_p50_ms:  float
    latency_p99_ms:  float
    jitter_ms:       float
    packet_loss_pct: float
    distribution:    str = "pareto"


FAULT_PROFILES: dict[int, FaultProfile] = {
    1: FaultProfile(latency_p50_ms=2,   latency_p99_ms=5,   jitter_ms=1,  packet_loss_pct=0.0),   # Stable
    2: FaultProfile(latency_p50_ms=5,   latency_p99_ms=50,  jitter_ms=5,  packet_loss_pct=0.0),   # Variable
    3: FaultProfile(latency_p50_ms=10,  latency_p99_ms=200, jitter_ms=20, packet_loss_pct=0.0),   # Burst p99
    4: FaultProfile(latency_p50_ms=10,  latency_p99_ms=100, jitter_ms=10, packet_loss_pct=10.0),  # Packet loss
    5: FaultProfile(latency_p50_ms=5,   latency_p99_ms=150, jitter_ms=15, packet_loss_pct=2.0),   # Regime switches
    6: FaultProfile(latency_p50_ms=20,  latency_p99_ms=300, jitter_ms=30, packet_loss_pct=5.0),   # Crash + adversarial
    7: FaultProfile(latency_p50_ms=80,  latency_p99_ms=500, jitter_ms=50, packet_loss_pct=0.0),   # Geo-distributed
    8: FaultProfile(latency_p50_ms=30,  latency_p99_ms=400, jitter_ms=40, packet_loss_pct=8.0),   # Non-stationary
}


def get_fault_profile(curriculum_level: int) -> FaultProfile:
    """Return the FaultProfile for the given curriculum level (1–8)."""
    level = max(1, min(8, curriculum_level))
    return FAULT_PROFILES[level]
