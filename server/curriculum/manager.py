"""CurriculumManager — rolling episode window with automatic level advancement.

Shared between both adapter versions. Operates on liveness scores only.
"""

from __future__ import annotations

from collections import deque
from typing import Deque


WINDOW_SIZE      = 50
ADVANCE_THRESHOLD = 0.85
MAX_LEVEL        = 8


class CurriculumManager:
    """Tracks episode performance and advances the curriculum level automatically.

    Uses a rolling 50-episode window of liveness_rate scores.
    Advances when the window is full AND the rolling average exceeds 0.85.
    """

    def __init__(self, initial_level: int = 1) -> None:
        self._level: int = max(1, min(MAX_LEVEL, initial_level))
        self._window: Deque[float] = deque(maxlen=WINDOW_SIZE)

    @property
    def level(self) -> int:
        """Current curriculum level (read-only)."""
        return self._level

    def record_episode(self, liveness_rate: float) -> None:
        """Append a liveness_rate to the rolling window.

        Args:
            liveness_rate: Float in [0.0, 1.0] from LivenessResult.
        """
        self._window.append(max(0.0, min(1.0, liveness_rate)))

    def should_advance(self) -> bool:
        """Return True if window is full (50 episodes) AND rolling average > 0.85."""
        if len(self._window) < WINDOW_SIZE:
            return False
        average = sum(self._window) / len(self._window)
        return average > ADVANCE_THRESHOLD

    def advance(self) -> int:
        """Increment level to max 8. Returns new level.

        Clears the rolling window after advancement.
        """
        if self._level < MAX_LEVEL:
            self._level += 1
            self._window.clear()
        return self._level

    def maybe_advance(self) -> bool:
        """Convenience: advance if should_advance(). Returns True if advanced."""
        if self.should_advance():
            self.advance()
            return True
        return False

    def reset_window(self) -> None:
        """Clear the rolling window without changing the level."""
        self._window.clear()

    def __repr__(self) -> str:
        return (
            f"CurriculumManager(level={self._level}, "
            f"window_size={len(self._window)}/{WINDOW_SIZE}, "
            f"avg={sum(self._window) / max(1, len(self._window)):.3f})"
        )
