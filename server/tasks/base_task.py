"""BaseTask — abstract base for all AccordisEnvironment tasks.

Tasks define initial conditions and grading logic only.
They never reference an adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from accordis.models import AccordisRubric, BFAStrategy, EpisodeLog, LeaderRotation


class BaseTask(ABC):
    """Abstract base for all task definitions."""

    task_id:           str
    curriculum_levels: List[int]
    n_nodes:           int
    f_byzantine:       int
    leader_rotation:   LeaderRotation
    max_steps:         int

    def get_initial_conditions(self) -> dict:
        """Return kwargs for AccordisEnvironment.reset()."""
        return {
            "n_nodes":          self.n_nodes,
            "f_byzantine":      self.f_byzantine,
            "leader_rotation":  self.leader_rotation,
            "curriculum_level": self.curriculum_levels[0],
            "max_steps":        self.max_steps,
        }

    def get_rubric(self) -> AccordisRubric:
        """Return the rubric for this task."""
        return AccordisRubric(
            task_id=self.task_id,
            curriculum_level=self.curriculum_levels[0],
        )

    @abstractmethod
    def grade(self, episode_log: EpisodeLog) -> float:
        """Deterministic score in [0.0, 1.0]. No adapter dependency.

        Args:
            episode_log: Full episode log from the environment.

        Returns:
            Scalar score in [0.0, 1.0].
        """
        ...
