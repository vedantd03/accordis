"""Main environment implementation for Accordis.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_server.interfaces import Environment, Transform
from openenv.core.env_server.types import State
from openenv.core.rubrics import Rubric

from ..models import AccordisObservation, AccordisAction


class AccordisTransform(Transform[AccordisObservation]):
    """Annotate a scoped observation with readable round-level metadata."""

    def __init__(self, env: "AccordisEnvironment") -> None:
        self.env = env

    def __call__(self, observation: AccordisObservation) -> AccordisObservation:
        raise NotImplementedError("Observation transformation is not implemented yet.")


class AccordisRubric(Rubric):
    """Compute the scalar reward emitted for the current step."""

    def __init__(self, env: "AccordisEnvironment") -> None:
        super().__init__()
        self.env = env

    def reset(self) -> None:
        raise NotImplementedError("Rubric reset logic is not implemented yet.")

    def forward(self, action, observation: AccordisObservation) -> float:
        raise NotImplementedError("Reward rubric is not implemented yet.")


class AccordisEnvironment(Environment):
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(
            transform=AccordisTransform(self), 
            rubric=AccordisRubric(self)
        )

    def reset(self) -> AccordisObservation:
        """Start a new episode and return one observer's initial local view.
        """
        raise NotImplementedError("Environment reset logic is not implemented yet.")

    def step(self, action: AccordisAction) -> AccordisObservation:  # type: ignore[override]
        """Advance the environment by one synchronous consensus round.
        """
        raise NotImplementedError("Environment step logic is not implemented yet.")

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(
        self,
    ) -> AccordisObservation:
        """Project the hidden full-system state into one observer's view."""
        raise NotImplementedError("Observation building logic is not implemented yet.")
       
