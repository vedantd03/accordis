"""Typed models for the Accordis synchronous distributed-consensus environment.

The key API decision in this module is intentional:

- Observations are agent-scoped. A caller receives exactly one observer's view.
- Actions are round-scoped. A single environment step represents one synchronous
  consensus round and therefore accepts one joint action containing all node
  decisions for that round, plus an optional adversary intervention.

This keeps the environment faithful to synchronous distributed consensus while
still exposing a partially observed interface to each training agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

# TODO: Define all the models required for action, observation, state, reward and rubric here.

class AccordisAction(Action):
    pass


class AccordisObservation(Observation):
    pass