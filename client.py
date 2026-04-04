# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client helpers for the Accordis synchronous-round environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from accordis.models import AccordisObservation, AccordisAction


class AccordisEnvironment(EnvClient[AccordisAction, AccordisObservation, State]):
    """
    Client for the Accordis Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    The environment is round-based rather than turn-based:
    one `step()` call sends the joint action for a full synchronous consensus
    round and receives one observer's local post-round observation.

    Example:
        >>> with AccordisEnvironment(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.observer_id)
        ...
        ...     action = AccordisAction(observer_id="node_0", node_actions={})
        ...     result = client.step(action)
        ...     print(result.observation.info)
    """

    def _step_payload(self, action: AccordisAction) -> Dict:
        """
        Convert the synchronous round action to the step request payload.

        Args:
            action: AccordisAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict) -> StepResult[AccordisObservation]:
        """
        Parse server response into StepResult[AccordisObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AccordisObservation
        """
        obs_data = payload.get("observation", {})
        observation = AccordisObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward")),
                "metadata": obs_data.get("metadata", {}),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
