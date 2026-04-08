# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client helpers for the Accordis synchronous-round environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from accordis.models import MultiNodeObservation, MultiNodeAction, AccordisState


class AccordisEnvironment(EnvClient[MultiNodeAction, MultiNodeObservation, AccordisState]):
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
        ...     print(result.observation.nodes.keys())
        ...
        ...     action = MultiNodeAction(nodes={})
        ...     result = client.step(action)
        ...     print(result.observation.nodes)
    """

    def _step_payload(self, action: MultiNodeAction) -> Dict:
        """
        Convert the synchronous round action to the step request payload.

        Args:
            action: MultiNodeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict) -> StepResult[MultiNodeObservation]:
        """
        Parse server response into StepResult[MultiNodeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MultiNodeObservation
        """
        obs_data = payload.get("observation", {})
        observation = MultiNodeObservation.model_validate(
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

    def _parse_state(self, payload: Dict) -> AccordisState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return AccordisState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            step=payload.get("step", 0),
            curriculum_level=payload.get("curriculum_level"),
            n_nodes=payload.get("n_nodes"),
            f_byzantine=payload.get("f_byzantine"),
            leader_rotation=payload.get("leader_rotation"),
            node_states=payload.get("node_states", {}),
            view_change_count=payload.get("view_change_count", 0),
            bfa_strategy=payload.get("bfa_strategy"),
            proposal_registry=payload.get("proposal_registry", {}),
            episode_txn_pool=payload.get("episode_txn_pool", []),
            finalized_txn_count=payload.get("finalized_txn_count", 0),
            episode_log=payload.get("episode_log"),
        )
        
