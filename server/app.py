# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Accordis consensus environment.

The HTTP surface exposes a synchronous-round environment:
- clients submit one joint round action,
- the server advances one logical round,
- the server returns one agent-scoped observation.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from dotenv import load_dotenv
load_dotenv()

from server.utils.logger import setup_logger
setup_logger()

from accordis.models import MultiNodeAction, MultiNodeObservation
from accordis.server.accordis_environment import AccordisEnvironment
from accordis.server.adapters import create_adapter


def _make_env() -> AccordisEnvironment:
    """Factory function: create adapter then inject into environment."""
    adapter = create_adapter()
    return AccordisEnvironment(adapter=adapter)


# Create the app with web interface and README integration
app = create_app(
    _make_env,
    MultiNodeAction,
    MultiNodeObservation,
    env_name="accordis",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m accordis.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn accordis.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
