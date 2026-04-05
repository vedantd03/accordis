---
title: Accordis Environment
emoji: "⚖️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - distributed-systems
  - consensus
---
# Accordis Environment

Accordis is an OpenEnv environment for tuning a synchronous Byzantine fault tolerant consensus system. Each environment step represents one full consensus round across multiple honest nodes, and the agent chooses per-node BFT configuration values to improve liveness, throughput, and recovery under adversarial conditions.

The current release ships with a fully in-memory simulated adapter, task definitions for `easy`, `medium`, and `hard`, a FastAPI/OpenEnv server, and a baseline runner that can use either a static policy or an LLM-backed policy.

## Project Overview

- Round-based environment: one `step()` call advances the cluster by one synchronous consensus round.
- Multi-node control surface: actions are joint per-node configuration updates, not single-agent moves.
- Partial observability: observations expose honest-node local metrics only.
- Built-in adversary: the environment selects Byzantine failure strategies as episodes progress.
- Curriculum and grading: tasks score performance differently across easy, medium, and hard settings.
- OpenEnv-compatible server: HTTP, WebSocket, web UI, and deployment manifest are already wired up.

## Repository Layout

```text
accordis/
├── client.py                    # Python client for connecting to a running Accordis server
├── inference.py                 # Baseline runner entrypoint
├── models.py                    # Pydantic models for actions, observations, rewards, and state
├── openenv.yaml                 # OpenEnv/Hugging Face deployment manifest
├── pyproject.toml               # Project metadata and dependencies
├── server/
│   ├── app.py                   # FastAPI/OpenEnv application
│   ├── accordis_environment.py  # Core environment orchestration
│   ├── adapters/                # Adapter factory and simulated adapter
│   ├── adversary/               # Byzantine failure agent logic
│   ├── api/v1/baseline.py       # Baseline evaluation endpoint
│   ├── curriculum/              # Curriculum progression logic
│   ├── network/                 # Fault profile definitions
│   ├── oracle/                  # Agreement/validity/liveness checks
│   ├── rewards/                 # Reward calculation
│   ├── tasks/                   # easy / medium / hard task definitions
│   └── utils/                   # Logging, prompts, LLM clients, helpers
└── tests/                       # App, model, reward, adapter, and verifier tests
```

## Core API Model

### Action

`MultiNodeAction` contains one `AccordisAction` per honest node. Each node action tunes:

- `view_timeout_ms`
- `pipeline_depth`
- `replication_batch_size`
- `equivocation_threshold`
- `vote_aggregation_timeout_ms`

The environment clamps all tunable values to safe bounds before they reach the adapter.

### Observation

`MultiNodeObservation` returns a dictionary of honest-node observations. Each `AccordisObservation` includes:

- current role and view
- per-phase latency percentiles
- QC miss streak and recent view changes
- equivocation and peer suspicion signals
- throughput, pending transactions, and pipeline utilisation
- the node's currently active BFT config

### Reward and Termination

Per-step reward blends liveness, throughput, latency, recovery, and stability signals. Episodes terminate when:

- the transaction pool is drained
- agreement is violated
- validity is violated
- `max_steps` is reached

## Tasks

The project includes three benchmark-style tasks:

- `easy`: 4 nodes, up to 1 Byzantine node, short 15-step budget, focused on stable consensus.
- `medium`: 7 nodes, 2 faulty nodes, 70-step budget, focused on recovery under delays/equivocation.
- `hard`: 10 nodes, 3 faulty nodes, 100-step budget, focused on throughput and stability under coordinated Byzantine pressure.

The hard task also increases transaction pool size and changes leader rotation at higher curriculum levels.

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Create a local `.env` with the variables you need, refer to the `.env.sample`. Common ones are:

```bash
PROVIDER=openai
MODEL=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=...
API_BASE_URL=https://api.openai.com/v1
ACCORDIS_ADAPTER=simulated
ACCORDIS_TASKS=easy,medium,hard
ACCORDIS_MAX_STEPS=150
```

Notes:

- `ACCORDIS_ADAPTER=simulated` is the supported adapter in this release.
- `PROVIDER` can be `static`, `openai`, or `gemini`.
- Gemini also supports numbered keys like `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc for key rotation to avoid rate limits.
- Do not commit real API keys into the repository.

### 3. Run the server locally

```bash
uv run --project . server
```

Or:

```bash
uvicorn accordis.server.app:app --reload
```

Once running, the main surfaces are:

- `http://localhost:8000/web` for the web UI
- `http://localhost:8000/docs` for OpenAPI docs
- `http://localhost:8000/health` for health checks
- `ws://localhost:8000/ws` for persistent stateful sessions

## Using the Python Client

```python
from accordis.client import AccordisEnvironment
from accordis.models import AccordisAction, MultiNodeAction

with AccordisEnvironment(base_url="http://localhost:8000") as env:
    reset_result = env.reset()
    node_ids = list(reset_result.observation.nodes.keys())

    action = MultiNodeAction(
        nodes={
            node_id: AccordisAction(
                node_id=node_id,
                view_timeout_ms=2000,
                pipeline_depth=2,
                replication_batch_size=64,
                equivocation_threshold=5,
                vote_aggregation_timeout_ms=500,
            )
            for node_id in node_ids
        }
    )

    step_result = env.step(action)
    print(step_result.reward, step_result.done)
```

## HTTP and WebSocket Behavior

The OpenEnv server supports both stateless HTTP endpoints and stateful WebSocket sessions:

- `POST /reset` works over HTTP and returns an initial observation.
- `POST /step` over plain HTTP is not useful for multi-step episodes because each request gets a fresh environment instance.
- `GET /state` over plain HTTP also operates on a fresh instance.
- `/ws` is the intended interface for interactive episodes because the environment session persists across `reset`, `step`, and `state` messages.

If you are building an agent loop, prefer the Python client or direct WebSocket usage.

## Baseline Evaluation

You can run the included baseline runner from the CLI:

```bash
uv run python inference.py --provider openai --tasks easy,medium,hard
```

Examples:

```bash
uv run python inference.py --provider openai --model Qwen/Qwen2.5-72B-Instruct --tasks easy
uv run python inference.py --provider gemini --model gemini-3.1-flash-lite-preview --tasks,medium,hard
```

The baseline API is also exposed from the server:

```bash
curl -X POST http://localhost:8000/baseline/ \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai","tasks":["easy","medium","hard"]}'
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

The repository includes tests for:

- app endpoints and WebSocket flow
- models and validation
- reward logic
- verifier/oracle behavior
- simulated adapter and task execution

## Docker

Build the image locally:

```bash
docker build -t accordis:latest .
```

Run it:

```bash
docker run --rm -p 8000:8000 --env-file .env accordis:latest
```

The Docker image starts `uvicorn accordis.server.app:app` on port `8000`.

## Deploying with OpenEnv

This repository already includes `openenv.yaml`, so it can be pushed as an OpenEnv space:

```bash
openenv push
```

Useful options:

- `openenv push --private`
- `openenv push --repo-id <namespace>/<repo>`
- `openenv push --base-image <image>`

## Current Implementation Notes

- The simulated adapter is the production path for this repo today.
- The `librabft` adapter path is declared but not implemented in this release.
- `max_concurrent_envs` is currently set to `1` in the server app.
- Runtime logs are written under `outputs/logs/`.
