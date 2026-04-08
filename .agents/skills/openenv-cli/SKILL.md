---
name: accordis
description: "Accordis is an OpenEnv reinforcement learning environment for training an LLM Agent to tune per-node BFT configuration parameters in a simulated Chained HotStuff cluster under Byzantine and adversarial network conditions, with a reactive adversary, partial observability, correctness checks, and an auto-advancing curriculum."
---
The OpenEnv CLI command `openenv` is available.
Use `openenv --help` to view available commands.

# SKILL.md — AccordisEnvironment

Guidance for coding assistants working in this repository.

## Project Summary

Accordis is an OpenEnv reinforcement-learning environment for tuning Byzantine Fault Tolerant consensus parameters under adversarial conditions. The repository contains:

- A FastAPI/OpenEnv server surface in `server/app.py`
- The core environment orchestrator in `server/accordis_environment.py`
- A simulated Chained HotStuff-style consensus adapter in `server/adapters/simulated/`
- Curriculum, reward, oracle, and task logic under `server/`
- Tests in `tests/`
- A baseline runner in `inference.py`

This is primarily a Python project with `uv` used for dependency management and execution.

## First Principles

When making changes, optimize for:

1. Preserving environment semantics and determinism
2. Keeping the environment adapter-agnostic outside adapter modules
3. Maintaining compatibility with the OpenEnv app surface
4. Updating or adding tests alongside behavior changes

Prefer small, surgical changes over broad rewrites.

## Source Of Truth

When documentation, tests, and implementation disagree, use this order of trust:

1. Current tests
2. Current implementation
3. README prose

There are some known mismatches in the repo between prose comments, README examples, and runtime values. Before "fixing" something that looks inconsistent, verify whether the tests intentionally encode the expected behavior.

## Repository Map

Top-level files:

- `README.md`: project overview, setup, API usage
- `pyproject.toml`: dependencies and package configuration
- `inference.py`: baseline evaluation entrypoint
- `client.py`: Python client for interacting with a running server

Core server modules:

- `server/app.py`: FastAPI/OpenEnv app factory and router mounting
- `server/accordis_environment.py`: main environment reset/step/state orchestration
- `server/router.py`: API routing setup
- `server/gradio_ui.py`: web UI integration

Consensus and simulation:

- `server/adapters/base.py`: adapter interface
- `server/adapters/__init__.py`: adapter factory
- `server/adapters/simulated/adapter.py`: simulated adapter entrypoint
- `server/adapters/simulated/hotstuff_sim.py`: consensus engine
- `server/adapters/simulated/network_sim.py`: latency/fault simulation
- `server/adapters/simulated/bfa_sim.py`: Byzantine action injection

Evaluation and control logic:

- `server/adversary/bfa.py`: strategy selection
- `server/oracle/verifier.py`: agreement/validity/liveness checks
- `server/rewards/reward_calculator.py`: reward shaping
- `server/curriculum/manager.py`: auto-advancing difficulty
- `server/tasks/`: benchmark task definitions

Tests:

- `tests/test_environment.py`: environment reset/step behavior
- `tests/test_app.py`: HTTP and WebSocket behavior
- `tests/test_tasks.py`: task grading and initial conditions
- Additional tests cover models, rewards, adapters, verifier, UI, and LLM factory

## Setup And Common Commands

Install dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run the server locally:

```bash
uv run --project . server
```

Alternative dev server:

```bash
uvicorn accordis.server.app:app --reload
```

Run the baseline script:

```bash
uv run python inference.py
```

## Environment Variables

Common environment variables mentioned or used in the repo:

- `ACCORDIS_ADAPTER=simulated`
- `ACCORDIS_TASKS=easy,medium,hard`
- `ACCORDIS_MAX_STEPS=150`
- `PROVIDER=static|openai|gemini`
- `ACCORDIS_BASE_URL=http://localhost:8000`

Notes:

- Only the `simulated` adapter is implemented.
- Do not commit real API keys or secrets.
- Check `.env.sample` before adding new configuration knobs.

## Architecture Rules

Follow these boundaries unless the task explicitly requires changing them:

- Keep `AccordisEnvironment` adapter-agnostic. It should depend on `BaseConsensusAdapter`, not simulated implementation details.
- Put simulation-specific logic in `server/adapters/simulated/`, not in the environment orchestrator.
- Put correctness checks in the oracle, not inline in unrelated modules.
- Put reward shaping in `reward_calculator.py`, not scattered across the environment.
- Keep API transport concerns in app/router layers, not mixed into environment logic.

If you need a new behavior, place it in the narrowest layer that owns it.

## Testing Expectations

Any change to behavior should be validated with tests when practical.

Priorities:

- If you change reset/step semantics, update `tests/test_environment.py`
- If you change HTTP or WebSocket behavior, update `tests/test_app.py`
- If you change grading or task defaults, update `tests/test_tasks.py`
- If you change model validation or schema behavior, update the corresponding model tests

Run targeted tests first when iterating, then the broader suite if the change is meaningful.

Examples:

```bash
uv run pytest tests/test_environment.py
uv run pytest tests/test_app.py
uv run pytest tests/test_tasks.py
```

## Common Pitfalls

- HTTP `/step` and `/state` are intentionally stateless in the current app wiring; multi-step interaction is expected over WebSocket or the Python client.
- `create_adapter()` only supports `simulated`; `librabft` is intentionally unimplemented.
- Task defaults, README descriptions, and inline comments may not always match each other exactly.
- Some tests validate behavior that may look odd at first glance. Read them before "correcting" an apparent bug.
- Avoid leaking hidden state into observations; the agent should only receive the intended partial view.

## Editing Guidance

When changing code:

- Preserve existing public model shapes unless the task requires a schema change
- Prefer explicit typing and clear data flow over clever abstractions
- Keep deterministic behavior where seeds are involved
- Avoid moving large amounts of code unless necessary for the task
- Update nearby comments/docstrings if they become false after your change

When changing docs:

- Keep README and code comments aligned with tested behavior
- Do not invent unsupported adapters, endpoints, or workflows

## If You Need To Explore

Good starting files for understanding behavior:

- `server/accordis_environment.py`
- `server/adapters/simulated/adapter.py`
- `server/adversary/bfa.py`
- `server/oracle/verifier.py`
- `server/rewards/reward_calculator.py`
- `tests/test_environment.py`
- `tests/test_app.py`
- `tests/test_tasks.py`

## Definition Of Done

A change is in good shape when:

- The implementation follows the existing module boundaries
- Tests covering the changed behavior pass, or any unrun tests are called out explicitly
- Docs and comments touched by the change remain accurate
- No secrets or machine-local artifacts were added to the repo
