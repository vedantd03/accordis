# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the server locally
uv run --project . server
uvicorn accordis.server.app:app --reload

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_environment.py

# Run a single test
uv run pytest tests/test_environment.py::TestReset::test_reset_returns_obs_dict

# Deploy to Hugging Face Spaces
openenv push
openenv push --repo-id my-org/my-env --private
```

## Architecture

Accordis is an **OpenEnv RL environment** for self-adaptive Byzantine Fault Tolerant (BFT) consensus tuning. An RL agent learns to configure BFT protocol parameters across honest nodes in a cluster while a Byzantine adversary tries to disrupt consensus.

### Layer Model

The codebase is organized as three explicit layers:

**Layer 1 — Data/Contracts** (`models.py`):
All Pydantic types. `AccordisAction` (per-node BFT config knobs), `AccordisObservation` (per-node partial view), `AccordisState` (full hidden state), `BFTConfig`, `SAFE_BFT_TUNING_BOUNDS`, reward/rubric types.

**Layer 2 — Environment Logic** (`server/accordis_environment.py`, `server/oracle/`, `server/rewards/`, `server/curriculum/`, `server/adversary/`):
`AccordisEnvironment` is the single orchestrator. It imports **only** from `BaseConsensusAdapter` (never from a concrete adapter). Each component is injected or instantiated at construction time:
- `ByzantineFailureAgent` — selects/applies BFA strategies each step
- `CorrectnessOracle` — verifies Agreement, Validity, Liveness on the full state
- `RewardCalculator` — computes per-step reward from state deltas and oracle output
- `CurriculumManager` — tracks episode outcomes and auto-advances difficulty

**Layer 3 — Adapter** (`server/adapters/`):
`BaseConsensusAdapter` (abstract) is the only seam between Layer 2 and any consensus engine. `create_adapter()` in `server/adapters/__init__.py` selects the implementation via `ACCORDIS_ADAPTER` env var (default: `"simulated"`). The LibraBFT adapter is stubbed — only `SimulatedConsensusAdapter` is implemented.

### Key design constraints

- `AccordisEnvironment` has **zero** imports from concrete adapter classes.
- `AccordisEnvironment.step()` clamps all action values via `_clamp_action()` before passing to the adapter. The adapter always receives in-bounds values per `SAFE_BFT_TUNING_BOUNDS`.
- Observations are **agent-scoped** (one honest node's partial view per call). Actions are **round-scoped** (`MultiNodeAction = Dict[NodeID, AccordisAction]` for a full synchronous round).
- `AccordisState` (full true state) is never exposed to the agent — only to the oracle, reward calculator, and episode log.
- `vote_aggregation_timeout_ms` is additionally constrained to `< view_timeout_ms // 2` regardless of bounds table.

### HTTP / WebSocket surface (`server/app.py`)

Uses `openenv-core`'s `create_app()` factory. The `ACCORDIS_ADAPTER` env var switches adapter at startup. Endpoints: `/web` (UI), `/docs` (OpenAPI), `/health`, `/ws` (WebSocket session).

### Tasks (`server/tasks/`)

`BaseTask` defines initial reset conditions and a `grade(episode_log)` method. Three difficulty tiers: `task_easy.py`, `task_medium.py`, `task_hard.py`. Tasks never reference an adapter.

### Client (`client.py`)

`AccordisEnvironment` (client-side) extends `EnvClient` from `openenv-core`. Uses WebSocket for low-latency multi-step episodes.

### Inference script (`inference.py`)

Template for submitting an LLM-driven agent. Requires `ACCORDIS_ADAPTER`, `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars. Must emit `[START]`, `[STEP]`, `[END]` lines to stdout in the exact format specified at the top of that file.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

See [AGENTS.md](./AGENTS.md) for the full guide on which tools to use and when. Prefer graph tools over Grep/Glob/Read for exploring the codebase.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes � gives risk-scored analysis |
| `get_review_context` | Need source snippets for review � token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
