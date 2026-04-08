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

Accordis is an [OpenEnv](https://openenv.dev) reinforcement learning environment with reactive adversary and auto-advancing curriculum for Byzantine Fault Tolerant (BFT) consensus tuning. An RL agent learns to configure per-node BFT protocol parameters across a cluster of honest nodes while a Byzantine adversary actively tries to disrupt consensus.

## Key Characteristics

**Auto-advancing curriculum.** The environment tracks agent performance over a rolling 50-episode window and automatically advances difficulty (levels 1–8) when the agent's liveness rate sustains above 85%. No manual configuration or external scheduler is needed — the environment promotes itself as the agent improves.

**Reactive adversary.** One Byzantine failure strategy (`ADAPTIVE_MIRROR`) reads the agent's live per-node configuration each step and calibrates its disruption timing to land just after the agent's vote aggregation window closes. The adversary tightens its attack as the agent tunes more precisely, creating a continuous pressure signal rather than a fixed obstacle.

**Partial observability under adversarial conditions.** Each honest node sees only its own local metrics — phase latencies, QC miss streaks, peer suspicion signals, throughput. The agent must infer cluster-wide health and Byzantine activity from these fragmented views while tuning five configuration knobs per node per round.

**Built-in correctness verification.** A correctness oracle runs Agreement, Validity, and Liveness checks against the full hidden state every step. Safety violations terminate the episode immediately with zero score, making the environment intolerant of unsafe configurations regardless of throughput gains.

**Reproducible adversarial episodes.** All Byzantine strategy selection is seeded, so every episode can be exactly replayed for debugging, ablations, or fair multi-agent comparison.

## What Is It and Why Was It Built

Modern distributed systems that run consensus protocols — blockchains, coordination services, replicated state machines — expose a set of configuration knobs (timeouts, pipeline depths, batch sizes, fault thresholds) that must be tuned correctly to maintain safety and liveness under adversarial network conditions. Incorrect tuning leads to either spurious view changes and throughput collapse, or delayed detection of Byzantine leaders and prolonged stalls.

Accordis makes this tuning problem an RL task. The agent observes per-node partial views of the cluster at each consensus round and must output a joint configuration update for all honest nodes. The environment simultaneously runs a Byzantine adversary that escalates its attack strategy as the curriculum difficulty increases.

The goal is to train agents that can:

- Maintain liveness and commit throughput under Byzantine network conditions.
- Recover quickly from leader failures and view timeouts without over-triggering spurious view changes.
- Contain equivocation and fork attacks through appropriate threshold configuration.
- Outperform a static baseline configuration on throughput and latency.

## What Is Being Simulated

### Consensus Protocol — Chained HotStuff

The core simulation runs a **Chained HotStuff** consensus engine entirely in memory. Chained HotStuff is a pipelined BFT consensus protocol designed for high throughput in partially synchronous networks. The simulation implements:

- **QC-based pipelining**: each block carries a `justify` Quorum Certificate (QC) for the previous block, allowing proposal and voting to overlap across views.
- **3-chain commit rule**: a block is finalized only when a chain of three consecutive QC-linked blocks is observed, ensuring safety.
- **Safe node predicate (`locked_qc`)**: replicas only vote for blocks extending their locked QC, preventing equivocation from causing safety violations.
- **Per-node pacemaker**: each node independently tracks view timeouts (`view_timeout_ms`) and sends `NEW_VIEW` messages when it suspects a Byzantine or crashed leader.
- **Sync protocol**: nodes that fall behind by more than `SYNC_LAG_THRESHOLD` views request a sync from peers to catch up their committed log.
- **Byzantine strategy execution**: the adapter injects disruption payloads directly into the message layer per step, simulating realistic adversarial behavior.

### Network and Fault Simulation

Each curriculum level uses a `FaultProfile` that configures network latency (p50, p99), jitter, and packet loss. The simulated message bus applies these parameters per-tick (at `VIEW_TICK_MS = 50ms`) to all inter-node messages.

### Byzantine Failure Agent

The `ByzantineFailureAgent` selects one of eight disruption strategies each step, drawn from a curriculum-level strategy pool using a seeded PRNG for reproducibility:

| Level | Strategy Pool                            |
| ----- | ---------------------------------------- |
| 1     | `NONE`                                 |
| 2     | `RANDOM_DELAY`                         |
| 3     | `SELECTIVE_DELAY`                      |
| 4     | `EQUIVOCATION`                         |
| 5     | `ADAPTIVE_MIRROR`                      |
| 6     | `LEADER_SUPPRESS`, `SELECTIVE_DELAY` |
| 7     | `CASCADE_TIMING`, `EQUIVOCATION`     |
| 8     | Full coalition (all strategies)          |

Strategy details:

- `RANDOM_DELAY`: injects 100–500ms delay on all messages from Byzantine nodes.
- `SELECTIVE_DELAY`: delays messages to half the honest nodes (200–800ms).
- `EQUIVOCATION`: sends conflicting proposals to two disjoint subsets of honest nodes.
- `LEADER_SUPPRESS`: suppresses all messages to honest nodes, stalling the current leader.
- `CASCADE_TIMING`: injects cascading delays (300–1000ms) timed to disrupt QC formation.
- `RECOVERY_DELAY`: targets nodes that just completed a view change (500–2000ms).
- `ADAPTIVE_MIRROR`: mirrors the minimum honest node `view_timeout_ms` with a delta offset to reliably trigger view changes.
- `FORK`: proposes conflicting blocks to force a fork attempt.

### Leader Rotation

Three rotation policies are supported, escalating with curriculum level:

- **Round Robin** (levels 1–6): deterministic rotation through node IDs.
- **VRF** (level 7): verifiable random function-based leader selection.
- **Reputation Weighted** (level 8): probabilistic selection weighted by historical commit success.

## Repository Layout

```text
accordis/
├── client.py                    # Python client for connecting to a running Accordis server
├── inference.py                 # Baseline runner entrypoint (static and LLM-backed policies)
├── models.py                    # Pydantic models: actions, observations, rewards, state
├── openenv.yaml                 # OpenEnv/Hugging Face deployment manifest
├── pyproject.toml               # Project metadata and dependencies
├── server/
│   ├── app.py                   # FastAPI/OpenEnv application factory
│   ├── accordis_environment.py  # Core environment orchestrator
│   ├── adapters/
│   │   ├── base.py              # BaseConsensusAdapter abstract interface
│   │   ├── __init__.py          # create_adapter() factory (ACCORDIS_ADAPTER env var)
│   │   └── simulated/           # Chained HotStuff in-memory adapter
│   │       ├── adapter.py       # SimulatedConsensusAdapter
│   │       ├── hotstuff_sim.py  # Chained HotStuff engine
│   │       ├── bfa_sim.py       # Byzantine message injection
│   │       └── network_sim.py   # Fault profile + latency simulation
│   ├── adversary/bfa.py         # ByzantineFailureAgent (strategy selection)
│   ├── api/v1/baseline.py       # Baseline evaluation endpoint
│   ├── curriculum/manager.py    # Episode outcome tracking and auto-advance
│   ├── network/                 # FaultProfile definitions per curriculum level
│   ├── oracle/verifier.py       # Agreement, Validity, Liveness correctness checks
│   ├── rewards/reward_calculator.py  # Per-step reward computation
│   ├── tasks/                   # easy / medium / hard task definitions
│   └── utils/                   # Logging, system prompt, LLM clients, helpers
└── tests/                       # App, model, reward, adapter, oracle, and task tests
```

## Core API

### Action

`MultiNodeAction` contains one `AccordisAction` per honest node. Each node action tunes five BFT knobs:

| Field                           | Bounds       | Default | Description                                             |
| ------------------------------- | ------------ | ------- | ------------------------------------------------------- |
| `view_timeout_ms`             | 200–3000 ms | 2000    | Leader timeout before triggering a view change          |
| `pipeline_depth`              | 1–8         | 2       | Number of in-flight proposal slots                      |
| `replication_batch_size`      | 1–512       | 64      | Transactions per proposed block                         |
| `equivocation_threshold`      | 1–15        | 5       | Conflicting-vote count before flagging a node Byzantine |
| `vote_aggregation_timeout_ms` | 50–1000 ms  | 500     | Max wait for votes before declaring a QC miss           |

All values are clamped to `SAFE_BFT_TUNING_BOUNDS` before reaching the adapter. `vote_aggregation_timeout_ms` is additionally constrained to `< view_timeout_ms // 2`.

Actions also accept optional `suspect_node` and `clear_suspicion` fields to directly signal Byzantine suspicion.

### Observation

`MultiNodeObservation` returns a dictionary of per-honest-node observations. Each `AccordisObservation` exposes:

- `current_role` and `current_view`
- `per_phase_latency_p50` / `per_phase_latency_p99` (prepare, pre_commit, commit, decide)
- `qc_formation_miss_streak` — consecutive rounds without a formed QC
- `view_change_count_recent` — view changes in the last 50 steps
- `view_stuck_ms` — wall-clock-equivalent ms the node has been in its current view
- `equivocation_miss_streak` — per-peer conflicting vote counts
- `message_arrival_variance` — inter-message variance per peer (jitter signal)
- `suspected_byzantine` — peer suspicion flags
- `commit_throughput_tps`, `pending_txn_count`, `pipeline_utilisation`
- `current_config` — the node's active BFTConfig

The full `AccordisState` (committed logs, BFA strategy, proposal registry) is never exposed to the agent — it is only available to the oracle, reward calculator, and episode log.

### Reward

Per-step reward blends nine signals:

| Signal                      | Value | When                                                  |
| --------------------------- | ----- | ----------------------------------------------------- |
| `liveness_cost`           | −1.0 | Every step while pending txns exist                   |
| `block_commit`            | +60   | Per block's worth of txns finalized (3-chain commit)  |
| `unnecessary_view_change` | −150 | View change with no Byzantine activity                |
| `liveness_stall`          | −30  | No new commits for > 80 steps                         |
| `fast_leader_recovery`    | +120  | View change under attack, liveness recovers           |
| `false_positive_avoided`  | +40   | Attack contained without agreement/validity violation |
| `pipeline_efficiency`     | +15   | Episode-end: avg pipeline utilisation above threshold |
| `throughput_improvement`  | +20   | Episode-end: TPS beats static baseline                |
| `latency_improvement`     | +10   | Episode-end: view-change overhead below baseline      |

### Termination

Episodes terminate when (in priority order):

1. **Pool drained** — all transactions committed (success).
2. **Agreement violated** — two honest nodes committed conflicting blocks.
3. **Validity violated** — a committed transaction was not in the honest proposal set.
4. **Max steps reached** — step budget exhausted.

## Tasks

Three benchmark tasks span eight curriculum levels:

### Easy (`easy`)

- **Curriculum levels**: 1–2
- **Nodes**: 4 honest, 0 (level 1) / 1 (level 2) Byzantine
- **Strategy**: `NONE` / `RANDOM_DELAY`
- **Leader rotation**: Round Robin
- **Max steps**: 50
- **Grader**: `0.5 × liveness_rate + 0.3 × max(0, 1 − vc_count/5) + 0.2 × correctness`

### Medium (`medium`)

- **Curriculum levels**: 3–5
- **Nodes**: 7 honest, 2 Byzantine (1 crash + 1 active)
- **Strategy**: `SELECTIVE_DELAY` / `EQUIVOCATION` / `ADAPTIVE_MIRROR`
- **Leader rotation**: Round Robin
- **Max steps**: 100
- **Grader**: `0.4 × liveness_rate + 0.2 × max(0, 1 − vc_count/10) + 0.2 × recovery_bonus + 0.2 × correctness`

### Hard (`hard`)

- **Curriculum levels**: 6–8
- **Nodes**: 10 honest, 2 Byzantine (coordinated attack)
- **Strategy**: `LEADER_SUPPRESS + SELECTIVE_DELAY` (lvl 6), `CASCADE_TIMING + EQUIVOCATION` (lvl 7), full coalition (lvl 8)
- **Leader rotation**: Round Robin (lvl 6) → VRF (lvl 7) → Reputation Weighted (lvl 8)
- **Pool size**: 1800 transactions
- **Max steps**: 100
- **Grader**: `0.05 × liveness_rate + 0.75 × throughput_score + 0.10 × vc_penalty + 0.10 × correctness`
- **Expected scores at level 6**: static defaults ~0.26, median LLM ~0.42, expert agent ~0.76

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Create a local `.env` from the provided `.env.sample`:

```bash
ACCORDIS_ADAPTER=simulated
ACCORDIS_TASKS=easy,medium,hard
ACCORDIS_MAX_STEPS=150

# LLM provider (for inference.py)
PROVIDER=openai
API_BASE_URL=https://router.huggingface.co/v1
MODEL=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=...
```

Notes:

- `ACCORDIS_ADAPTER=simulated` is the only supported adapter.
- `PROVIDER` can be `static`, `openai`, or `gemini`.
- Gemini supports key rotation via `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc.
- Do not commit real API keys into the repository.

### 3. Run the server

```bash
uv run --project . server
```

Or:

```bash
uvicorn accordis.server.app:app --reload
```

Once running:

| Surface           | URL                              |
| ----------------- | -------------------------------- |
| Web UI            | `http://localhost:8000/web`    |
| OpenAPI docs      | `http://localhost:8000/docs`   |
| Health check      | `http://localhost:8000/health` |
| WebSocket session | `ws://localhost:8000/ws`       |

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

- `POST /reset` works over HTTP and returns an initial observation.
- `POST /step` over plain HTTP is not useful for multi-step episodes — each request creates a fresh environment instance.
- `/ws` is the intended interface for interactive episodes; the environment session persists across `reset`, `step`, and `state` messages.

If building an agent loop, prefer the Python client or direct WebSocket usage.

## Running the Baseline

```bash
uv run python inference.py --provider openai --tasks easy,medium,hard
```

Examples:

```bash
uv run python inference.py
uv run python inference.py --provider huggingface--model Qwen/Qwen2.5-72B-Instruct --tasks easy
uv run python inference.py --provider static --tasks easy
```

The baseline is also available via the server API:

```bash
curl -X POST http://localhost:8000/baseline/ \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai","tasks":["easy","medium","hard"]}'
```

### Inference stdout format

The inference script emits exactly three line types:

```
[START] task=<task_name> adapter=<adapter>
[STEP]  step=<n> reward=<r> total=<cumulative> done=<True|False>
[END]   steps=<n> total_reward=<r> score=<s>
```

## Testing

```bash
uv run pytest
```

The test suite covers app endpoints, WebSocket flow, models and validation, reward logic, oracle/verifier behavior, the simulated adapter, and task graders.

## Docker

```bash
docker build -t accordis:latest .
docker run --rm -p 8000:8000 --env-file .env accordis:latest
```

The image starts `uvicorn accordis.server.app:app` on port `8000`.

## Deploying with OpenEnv

```bash
openenv push
openenv push --private
openenv push --repo-id <namespace>/<repo>
```
