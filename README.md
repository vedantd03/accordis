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

> Infrastructure teams at companies like Coinbase, Visa, and AWS run distributed systems that must reach agreement across dozens or hundreds of nodes — on which transactions are valid, who owns what, what the current state of the database is. This state of agreement is achieved with the help of consensus protocols, these protocols expose configuration knobs (timeouts, batch sizes, fault thresholds) that are typically set once during deployment and left static. That works fine under normal conditions. But when a coordinated group of compromised validators starts injecting conflicting proposals, selectively withholding votes, and timing message delays to exploit the current configuration — static parameters become the weakest link. The protocol isn't broken; it's just tuned for calm weather, not a storm.

> Today, the response is experienced engineers working through runbooks — adjusting timeout values, analyzing historical patterns, deploying updated configurations. This process is rigorous but fundamentally slow: it takes hours or days of human analysis while an adversary adapts in minutes. Accordis exists to close that gap.

**Accordis is an [OpenEnv](https://openenv.dev) reinforcement learning environment that simulates this exact operational task** — tuning the configuration of a distributed consensus system in real time, under adversarial conditions, with partial information — so that RL agents can learn adaptive policies that outperform static configurations. The RL Agent will operate as an automated SRE, and will continuously re-tune consensus parameters at machine speed.

---

# The Real-World Problem

Distributed systems that need to reach agreement — payment networks, blockchain infrastructure, cloud databases (CockroachDB, Spanner, etcd), coordination services powering Kubernetes clusters and financial clearinghouses — all run consensus protocols.

These protocols expose a handful of configuration knobs — timeouts, batch sizes, pipeline depths, fault thresholds — that determine whether the system stays fast and correct or falls over. In production, these parameters are set once during deployment, usually to defaults or community-recommended values, and rarely changed afterward. This static approach works under stable conditions but creates a fundamental brittleness: **when conditions change, the configuration doesn't adapt.**

This brittleness is well-documented across production systems:

- **CockroachDB** engineers found that reducing the Raft election timeout from 3s to 2s cut crash-failover latency from 14.0s to 10.2s — but the new setting breaks for clusters with >500ms RTT. There is no single correct value; the optimal timeout depends on conditions that change at runtime. A 2016 engineering issue explicitly asked whether the election timeout could be set based on measured RTT — a request for adaptive tuning that was never fully implemented. ([PR #91947](https://github.com/cockroachdb/cockroach/pull/91947), [Issue #10133](https://github.com/cockroachdb/cockroach/issues/10133))
- **etcd**, the consensus store backing every Kubernetes cluster, states in its official documentation that default settings work for local networks but must be manually retuned for cross-datacenter deployments. OpenShift's etcd guide warns that disk latency alone can make a leader "effectively unavailable," and that more than one leader election per hour indicates instability. ([etcd tuning docs](https://etcd.io/docs/v3.4/tuning/))
- **Tendermint/CometBFT** formally recognized this problem in ADR-074, which proposed migrating timeout parameters from per-validator local config to network-wide consensus parameters because "proper functioning of the Tendermint consensus algorithm relies on these parameters being uniform across validators." A practical example: the Evmos chain discovered its default `timeout_commit` of 5s wasted time when actual consensus took under 1s — reducing it to 1s improved block times from ~5.9s to ~2s, a 3× improvement from adjusting a single parameter. ([ADR-074](https://github.com/tendermint/tendermint/blob/master/docs/architecture/adr-074-timeout-params.md))

The common thread: **operators set parameters once based on controlled benchmarks, then deploy into environments that are anything but controlled.** Network latency fluctuates. Nodes crash and recover unpredictably. And in blockchain and financial systems, some participants may be actively malicious — not just failing, but strategically undermining the protocol. A timeout that works at 50ms latency becomes a vulnerability at 500ms, and a threshold tuned for benign conditions becomes an exploit surface when an adversary times attacks to land just outside it.

No production-grade adaptive consensus parameter tuning exists today. Operators use runbooks, monitoring dashboards, and occasionally automated alerts — but the parameters themselves remain static between manual interventions. This gap between static configuration and dynamic, adversarial reality is where systems break. Accordis targets that gap.

---

## Why Reinforcement Learning?

Consensus parameter tuning under adversarial conditions has a specific structure that makes it a poor fit for static rules and a natural fit for RL.

**It is sequential with delayed consequences.** A timeout change at step *t* determines whether the system can recover from a leader failure at step *t + 5*, which determines whether the commit pipeline stalls at step *t + 10*. Heuristics and lookup tables react to the current state without considering downstream effects. RL's credit assignment is designed for exactly this kind of temporally extended reasoning.

**It is partially observable.** Each node reports only its own local view — its latencies, its vote counts, its suspicions about peers. The agent never sees the global cluster state, the adversary's chosen strategy, or the full message log. It must infer cluster-wide health from fragmented, noisy signals — the same constraint a human operator faces, but at a speed and scale no human can match.

**The adversary is adaptive.** This is the critical distinction from standard parameter optimization. A real attacker does not follow a fixed playbook — they observe the system's tuning and exploit it. If the agent tightens timeouts, the attacker delays messages just past the new threshold. If the agent raises the equivocation threshold, the attacker sends exactly that many conflicting votes. This creates a non-stationary environment where any fixed policy, no matter how well-tuned initially, will eventually be exploited. RL handles non-stationarity through continuous policy adaptation; static configurations cannot.

**The action space has complex constraints.** Five parameters per node, across all honest nodes simultaneously, with interdependencies between them (e.g., vote aggregation timeout must be less than half the view timeout). Exhaustive search is infeasible; gradient-based policy optimization is tractable.

BFTBrain (Wu et al., NSDI 2025) demonstrated 18–119% throughput improvements over fixed protocol configurations using RL-based adaptive switching — peer-reviewed validation that the adaptive approach works and that the gains are substantial.

---

# What is Accordis ?

```
                              ACCORDIS ARCHITECTURE
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │    ┌───────────────┐    Action (5 params x N nodes)    ┌────────────────┐   │
 │    │   RL Agent    │ ─────────────────────────────────> │  OpenEnv API   │   │
 │    │   (Policy)    │ <───────────────────────────────── │  (FastAPI/WS)  │   │
 │    └───────────────┘    Observation + Reward            └───────┬────────┘   │
 │                                                                │            │
 │  ┌─────────────────────────────────────────────────────────────┼──────────┐  │
 │  │  AccordisEnvironment Orchestrator                           │          │  │
 │  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │          │  │
 │  │  │  Curriculum   │  │     Reward       │  │ Correctness  │  │          │  │
 │  │  │   Manager     │  │   Calculator     │  │   Oracle     │  │          │  │
 │  │  │ (Levels 1-8)  │  │  (9 signals)     │  │ (Agreement,  │  │          │  │
 │  │  │ Auto-advance  │  │  Throughput +     │  │  Validity,   │  │          │  │
 │  │  │ on 85% liven. │  │  Safety + Resil. │  │  Liveness)   │  │          │  │
 │  │  └──────┬───────┘  └────────▲─────────┘  └──────┬───────┘  │          │  │
 │  └─────────┼───────────────────┼────────────────────┼──────────┘          │  │
 │            │                   │                    │                     │  │
 │  ┌─────────▼───────────────────┼────────────────────▼──────────────────┐  │  │
 │  │  Chained HotStuff Simulation                                        │  │  │
 │  │                                                                     │  │  │
 │  │  ┌─────────────────────────────────────────────────────────────┐    │  │  │
 │  │  │  Node Cluster                                               │    │  │  │
 │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │    │  │  │
 │  │  │  │ Honest   │  │ Honest   │  │ Honest   │  │ Byzantine  │ │    │  │  │
 │  │  │  │ Node 1   │  │ Node 2   │  │ Node N   │  │   Node(s)  │ │    │  │  │
 │  │  │  │          │  │          │  │          │  │            │ │    │  │  │
 │  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │    │  │  │
 │  │  └───────┼──────────────┼──────────────┼──────────────┼────────┘    │  │  │
 │  │          └──────────────┼──────────────┘              │             │  │  │
 │  │                    ┌────▼────┐                   ┌────▼──────────┐  │  │  │
 │  │                    │ Network │ <── disruption ── │  Byzantine    │  │  │  │
 │  │                    │   Sim   │    injection      │ Failure Agent │  │  │  │
 │  │                    │ Latency │                   │ 8 strategies  │  │  │  │
 │  │                    │ Jitter  │                   │ ADAPTIVE_     │  │  │  │
 │  │                    │ Loss    │                   │ MIRROR, FORK, │  │  │  │
 │  │                    └─────────┘                   │ EQUIVOCATION..│  │  │  │
 │  │                                                  └───────────────┘  │  │  │
 │  └─────────────────────────────────────────────────────────────────────┘  │  │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘

 Data Flow:
   Agent ──action──> API ──> Orchestrator ──> Simulation ──> Nodes execute round
   Nodes ──partial obs──> API ──obs+reward──> Agent
   BFA ──disruption──> Network Sim ──delayed/dropped msgs──> Honest Nodes
   Oracle ──safety check──> Simulation (terminates on violation)
   Curriculum ──advances difficulty──> BFA (escalates attack strategies)
```

### Consensus Protocol Simulator

The core simulation runs a **Chained HotStuff** consensus engine entirely in memory. Chained HotStuff is a pipelined BFT consensus protocol designed for high throughput in partially synchronous networks. The simulation implements:

- **QC-based pipelining**: each block carries a `justify` Quorum Certificate (QC) for the previous block, allowing proposal and voting to overlap across views.
- **3-chain commit rule**: a block is finalized only when a chain of three consecutive QC-linked blocks is observed, ensuring safety.
- **Safe node predicate (`locked_qc`)**: replicas only vote for blocks extending their locked QC, preventing equivocation from causing safety violations.
- **Per-node pacemaker**: each node independently tracks view timeouts (`view_timeout_ms`) and sends `NEW_VIEW` messages when it suspects a Byzantine or crashed leader.
- **Sync protocol**: nodes that fall behind by more than `SYNC_LAG_THRESHOLD` views request a sync from peers to catch up their committed log.
- **Byzantine strategy execution**: the adapter injects disruption payloads directly into the message layer per step, simulating realistic adversarial behavior.

### Network and Fault Simulation

Each curriculum level uses a `FaultProfile` that configures network latency (p50, p99), jitter, and packet loss. The simulated message bus applies these parameters per-tick (at `VIEW_TICK_MS = 50ms`) to all inter-node messages, creating realistic network conditions that the agent must adapt to.

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

- `RANDOM_DELAY`: injects 100-500ms delay on all messages from Byzantine nodes.
- `SELECTIVE_DELAY`: delays messages to half the honest nodes (200-800ms).
- `EQUIVOCATION`: sends conflicting proposals to two disjoint subsets of honest nodes.
- `LEADER_SUPPRESS`: suppresses all messages to honest nodes, stalling the current leader.
- `CASCADE_TIMING`: injects cascading delays (300-1000ms) timed to disrupt QC formation.
- `RECOVERY_DELAY`: targets nodes that just completed a view change (500-2000ms).
- `ADAPTIVE_MIRROR`: mirrors the minimum honest node `view_timeout_ms` with a delta offset to reliably trigger view changes.
- `FORK`: proposes conflicting blocks to force a fork attempt.

### Leader Rotation

Three rotation policies are supported, escalating with curriculum level:

- **Round Robin** (levels 1-6): deterministic rotation through node IDs.
- **VRF** (level 7): verifiable random function-based leader selection.
- **Reputation Weighted** (level 8): probabilistic selection weighted by historical commit success.

---

# Key Contributions

- **First OpenEnv environment for adaptive consensus tuning.** No existing RL environment targets the operator task of live BFT parameter tuning under adversarial conditions.
- **Reactive adversary with auto-advancing curriculum.** A Byzantine failure agent that escalates from simple delays to coordinated coalition attacks across eight difficulty levels. The adversary watches the agent and fights back, creating the non-stationary co-adaptation dynamic that makes RL necessary and static policies insufficient.
- **Built-in correctness oracle.** Every step is verified against formal safety invariants (Agreement, Validity, Liveness). An agent cannot score well through unsafe configurations — the episode terminates immediately on a safety violation.
- **Baseline comparison baked in.** Every episode is evaluated against static default parameters using the same adversary seed, directly answering: "Is the RL agent actually better than leaving the defaults alone?"
- **Reproducible and deterministic.** Seeded adversary selection enables exact episode replay for debugging, ablations, and fair cross-agent comparison.

---

# Related Work

The problem of adaptive consensus tuning under dynamic conditions is an active area of research. The following papers explore complementary approaches:

- **BFTBrain: Adaptive BFT Consensus with Reinforcement Learning** (Wu et al.) — Demonstrates that no single BFT protocol performs optimally across all conditions. Uses reinforcement learning to dynamically switch between multiple BFT protocols in real time, with a decentralized learning coordination mechanism resilient to adversarial data manipulation. [arXiv:2408.06432](https://arxiv.org/html/2408.06432v1)
- **Meta Reinforcement Learning Based Dynamic Tuning for Blockchain Systems in Diverse Network Environments** (Pei et al.) — Proposes MetaTune, a meta-RL framework that automatically discovers optimal blockchain parameter configurations as network bandwidth fluctuates. Tested on the ChainMaker platform, it shows that adaptive approaches significantly reduce the data samples needed to generalize across varying conditions compared to non-adaptive methods. [Elsevier](https://www.sciencedirect.com/science/article/pii/S2096720924000745)
- **An Adaptive Blockchain Framework for Federated IoMT with Reinforcement Learning-Based Consensus and Resource Forecasting** (Murthy & Shri) — Addresses real-world consensus challenges in healthcare IoT by combining an Adaptive Byzantine Fault Tolerance consensus protocol with Deep Q-Learning for resource allocation and anomaly detection, demonstrating the practical need for RL-driven consensus in latency-sensitive, adversarial environments. [PubMed](https://pubmed.ncbi.nlm.nih.gov/41667513/)

These works collectively validate the core premise behind Accordis: static consensus configurations are insufficient for real-world deployments, and reinforcement learning offers a principled path toward adaptive, self-tuning protocols.

---

# Core API

### Action - What the agent controls

A joint configuration update for all honest nodes each round, tuning five parameters per node — view timeout, pipeline depth, batch size, equivocation threshold, and vote aggregation timeout.

`MultiNodeAction` contains one `AccordisAction` per honest node. Each node action tunes five BFT knobs:

| Field                           | Bounds      | Default | Description                                             |
| ------------------------------- | ----------- | ------- | ------------------------------------------------------- |
| `view_timeout_ms`             | 200-3000 ms | 2000    | Leader timeout before triggering a view change          |
| `pipeline_depth`              | 1-8         | 2       | Number of in-flight proposal slots                      |
| `replication_batch_size`      | 1-512       | 64      | Transactions per proposed block                         |
| `equivocation_threshold`      | 1-15        | 5       | Conflicting-vote count before flagging a node Byzantine |
| `vote_aggregation_timeout_ms` | 50-1000 ms  | 500     | Max wait for votes before declaring a QC miss           |

All values are clamped to `SAFE_BFT_TUNING_BOUNDS` before reaching the adapter. `vote_aggregation_timeout_ms` is additionally constrained to `< view_timeout_ms // 2`.

Actions also accept optional `suspect_node` and `clear_suspicion` fields to directly signal Byzantine suspicion.

### Observation - What the agent sees

Per-node health metrics — phase latencies, vote miss streaks, view change frequency, peer suspicion flags, throughput, and current configuration. The agent never sees the full cluster state or the adversary's strategy. It reasons under the same partial information a human operator would have.

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

### Reward - What the agent optimizes

A blend of nine signals balancing throughput (block commits, TPS improvement over static defaults), safety (penalties for unnecessary disruptions, liveness stalls), and resilience (bonuses for fast recovery from attacks without safety violations). High throughput achieved by cutting safety margins is explicitly penalized — the agent cannot game the reward by being reckless.

| Signal                      | Value | When                                                  |
| --------------------------- | ----- | ----------------------------------------------------- |
| `liveness_cost`           | -1.0  | Every step while pending txns exist                   |
| `block_commit`            | +60   | Per block's worth of txns finalized (3-chain commit)  |
| `unnecessary_view_change` | -150  | View change with no Byzantine activity                |
| `liveness_stall`          | -30   | No new commits for > 80 steps                         |
| `fast_leader_recovery`    | +120  | View change under attack, liveness recovers           |
| `false_positive_avoided`  | +40   | Attack contained without agreement/validity violation |
| `pipeline_efficiency`     | +15   | Episode-end: avg pipeline utilisation above threshold |
| `throughput_improvement`  | +20   | Episode-end: TPS beats static baseline                |
| `latency_improvement`     | +10   | Episode-end: view-change overhead below baseline      |

### Curriculum - How the agent is challenged

Difficulty auto-advances across eight levels as the agent demonstrates sustained performance. At level 1, there are no Byzantine nodes. By level 8, the agent faces a full coalition attack combining leader suppression, cascade timing, equivocation, and fork attempts — with reputation-weighted leader rotation replacing deterministic round-robin. An agent that reaches level 8 has demonstrated sustained liveness across 350+ episodes of progressively harder adversarial conditions.

### Termination

Episodes terminate when (in priority order):

1. **Pool drained** — all transactions committed (success).
2. **Agreement violated** — two honest nodes committed conflicting blocks.
3. **Validity violated** — a committed transaction was not in the honest proposal set.
4. **Max steps reached** — step budget exhausted.

---

## Tasks

Three benchmark tasks span eight curriculum levels, designed to progressively challenge the agent from benign conditions to full adversarial coalitions:

### Easy (`easy`)

- **Curriculum levels**: 1-2
- **Nodes**: 4 honest, 0 (level 1) / 1 (level 2) Byzantine
- **Strategy**: `NONE` / `RANDOM_DELAY`
- **Leader rotation**: Round Robin
- **Max steps**: 50  `(Max steps have been reduced to 40 to mitigate the errors while running the inference.py submission in phase 2 validation)`
- **Grader**: `0.5 * liveness_rate + 0.3 * max(0, 1 - vc_count/5) + 0.2 * correctness`

### Medium (`medium`)

- **Curriculum levels**: 3-5
- **Nodes**: 7 honest, 2 Byzantine (1 crash + 1 active)
- **Strategy**: `SELECTIVE_DELAY` / `EQUIVOCATION` / `ADAPTIVE_MIRROR`
- **Leader rotation**: Round Robin
- **Max steps**: 100 ` (Since this is a long running task with multiple trajectories, Max steps have been reduced to 40 to mitigate the errors while running the inference.py submission in phase 2 validation)`
- **Grader**: `0.4 * liveness_rate + 0.2 * max(0, 1 - vc_count/10) + 0.2 * recovery_bonus + 0.2 * correctness`

### Hard (`hard`)

- **Curriculum levels**: 6-8
- **Nodes**: 10 honest, 2 Byzantine (coordinated attack)
- **Strategy**: `LEADER_SUPPRESS + SELECTIVE_DELAY` (lvl 6), `CASCADE_TIMING + EQUIVOCATION` (lvl 7), full coalition (lvl 8)
- **Leader rotation**: Round Robin (lvl 6) -> VRF (lvl 7) -> Reputation Weighted (lvl 8)
- **Pool size**: 1800 transactions
- **Max steps**: 100 ` (Since this is a long running task with multiple trajectories, Max steps have been reduced to 40 to mitigate the errors while running the inference.py submission in phase 2 validation)`
- **Grader**: `0.05 * liveness_rate + 0.75 * throughput_score + 0.10 * vc_penalty + 0.10 * correctness`
- **Expected scores at level 6**: static defaults ~0.26, median LLM ~0.42, expert agent ~0.76

---

# Repository Layout

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

---

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
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
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

---

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

---

## HTTP and WebSocket Behavior

- `POST /reset` works over HTTP and returns an initial observation.
- `POST /step` over plain HTTP is not useful for multi-step episodes — each request creates a fresh environment instance.
- `/ws` is the intended interface for interactive episodes; the environment session persists across `reset`, `step`, and `state` messages.

If building an agent loop, prefer the Python client or direct WebSocket usage.

---

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

`inference.py` can connect either to a Docker image via `LOCAL_IMAGE_NAME=accordis` or to an already-running server via `ACCORDIS_BASE_URL=http://localhost:8000`. If Docker is unavailable, prefer the base URL mode after starting the server locally.

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

---

## Testing

```bash
uv run pytest
```

The test suite covers app endpoints, WebSocket flow, models and validation, reward logic, oracle/verifier behavior, the simulated adapter, and task graders.

---

## Docker

```bash
docker build -t accordis:latest .
docker run --rm -p 8000:8000 --env-file .env accordis:latest
```

The image starts `uvicorn accordis.server.app:app` on port `8000`.

---

## Deploying with OpenEnv

```bash
openenv push
openenv push --private
openenv push --repo-id <namespace>/<repo>
```

---

## Future Scope

- **Broader protocol support** — extend the adapter to PBFT, Tendermint, or DAG-based consensus (Narwhal/Tusk) to study whether tuning policies transfer across protocols.
- **Learned adversaries** — train a second RL agent as the attacker, creating a two-player game where both sides improve simultaneously.
- **Live testnet integration** — replace the simulated adapter with a connector to a real testnet (e.g., CometBFT, Ethereum Beacon Chain) to validate sim-to-real transfer.
- **Dynamic topologies** — simulate nodes joining, leaving, and network partitions forming, testing adaptation to structural changes beyond adversarial behavior.
