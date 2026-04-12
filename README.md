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

> **Imagine a blockchain network processing thousands of financial transactions per second. A coordinated group of compromised validators begins injecting conflicting proposals. Within seconds, honest nodes disagree on the canonical chain, throughput collapses, and the system stalls — not because the protocol was broken, but because its configuration parameters were tuned for calm weather, not a storm.**

> This is not a hypothetical. Networks running BFT consensus protocols routinely face performance degradation — sometimes catastrophic — when real-world conditions deviate from the assumptions baked into their static configurations. Accordis exists to fix that.

Accordis is an [OpenEnv](https://openenv.dev) reinforcement learning environment with a reactive adversary and auto-advancing curriculum for Byzantine Fault Tolerant (BFT) consensus tuning. An RL agent learns to configure per-node BFT protocol parameters across a cluster of honest nodes while a Byzantine adversary actively tries to disrupt consensus.

---

## Why This Problem Matters

Consensus protocols are the backbone of systems where trust cannot be assumed:

- **Blockchains and DeFi** — every transaction settlement, every smart contract execution, every cross-chain bridge depends on nodes reaching agreement under adversarial conditions.
- **Financial infrastructure** — stock exchanges, payment networks, and clearinghouses use replicated state machines to guarantee that no single failure can corrupt the ledger.
- **Critical coordination services** — distributed databases, cloud control planes, and military command systems rely on consensus to maintain a single source of truth across geographically dispersed nodes.

When consensus fails, the consequences are immediate and severe: double-spends, frozen assets, data corruption, or complete system unavailability. A single misconfigured timeout parameter can cascade into a network-wide liveness stall affecting millions of users.

The core issue is that **production consensus deployments overwhelmingly rely on static parameter tuning** — operators pick timeout values, batch sizes, and fault thresholds based on benchmarks run under controlled conditions, then deploy those fixed configurations into environments that are anything but controlled. Network latency fluctuates. Nodes crash and recover unpredictably. Adversaries adapt their strategies in real time. A timeout that works perfectly at 50ms network latency becomes a liability at 500ms, and a throughput-optimized batch size under normal load becomes a bottleneck during congestion.

The gap between static tuning and dynamic reality is where systems break. Accordis targets that gap directly.

---

## The Consensus Problem (Intuition)

At its core, consensus is deceptively simple: get a group of computers to agree on the same sequence of events, even when some of them might be lying.

Think of it like a group of generals coordinating an attack plan by sending messengers between their camps. They need to agree on a single plan — attack or retreat — but some generals might be traitors sending contradictory messages. The challenge is not just reaching agreement; it is reaching agreement when you cannot trust everyone at the table.

What makes this genuinely hard:

- **You cannot tell who is lying.** A malicious node can send different messages to different peers, and there is no central authority to arbitrate.
- **The network is unreliable.** Messages get delayed, reordered, or dropped entirely. A slow node looks identical to a dead node, which looks identical to a compromised one.
- **Speed and safety are in tension.** Wait too long for votes, and throughput dies. Move too fast, and you commit based on incomplete information, opening the door to forks and inconsistencies.
- **Adversaries are adaptive.** A real attacker does not follow a fixed playbook — they observe the system's behavior and exploit its tuning. If your timeout is 2 seconds, they will stall you for 2.1.

This is why getting the protocol *parameters* right matters as much as getting the protocol *logic* right. A perfectly correct BFT protocol with poorly tuned timeouts will still stall, fork, or hemorrhage throughput in adversarial conditions.

---

## Byzantine Fault Tolerance & HotStuff

### Byzantine Faults

A Byzantine fault is the worst kind of failure a distributed system can face. Unlike a crash (where a node simply stops responding), a Byzantine node can behave *arbitrarily* — sending conflicting messages, selectively withholding votes, or colluding with other compromised nodes to undermine the protocol.

BFT protocols guarantee two properties even in the presence of up to *f* Byzantine nodes (out of *3f + 1* total):

- **Safety** — honest nodes never commit conflicting decisions. If node A finalizes block X at height 10, no honest node will ever finalize a different block at height 10.
- **Liveness** — the system continues to make progress. Transactions eventually get committed, even if Byzantine nodes are actively trying to stall the protocol.

### HotStuff

HotStuff is a leader-based BFT consensus protocol designed for high throughput in partially synchronous networks. Its key innovation is *pipelined* consensus: instead of completing all phases for one block before starting the next, HotStuff overlaps proposal and voting across consecutive views, dramatically increasing throughput.

Accordis simulates **Chained HotStuff**, which uses a 3-chain commit rule — a block is only finalized when three consecutive Quorum Certificates (QCs) are linked in a chain. This provides strong safety guarantees but introduces a critical dependency: if *any* of the three leaders in the chain is Byzantine or slow, the commit pipeline stalls.

**The fundamental insight**: even a protocol as well-designed as HotStuff has performance that is *extremely sensitive* to its configuration parameters. The view timeout determines how quickly the system detects a failed leader — too short and you trigger unnecessary view changes that waste bandwidth and reset the commit pipeline; too long and a Byzantine leader can stall the system for the full timeout duration. The vote aggregation window, pipeline depth, batch size, and equivocation threshold all interact in complex, nonlinear ways that defy manual tuning under dynamic conditions.

This is where reinforcement learning enters the picture.

---

## What Is It and Why Was It Built

### The Problem

Modern distributed systems running consensus protocols — blockchains, coordination services, replicated state machines — expose a set of configuration knobs (timeouts, pipeline depths, batch sizes, fault thresholds) that must be tuned correctly to maintain safety and liveness under adversarial network conditions. In practice, operators set these parameters once based on controlled benchmarks and never touch them again. This static approach creates a fundamental brittleness:

- When network conditions degrade, fixed timeouts either trigger cascading unnecessary view changes (destroying throughput) or fail to detect Byzantine leaders (allowing prolonged stalls).
- When adversaries adapt their strategies — timing attacks to land just outside aggregation windows, equivocating to exploit fixed thresholds — static configurations offer no counter-adaptation.
- When the system scales or load patterns shift, batch sizes and pipeline depths optimized for one regime become bottlenecks in another.

The result is a widening gap between protocol *correctness* (which BFT guarantees formally) and protocol *performance* (which depends entirely on parameter tuning that no one maintains).

### The Research Gap

Existing approaches to this problem fall into two camps: manual tuning by expert operators (expensive, slow, does not adapt), and rule-based heuristics (slightly better but still fundamentally reactive and unable to anticipate adversarial counter-strategies). What is missing is a system that can *learn* optimal tuning policies from experience, adapt in real time to changing conditions, and generalize across attack strategies it has never seen before.

### Our Solution

Accordis frames BFT parameter tuning as a reinforcement learning problem. Rather than prescribing fixed configurations, a trained agent observes per-node partial views of the cluster each round and outputs joint configuration updates for all honest nodes. The environment simultaneously runs a Byzantine adversary that escalates its attack strategy as the curriculum difficulty increases, forcing the agent to develop robust, adaptive policies rather than memorizing responses to fixed scenarios.

The goal is to train agents that can:

- Maintain liveness and commit throughput under Byzantine network conditions.
- Recover quickly from leader failures and view timeouts without over-triggering spurious view changes.
- Contain equivocation and fork attacks through appropriate threshold configuration.
- Outperform a static baseline configuration on throughput and latency — consistently, across varying conditions.

---

## Why Reinforcement Learning?

BFT parameter tuning is fundamentally a **sequential decision-making problem under uncertainty**. Each configuration choice affects future system states in complex, delayed ways — a timeout change at step *t* determines whether the system can recover from a leader failure at step *t + 5*, which in turn affects whether the commit pipeline stalls at step *t + 10*.

Static heuristics and lookup tables cannot capture these temporal dependencies. They react to the current state without considering downstream consequences. More importantly, they cannot *learn* — when an adversary changes its strategy, a heuristic continues applying the same rules until a human operator manually intervenes.

Reinforcement learning is a natural fit because:

- **The environment is partially observable.** Each node sees only its own local metrics — latencies, QC miss streaks, peer suspicion signals. The agent must infer cluster-wide health from these fragmented views, a task that RL handles through learned representations.
- **The adversary is adaptive.** As the agent tunes more precisely, the adversary calibrates its attacks to exploit the new configuration. This creates a non-stationary environment where policies must continuously adapt — exactly the setting where RL outperforms fixed rules.
- **The reward signal is rich but delayed.** The payoff of a good configuration choice may not materialize for many rounds. RL's credit assignment mechanisms are designed for exactly this kind of delayed, multi-signal feedback.
- **The action space has complex interactions.** Five configuration knobs per node, with constraints between them (e.g., vote aggregation timeout must be less than half the view timeout), create a high-dimensional action space where exhaustive search is infeasible but gradient-based policy optimization is tractable.

---

## Related Work

The problem of adaptive consensus tuning under dynamic conditions is an active area of research. The following papers explore complementary approaches:

- **BFTBrain: Adaptive BFT Consensus with Reinforcement Learning** (Wu et al.) — Demonstrates that no single BFT protocol performs optimally across all conditions. Uses reinforcement learning to dynamically switch between multiple BFT protocols in real time, with a decentralized learning coordination mechanism resilient to adversarial data manipulation. [arXiv:2408.06432](https://arxiv.org/html/2408.06432v1)

- **Meta Reinforcement Learning Based Dynamic Tuning for Blockchain Systems in Diverse Network Environments** (Pei et al.) — Proposes MetaTune, a meta-RL framework that automatically discovers optimal blockchain parameter configurations as network bandwidth fluctuates. Tested on the ChainMaker platform, it shows that adaptive approaches significantly reduce the data samples needed to generalize across varying conditions compared to non-adaptive methods. [Elsevier](https://www.sciencedirect.com/science/article/pii/S2096720924000745)

- **An Adaptive Blockchain Framework for Federated IoMT with Reinforcement Learning-Based Consensus and Resource Forecasting** (Murthy & Shri) — Addresses real-world consensus challenges in healthcare IoT by combining an Adaptive Byzantine Fault Tolerance consensus protocol with Deep Q-Learning for resource allocation and anomaly detection, demonstrating the practical need for RL-driven consensus in latency-sensitive, adversarial environments. [PubMed](https://pubmed.ncbi.nlm.nih.gov/41667513/)

These works collectively validate the core premise behind Accordis: static consensus configurations are insufficient for real-world deployments, and reinforcement learning offers a principled path toward adaptive, self-tuning protocols.

---

## Our Approach

Accordis structures the RL problem as follows:

**State (what the agent observes):** Per-node partial views including phase latencies, QC miss streaks, view change frequency, peer suspicion flags, throughput metrics, and the node's current configuration. The agent never sees the global state, the adversary's chosen strategy, or the full message log — it must reason under partial observability.

**Action (what the agent controls):** A joint configuration update for all honest nodes each round, tuning five parameters per node — view timeout, pipeline depth, batch size, equivocation threshold, and vote aggregation timeout. All actions are clamped to safe bounds before reaching the consensus engine.

**Reward (what the agent optimizes):** A blend of nine signals that balance throughput (block commits, TPS improvement), safety (penalties for unnecessary view changes, liveness stalls), and resilience (bonuses for fast leader recovery and containing Byzantine attacks without safety violations). The reward is shaped to make the agent prefer configurations that are both *performant* and *safe* — high throughput achieved by cutting safety margins is explicitly penalized.

---

## Key Contributions

- **RL-based adaptive BFT tuning.** The first OpenEnv environment that frames consensus parameter tuning as a reinforcement learning task, enabling agents to learn adaptive policies that outperform static configurations under adversarial conditions.
- **Reactive adversary with auto-advancing curriculum.** A Byzantine failure agent that escalates from simple delays to coordinated coalition attacks across eight difficulty levels, with one strategy (`ADAPTIVE_MIRROR`) that directly counter-adapts to the agent's live configuration.
- **Realistic BFT modeling.** A full Chained HotStuff simulation with QC-based pipelining, 3-chain commits, per-node pacemakers, sync protocols, and calibrated network fault profiles — not a simplified toy model.
- **Built-in correctness verification.** A correctness oracle that enforces Agreement, Validity, and Liveness invariants every step, ensuring that no agent can achieve high scores through unsafe configurations.
- **OpenEnv-compatible and reproducible.** Standard OpenEnv API integration with seeded adversary selection, enabling exact episode replay for debugging, ablations, and fair comparison across agents.

---

## Key Characteristics

**Auto-advancing curriculum.** The environment tracks agent performance over a rolling 50-episode window and automatically advances difficulty (levels 1-8) when the agent's liveness rate sustains above 85%. No manual configuration or external scheduler is needed — the environment promotes itself as the agent improves.

**Reactive adversary.** One Byzantine failure strategy (`ADAPTIVE_MIRROR`) reads the agent's live per-node configuration each step and calibrates its disruption timing to land just after the agent's vote aggregation window closes. The adversary tightens its attack as the agent tunes more precisely, creating a continuous pressure signal rather than a fixed obstacle.

**Partial observability under adversarial conditions.** Each honest node sees only its own local metrics — phase latencies, QC miss streaks, peer suspicion signals, throughput. The agent must infer cluster-wide health and Byzantine activity from these fragmented views while tuning five configuration knobs per node per round.

**Built-in correctness verification.** A correctness oracle runs Agreement, Validity, and Liveness checks against the full hidden state every step. Safety violations terminate the episode immediately with zero score, making the environment intolerant of unsafe configurations regardless of throughput gains.

**Reproducible adversarial episodes.** All Byzantine strategy selection is seeded, so every episode can be exactly replayed for debugging, ablations, or fair multi-agent comparison.

---

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

## Core API

### Action

`MultiNodeAction` contains one `AccordisAction` per honest node. Each node action tunes five BFT knobs:

| Field                           | Bounds       | Default | Description                                             |
| ------------------------------- | ------------ | ------- | ------------------------------------------------------- |
| `view_timeout_ms`             | 200-3000 ms | 2000    | Leader timeout before triggering a view change          |
| `pipeline_depth`              | 1-8         | 2       | Number of in-flight proposal slots                      |
| `replication_batch_size`      | 1-512       | 64      | Transactions per proposed block                         |
| `equivocation_threshold`      | 1-15        | 5       | Conflicting-vote count before flagging a node Byzantine |
| `vote_aggregation_timeout_ms` | 50-1000 ms  | 500     | Max wait for votes before declaring a QC miss           |

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

Per-step reward blends nine signals designed to balance throughput, safety, and resilience:

| Signal                      | Value | When                                                  |
| --------------------------- | ----- | ----------------------------------------------------- |
| `liveness_cost`           | -1.0 | Every step while pending txns exist                   |
| `block_commit`            | +60   | Per block's worth of txns finalized (3-chain commit)  |
| `unnecessary_view_change` | -150 | View change with no Byzantine activity                |
| `liveness_stall`          | -30  | No new commits for > 80 steps                         |
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

---

## Tasks

Three benchmark tasks span eight curriculum levels, designed to progressively challenge the agent from benign conditions to full adversarial coalitions:

### Easy (`easy`)

- **Curriculum levels**: 1-2
- **Nodes**: 4 honest, 0 (level 1) / 1 (level 2) Byzantine
- **Strategy**: `NONE` / `RANDOM_DELAY`
- **Leader rotation**: Round Robin
- **Max steps**: 50 (Since this is a long running task, Max steps have been reduced to 40 to mitigate the errors while running the inference.py submission in phase 2 validation)
- **Grader**: `0.5 * liveness_rate + 0.3 * max(0, 1 - vc_count/5) + 0.2 * correctness`

### Medium (`medium`)

- **Curriculum levels**: 3-5
- **Nodes**: 7 honest, 2 Byzantine (1 crash + 1 active)
- **Strategy**: `SELECTIVE_DELAY` / `EQUIVOCATION` / `ADAPTIVE_MIRROR`
- **Leader rotation**: Round Robin
- **Max steps**: 100 (Since this is a long running task, Max steps have been reduced to 40 to mitigate the errors while running the inference.py submission in phase 2 validation)
- **Grader**: `0.4 * liveness_rate + 0.2 * max(0, 1 - vc_count/10) + 0.2 * recovery_bonus + 0.2 * correctness`

### Hard (`hard`)

- **Curriculum levels**: 6-8
- **Nodes**: 10 honest, 2 Byzantine (coordinated attack)
- **Strategy**: `LEADER_SUPPRESS + SELECTIVE_DELAY` (lvl 6), `CASCADE_TIMING + EQUIVOCATION` (lvl 7), full coalition (lvl 8)
- **Leader rotation**: Round Robin (lvl 6) -> VRF (lvl 7) -> Reputation Weighted (lvl 8)
- **Pool size**: 1800 transactions
- **Max steps**: 100 (Since this is a long running task, Max steps have been reduced to 40 to mitigate the errors while running the inference.py submission in phase 2 validation)
- **Grader**: `0.05 * liveness_rate + 0.75 * throughput_score + 0.10 * vc_penalty + 0.10 * correctness`
- **Expected scores at level 6**: static defaults ~0.26, median LLM ~0.42, expert agent ~0.76

---

## Evaluation & Validation

Accordis validates agent performance through a multi-layered evaluation pipeline that checks correctness, measures performance, and compares against baselines — ensuring that high scores reflect genuinely robust tuning, not lucky runs or unsafe shortcuts.

### Correctness Oracle (Every Step)

A `CorrectnessOracle` runs three formal verification checks against the full hidden state after every environment step:

- **Agreement** — For every committed log slot, all honest nodes must share the same block hash. A disagreement means two honest nodes finalized conflicting blocks — a fundamental safety violation that immediately terminates the episode with zero score.
- **Validity** — Every committed transaction must appear in the original honest proposal set. If a transaction was never proposed by an honest node but appears in the committed log, the episode terminates immediately.
- **Liveness** — The ratio of finalized transactions to submitted transactions. This is not a pass/fail check but a continuous metric that feeds into both the reward signal and the task grader.

These checks are adapter-agnostic and operate solely on `AccordisState`, meaning they work identically regardless of the underlying consensus simulation.

### Per-Step Reward Verification

The reward calculator cross-references nine independent signals (detailed in the Reward section above) against the oracle's output. Key design choices that prevent reward gaming:

- **Unnecessary view changes are heavily penalized (-150)** even if throughput remains high, preventing agents from learning "aggressive timeout" policies that happen to work under benign conditions.
- **Block commits are counted only at 3-chain finalization**, not at QC formation, so the reward cannot overstate progress relative to the oracle's committed log.
- **False positive avoidance (+40)** is only awarded when a Byzantine attack was actively occurring AND no safety violation resulted — the agent cannot earn this bonus by simply doing nothing.

### Task-Level Grading

Each task defines a deterministic grading function that produces a scalar score in [0.0, 1.0] from the full episode log. The grader weights shift across difficulty levels to reflect changing priorities:

- **Easy** emphasizes liveness (50%) and view change discipline (30%) — can the agent keep the system running without triggering spurious timeouts?
- **Medium** introduces recovery quality (20%) — can the agent detect and recover from Byzantine leaders efficiently?
- **Hard** pivots to throughput efficiency (75%) — given that the pool will eventually drain regardless, how *fast* can the agent push transactions through under coordinated attack?

Grader scores are clamped to [0.01, 0.99] to ensure meaningful discrimination across the score range.

### Baseline Comparison

The oracle computes a `BaselineComparison` that re-evaluates episode performance against a static default configuration (batch=64, VAT=500ms, timeout=2000ms) using the same adversary seed. This provides three relative metrics:

- **Relative TPS improvement** — did the agent commit transactions faster than static defaults?
- **Relative latency improvement** — did the agent reduce view-change overhead?
- **View change count comparison** — did the agent trigger fewer unnecessary view changes?

These comparisons feed into the `throughput_improvement` and `latency_improvement` reward signals and give a concrete answer to: *"Is the RL agent actually better than just leaving the defaults alone?"*

### Curriculum Advancement as Validation

The auto-advancing curriculum itself serves as a validation mechanism. An agent only advances from level *n* to level *n+1* when its rolling 50-episode liveness average exceeds 85%. This means:

- A level-8 agent has *demonstrated* sustained liveness across 350+ episodes of progressively harder adversarial conditions.
- Advancement clears the rolling window, preventing an agent from coasting on stale performance.
- The curriculum cannot be skipped — there is no way to reach level 8 without passing through levels 1-7.

### Reproducibility

All Byzantine strategy selection is seeded via a PRNG, so any episode can be exactly replayed with identical adversary behavior. This enables:

- **Fair comparison** — two different agents evaluated on the same seed face identical attack sequences.
- **Ablation studies** — isolate the effect of a single parameter change by holding the adversary constant.
- **Debugging** — replay a failed episode step-by-step to identify exactly where the agent's configuration choice led to a safety violation or stall.

---

## Future Scope

Accordis in its current form is a complete RL environment for adaptive BFT tuning, but several directions can extend its impact and realism significantly.

### Broader Protocol Support

The environment currently simulates Chained HotStuff. Extending the adapter interface to support additional BFT protocols — such as PBFT, Tendermint, or DAG-based consensus (Narwhal/Tusk, Bullshark) — would allow researchers to study whether learned tuning policies transfer across protocols or whether each protocol requires its own specialized policy. This directly addresses the open question of whether RL-based tuning is protocol-specific or captures general principles of consensus performance.

### Dynamic Network Topologies

The current fault profiles simulate latency and packet loss but assume a fixed, fully connected network topology. Real-world validator networks experience **topology changes** — nodes join and leave, network partitions form and heal, geographic routing shifts. Adding dynamic topology simulation would test whether agents can adapt not just to adversarial behavior but to structural changes in the network itself.

### Richer Adversary Models

The Byzantine failure agent currently selects from eight predefined strategies. Future work could introduce:

- **Learned adversaries** — train a second RL agent as the adversary, creating a two-player game where both the tuner and the attacker improve simultaneously.
- **Economic adversaries** — model attackers with cost constraints (e.g., bribing validators costs resources), introducing game-theoretic considerations into the tuning problem.
- **Colluding adversary coalitions** — allow Byzantine nodes to share state and coordinate strategies dynamically rather than drawing from a fixed pool.

### Transfer to Live Testnets

The ultimate validation of this approach is deployment on a real consensus network. A clear path forward is:

1. **Testnet integration** — replace the simulated adapter with one that connects to a live testnet (e.g., a private Ethereum Beacon Chain or CometBFT network) and applies configuration updates to real validator nodes.
2. **Safety constraints in production** — design action filtering and rollback mechanisms that prevent the RL agent from applying configurations that could compromise a live network's safety guarantees.
---

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
