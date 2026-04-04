

---
name: accordis
description: "Accordis is an OpenEnv reinforcement learning environment for tuning and evaluating Byzantine fault-tolerant consensus protocols under adversarial network conditions."
---
The OpenEnv CLI command `openenv` is available.
Use `openenv --help` to view available commands.

# SKILL.md — AccordisEnvironment: Self-Adaptive BFT Consensus Tuning Environment

> **For the coding agent:** Before implementing anything, read the OpenEnv SDK reference at:
> `.agents/skills/openenv-cli/OPENENV_REFERENCE.md`
> It is the authoritative source for `Environment`, `Observation`, `Action`, `Reward`,
> `Transform`, `Rubric`, and the `openenv.yaml` schema. Every OpenEnv parent class used
> in this project must be imported from the SDK. Do not create parallel base classes.
>
> Confirmed OpenEnv imports:
>
> ```
> from openenv.core import EnvClient
> from openenv.core.client_types import StepResult
> from openenv.core.env_server.types import State, Action, Observation
> from openenv.core.env_server.http_server import create_app
> from openenv.core.env_server.interfaces import Environment, Transform
> from openenv.core.rubrics import Rubric
> ```

---

## Environment Identity

| Field               | Value                                               |
| ------------------- | --------------------------------------------------- |
| Environment ID      | `AccordisEnvironment`                             |
| Formulation         | POMDP — partially observable, adversarial          |
| Consensus Engine    | Pluggable via `BaseConsensusAdapter`              |
| Version 1 Engine    | In-memory HotStuff simulator (zero dependencies)    |
| Version 2 Engine    | LibraBFT (HotStuff,`diem/diem`, Rust)             |
| Network Injection   | Version 1: synthetic Pareto latency model           |
|                     | Version 2: Toxiproxy (real TCP-level faults)        |
| Byzantine Injection | Version 1: internal BFA state machine               |
|                     | Version 2: Twins methodology (LibraBFT native)      |
| Crypto Verification | Disabled in both versions — vote-count quorum stub |
| OpenEnv Compliant   | Yes — RLVE verifiable reward oracle                |

---

## Canonical Folder Structure

```
accordis/                               ← project root (openenv init accordis)
├── README.md
├── Dockerfile
├── __init__.py
├── client.py
├── llm_factory.py
├── inference.py
├── models.py                           ← ALL Pydantic models. Single source of truth.
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── openenv_accordis.egg-info/
├── outputs/
│   └── logs/
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_environment.py             ← parameterised; runs against BOTH adapters
│   ├── test_verifier.py                ← adapter-agnostic
│   ├── test_tasks.py                   ← adapter-agnostic
│   ├── test_reward.py                  ← adapter-agnostic
│   ├── test_simulated_adapter.py       ← Version 1 adapter internals only
│   └── test_librabft_adapter.py        ← Version 2 only (skipped if binary absent)
└── server/
    ├── __init__.py
    ├── app.py                          ← FastAPI entry point; uses create_adapter()
    ├── accordis_environment.py         ← AccordisEnvironment — NEVER imports an adapter impl
    ├── requirements.txt
    ├── adapters/                       ← THE ONLY VERSION-SPECIFIC SUBTREE
    │   ├── __init__.py                 ← create_adapter() factory lives here
    │   ├── base.py                     ← BaseConsensusAdapter interface definition
    │   ├── simulated/                  ← Version 1 implementation
    │   │   ├── __init__.py
    │   │   ├── adapter.py              ← SimulatedConsensusAdapter
    │   │   ├── hotstuff_sim.py         ← In-memory HotStuff phase engine
    │   │   ├── network_sim.py          ← Synthetic Pareto latency/loss model
    │   │   └── bfa_sim.py              ← In-process Byzantine behavior injector
    │   └── librabft/                   ← Version 2 implementation
    │       ├── __init__.py
    │       ├── adapter.py              ← LibraBFTAdapter
    │       ├── grpc_client.py          ← gRPC management stubs
    │       ├── toxiproxy_client.py     ← Toxiproxy REST API wrapper
    │       └── twins_client.py         ← Twins Byzantine injection gRPC wrapper
    ├── network/
    │   ├── __init__.py
    │   └── fault_profiles.py           ← FaultProfile definitions — shared by both versions
    ├── adversary/
    │   ├── __init__.py
    │   └── bfa.py                      ← ByzantineFailureAgent — shared, adapter-agnostic
    ├── oracle/
    │   ├── __init__.py
    │   └── verifier.py                 ← CorrectnessOracle — shared, adapter-agnostic
    ├── curriculum/
    │   ├── __init__.py
    │   └── manager.py                  ← CurriculumManager — shared
    ├── tasks/
    │   ├── __init__.py
    │   ├── base_task.py
    │   ├── task_easy.py
    │   ├── task_medium.py
    │   └── task_hard.py
    └── rewards/
        ├── __init__.py
        └── reward_calculator.py        ← RewardCalculator — shared, adapter-agnostic
```

**Shared (identical in both versions, never modified when switching):**
`models.py`, `accordis_environment.py`, `verifier.py`, `reward_calculator.py`,
`bfa.py`, `manager.py`, `fault_profiles.py`, `base_task.py`, `task_*.py`,
`llm_factory.py`, `inference.py`, `openenv.yaml`

**Version-specific (only the adapter subtree changes):**
`server/adapters/simulated/` ↔ `server/adapters/librabft/`

---

## Architecture Overview

The system is organised into four strictly separated layers. Each layer depends only on
the layer immediately below it through a defined interface. No layer reaches across.

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — OpenEnv API Layer                                         │
│  app.py → AccordisEnvironment.reset() / step() / state()            │
│  Exposes the environment to agents and the OpenEnv validator.        │
│  Depends on: Layer 2 via BaseConsensusAdapter interface only.        │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  BaseConsensusAdapter (the only seam)
┌─────────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Environment Logic Layer                                   │
│  AccordisEnvironment, ByzantineFailureAgent, CorrectnessOracle,      │
│  RewardCalculator, CurriculumManager, Tasks, Rubric                  │
│  Contains all BFT protocol reasoning and RL reward logic.            │
│  Has ZERO knowledge of LibraBFT, Toxiproxy, gRPC, or simulation.    │
│  Depends on: Layer 3 via BaseConsensusAdapter interface only.        │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  BaseConsensusAdapter
               ┌──────────────┴──────────────┐
               │                             │
┌──────────────▼──────────────┐ ┌────────────▼──────────────────────────┐
│  LAYER 3A — Version 1       │ │  LAYER 3B — Version 2                  │
│  SimulatedConsensusAdapter  │ │  LibraBFTAdapter                       │
│  + hotstuff_sim.py          │ │  + grpc_client.py                      │
│  + network_sim.py           │ │  + toxiproxy_client.py                 │
│  + bfa_sim.py               │ │  + twins_client.py                     │
│  In-process, no I/O         │ │  Subprocesses + gRPC + TCP proxies     │
└──────────────┬──────────────┘ └────────────┬──────────────────────────┘
               │                             │
┌──────────────▼──────────────┐ ┌────────────▼──────────────────────────┐
│  LAYER 4A — Simulation      │ │  LAYER 4B — Real System                │
│  In-memory node objects     │ │  LibraBFT OS processes (Rust)          │
│  Message queues + clocks    │ │  Toxiproxy TCP proxies                 │
│  Zero external dependencies │ │  Twins injection API                   │
└─────────────────────────────┘ └────────────────────────────────────────┘
```

**The seam is `BaseConsensusAdapter`.** All of Layer 2 calls only methods on this
interface. Swapping the adapter at Layer 3 requires zero changes anywhere in Layer 2.

---

## Section 1 — Adapter Interface (BaseConsensusAdapter)

**File:** `server/adapters/base.py`

This is the **only** contract that `AccordisEnvironment` and all Layer 2 components are
permitted to call. No Layer 2 file may import from `server/adapters/simulated/` or
`server/adapters/librabft/`. Violations of this import rule break the swappability guarantee.

### Lifecycle Methods

```
start_cluster(n_nodes: int, f_byzantine: int, leader_rotation: LeaderRotation) → List[NodeID]
    Start a fresh cluster of n_nodes. Designate f_byzantine nodes as Byzantine.
    Return the ordered list of all node IDs (honest first, Byzantine last).
    Must be callable multiple times — each call fully replaces the prior cluster.

stop_cluster() → None
    Tear down the running cluster and release all resources.
    Must be idempotent — safe to call when no cluster is running.
```

### Configuration Methods

```
apply_config(node_id: NodeID, config: BFTConfig) → None
    Write a complete BFTConfig to a single honest node.
    Takes effect before the next call to advance_one_step().
    Byzantine node IDs must never be passed to this method.

configure_network(curriculum_level: int, node_ids: List[NodeID]) → None
    Apply the FaultProfile for the given curriculum level to all inter-node links.
    Called once per reset(). May also be called mid-episode for regime-switch levels.
    Reads from fault_profiles.get_fault_profile(curriculum_level) internally.
```

### Execution Method

```
advance_one_step() → None
    Tick the cluster forward by exactly one logical timestep.
    On return, the following are guaranteed to have occurred:
      - All pending messages with delivery_tick <= current_tick are processed.
      - View timeouts have fired for any node that exceeded view_timeout_ms.
      - QC formation has been attempted. Accepted on 2f+1 votes (no crypto check).
      - Byzantine nodes have executed any injected actions for this tick.
      - Network fault model has been applied to all messages in transit.
    This method is synchronous. It returns only when the full tick is complete.
```

### Observation Methods

```
read_observation(node_id: NodeID) → dict
    Return raw metrics for a single honest node as a plain Python dict.
    The dict must contain exactly these keys (both adapters must use identical keys):
      role, current_view, phase_latency_p50, phase_latency_p99,
      qc_miss_streak, view_changes_last_50, equivocation_counts,
      inter_message_variance, suspected_peers, committed_tps,
      pending_count, pipeline_utilisation
    AccordisTransform.transform() converts this dict to AccordisObservation.
    Byzantine node IDs must never be passed to this method.

get_committed_log(node_id: NodeID) → List[dict]
    Return the full committed block list for node_id as a list of plain dicts.
    Each dict must be deserializable to a Block model.
    Used exclusively by CorrectnessOracle. Never called by the agent.

get_honest_nodes() → List[NodeID]
    Return the current list of non-Byzantine node IDs.

get_byzantine_nodes() → List[NodeID]
    Return the current list of Byzantine node IDs.
```

### Byzantine Injection Method

```
inject_byzantine_action(
    byzantine_node_id: NodeID,
    strategy: BFAStrategy,
    target_nodes: List[NodeID],
    parameters: dict
) → None
    Instruct a Byzantine node to execute a disruption strategy on the immediately
    following advance_one_step() call only. The effect does not persist across steps.
    Called once per Byzantine node per step, before advance_one_step().
    The adapter translates BFAStrategy into engine-level behavior.

    Strategy-to-parameter mapping (both adapters must honour the same contract):
      NONE              → no-op; parameters: {}
      RANDOM_DELAY      → parameters: {delay_ms: int}
      SELECTIVE_DELAY   → parameters: {delay_ms: int, targets: List[NodeID]}
      EQUIVOCATION      → parameters: {targets_A: List[NodeID], targets_B: List[NodeID]}
      LEADER_SUPPRESS   → parameters: {targets: List[NodeID]}
      CASCADE_TIMING    → parameters: {delay_ms: int}  (all Byzantine nodes, same params)
      RECOVERY_DELAY    → parameters: {delay_ms: int}
      ADAPTIVE_MIRROR   → parameters: {view_timeout_ms: int, delta_ms: int}
```

### Adapter Invariants (Both Versions Must Enforce)

1. `advance_one_step()` always completes in full before any observation is read.
2. `inject_byzantine_action()` takes effect on the immediately following `advance_one_step()` only.
3. QC formation is accepted on `2f+1` vote count. No cryptographic verification in either version.
4. `quorum_size = 2f+1` is fixed inside the adapter. It is never configurable from outside.
5. `get_committed_log()` reflects only state after the most recent `advance_one_step()`.
6. `apply_config()` is only ever called with honest node IDs.
7. `read_observation()` dict must contain exactly the keys listed above — no more, no fewer.

---

## Section 2 — Shared Components (Identical in Both Versions)

These components are implemented once. They never change when the adapter is swapped.

### models.py — All Pydantic Models

> **All models live exclusively in `models.py`.** No model is defined in any file under
> `server/`. Server modules import from `models` — they never redefine models.
> Import OpenEnv parent classes from the SDK; subclass them. Do not duplicate parent fields.

#### Task 1.1 — Import OpenEnv Base Classes

Read `OPENENV_REFERENCE.md` and confirm the exact import paths. The minimum required:

```python
from openenv.core.env_server.interfaces import Transform
from openenv.core.rubrics import Rubric
from openenv.core.env_server.types import Action, Observation
```

#### Task 1.2 — Primitive Types and Enums

```python
NodeID = str   # e.g. "node_0", "node_1"

class Phase(str, Enum):
    PREPARE    = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT     = "commit"
    DECIDE     = "decide"

class NodeRole(str, Enum):
    LEADER    = "leader"
    REPLICA   = "replica"
    CANDIDATE = "candidate"

class LeaderRotation(str, Enum):
    ROUND_ROBIN         = "round_robin"
    VRF                 = "vrf"
    REPUTATION_WEIGHTED = "reputation_weighted"

class BFAStrategy(str, Enum):
    NONE            = "none"
    RANDOM_DELAY    = "random_delay"
    SELECTIVE_DELAY = "selective_delay"
    EQUIVOCATION    = "equivocation"
    LEADER_SUPPRESS = "leader_suppression"
    CASCADE_TIMING  = "cascade_timing"
    RECOVERY_DELAY  = "recovery_delay"
    ADAPTIVE_MIRROR = "adaptive_mirror"
```

#### Task 1.3 — BFTConfig (Tunable Parameters)

```python
class BFTConfig(BaseModel):
    """
    Tunable configuration for a single honest node.
    quorum_size is enforced by the adapter — NOT a tunable field.
    All bounds enforced by AccordisEnvironment._clamp_action() before reaching adapter.
    """
    view_timeout_ms:             int = 2000   # bounds: (200, 10000)
    pipeline_depth:              int = 2       # bounds: (1, 8)
    replication_batch_size:      int = 64      # bounds: (1, 512)
    equivocation_threshold:      int = 5       # bounds: (1, 15)
    vote_aggregation_timeout_ms: int = 500     # bounds: (50, 1000); must be < view_timeout/2

SAFE_BFT_TUNING_BOUNDS: Dict[str, Tuple[int, int]] = {
    "view_timeout_ms":             (200, 10000),
    "pipeline_depth":              (1, 8),
    "replication_batch_size":      (1, 512),
    "equivocation_threshold":      (1, 15),
    "vote_aggregation_timeout_ms": (50, 1000),
}

STATIC_BASELINE_CONFIG = BFTConfig(
    view_timeout_ms=2000,
    pipeline_depth=2,
    replication_batch_size=64,
    equivocation_threshold=5,
    vote_aggregation_timeout_ms=500,
)
```

#### Task 1.4 — AccordisObservation (extends OpenEnv Observation)

```python
class AccordisObservation(Observation):
    """
    Per-node partial observation. Each honest node sees only its own metrics.
    Agent cannot observe Byzantine node internals or other nodes' observations.
    """
    node_id:      NodeID
    current_role: NodeRole
    current_view: int

    # QC latency signals — primary tuning inputs
    per_phase_latency_p50:    Dict[Phase, float]    # ms, rolling 10-step median
    per_phase_latency_p99:    Dict[Phase, float]    # ms, rolling 99th percentile
    qc_formation_miss_streak: int                   # consecutive rounds without valid QC
    view_change_count_recent: int                   # view changes in last 50 steps

    # Byzantine suspicion signals (derived from observable patterns only)
    equivocation_miss_streak:  Dict[NodeID, int]    # inconsistent messages per peer
    message_arrival_variance:  Dict[NodeID, float]  # inter-message timing variance per peer
    suspected_byzantine:       Dict[NodeID, bool]   # agent's own suspicion flags

    # Throughput signals
    commit_throughput_tps:  float   # confirmed TPS
    pending_txn_count:      int     # uncommitted transactions in pool
    pipeline_utilisation:   float   # fraction of pipeline_depth in use [0.0, 1.0]

    # Self-knowledge
    current_config: BFTConfig
    step:           int
```

#### Task 1.5 — AccordisAction (extends OpenEnv Action)

```python
class AccordisAction(Action):
    """
    Configuration assignment for a single honest node per step.
    All values are clamped to SAFE_BFT_TUNING_BOUNDS by AccordisEnvironment before
    being passed to the adapter. The adapter always receives valid, in-bounds values.
    """
    node_id:                     NodeID
    view_timeout_ms:             int
    pipeline_depth:              int
    replication_batch_size:      int
    equivocation_threshold:      int
    vote_aggregation_timeout_ms: int
    suspect_node:                Optional[NodeID] = None
    clear_suspicion:             Optional[NodeID] = None

MultiNodeAction = Dict[NodeID, AccordisAction]
```

#### Task 1.6 — AccordisReward

```python
class AccordisReward(BaseModel):
    """
    Per-step reward breakdown. total is the sum used for policy gradient.
    Final episode reward normalised to [0.0, 1.0] by AccordisRubric.
    """
    liveness_cost:           float = 0.0   # -1/step while pending transactions exist
    unnecessary_view_change: float = 0.0   # -150 per false view change
    liveness_stall:          float = 0.0   # -300 if no commit >20 steps with honest quorum
    block_commit:            float = 0.0   # +60 per block committed by 2f+1 honest replicas
    fast_leader_recovery:    float = 0.0   # +120 per quick Byzantine leader replacement
    false_positive_avoided:  float = 0.0   # +40 per resolved suspicion without view change
    pipeline_efficiency:     float = 0.0   # +15 if utilisation > 0.8 × pipeline_depth at end
    throughput_improvement:  float = 0.0   # +20 if episode TPS > static baseline TPS
    latency_improvement:     float = 0.0   # +10 if p99 < static baseline p99
    total:                   float = 0.0   # sum of all components
```

#### Task 1.7 — AccordisTransform (extends OpenEnv Transform)

```python
class AccordisTransform(Transform):
    """
    Converts a raw metrics dict (from BaseConsensusAdapter.read_observation()) into
    a typed AccordisObservation. Version-agnostic: the dict schema is identical
    from both SimulatedConsensusAdapter and LibraBFTAdapter.
    """
    smoothing_window:     int   = 10
    normalise_latency_ms: float = 10000.0

    def transform(
        self, raw_metrics: dict, node_id: NodeID, step: int, current_config: BFTConfig
    ) -> AccordisObservation:
        # Compute rolling p50/p99 from raw_metrics phase_latency values
        # Compute message_arrival_variance from inter-message timing history
        # Return fully constructed AccordisObservation
        raise NotImplementedError
```

#### Task 1.8 — AccordisRubric (extends OpenEnv Rubric)

```python
class AccordisRubric(Rubric):
    """
    Episode-level grader. Normalises cumulative reward to [0.0, 1.0].
    Used by openenv validate and all task graders.
    """
    task_id:             str
    curriculum_level:    int
    max_possible_reward: float
    min_possible_reward: float

    def grade(
        self,
        episode_rewards:  List[AccordisReward],
        verifier_results: "VerifierResults",
    ) -> float:
        """
        score = (sum(r.total) - min_possible_reward) / (max_possible_reward - min_possible_reward)
        Clamp to [0.0, 1.0].
        Return 0.0 immediately if verifier_results.agreement_violated or validity_violated.
        """
        raise NotImplementedError
```

#### Task 1.9 — State and Supporting Models

```python
class NodeState(BaseModel):
    node_id:       NodeID
    is_byzantine:  bool
    committed_log: List["Block"]
    current_view:  int
    current_role:  NodeRole
    config:        BFTConfig

class Block(BaseModel):
    slot:         int
    hash:         str
    proposer_id:  NodeID
    transactions: List["Transaction"]

class Transaction(BaseModel):
    id:           str
    submitted_at: int

class ProposalRegistry(BaseModel):
    honest_proposals: Dict[str, Transaction]

class AccordisState(BaseModel):
    """Full true state — accessible to oracle and environment only. Never exposed to agent."""
    episode_id:        str
    step:              int
    curriculum_level:  int
    n_nodes:           int
    f_byzantine:       int
    leader_rotation:   LeaderRotation
    node_states:       Dict[NodeID, NodeState]
    view_change_count: int
    bfa_strategy:      BFAStrategy
    proposal_registry: ProposalRegistry
    episode_txn_pool:  List[Transaction]

class VerificationResult(BaseModel):
    passed:   bool
    property: Literal["Agreement", "Validity", "Liveness"]
    evidence: str = ""

class VerifierResults(BaseModel):
    agreement:          VerificationResult
    validity:           VerificationResult
    liveness:           Optional["LivenessResult"] = None
    agreement_violated: bool = False
    validity_violated:  bool = False

class LivenessResult(BaseModel):
    committed_count:      int
    pending_count:        int
    liveness_rate:        float
    view_change_overhead: int

class BaselineComparison(BaseModel):
    baseline_throughput_tps:      float
    baseline_view_change_count:   int
    baseline_commit_latency_p99:  float
    relative_tps_improvement:     float
    relative_latency_improvement: float

class EpisodeLog(BaseModel):
    episode_id:        str
    curriculum_level:  int
    bfa_strategy:      BFAStrategy
    bfa_strategy_seed: int
    steps:             List[AccordisState]
    rewards:           List[AccordisReward]
    final_score:       float
```

---

### AccordisEnvironment (server/accordis_environment.py)

This class is implemented once and never changes between versions.
It accepts a `BaseConsensusAdapter` parameter at construction.
It contains **no import** from `server/adapters/simulated/` or `server/adapters/librabft/`.

#### Constructor

```python
def __init__(self, adapter: BaseConsensusAdapter):
    self._adapter    = adapter            # injected — never constructed here
    self._bfa        = ByzantineFailureAgent()
    self._oracle     = CorrectnessOracle()
    self._calculator = RewardCalculator()
    self._curriculum = CurriculumManager()
    self._transform  = AccordisTransform()
    self._state: Optional[AccordisState] = None
    self._episode_rewards: List[AccordisReward] = []
```

#### Responsibility Boundary

AccordisEnvironment is responsible for:

- Orchestrating the episode loop: reset, step, done detection
- Clamping agent actions to SAFE_BFT_TUNING_BOUNDS before passing to adapter
- Delegating all consensus execution to `self._adapter`
- Invoking ByzantineFailureAgent to select disruption strategies per step
- Invoking CorrectnessOracle after every step
- Invoking RewardCalculator to produce per-step rewards
- Maintaining AccordisState by reading from the adapter after each tick
- Enforcing the episode horizon

AccordisEnvironment is **not** responsible for:

- How nodes are started, stopped, or configured internally
- How network latency is injected (real or simulated)
- How Byzantine behaviors are implemented at the protocol level
- Any gRPC, subprocess, socket, or I/O operations

#### reset() Internal Sequence

```
1. self._adapter.stop_cluster()
2. self._adapter.configure_network(curriculum_level, [])
3. node_ids = self._adapter.start_cluster(n_nodes, f_byzantine, leader_rotation)
4. for each honest node: self._adapter.apply_config(node_id, STATIC_BASELINE_CONFIG)
5. bfa_strategy = self._bfa.select_strategy(curriculum_level, seed=bfa_strategy_seed)
6. for each honest node:
     raw = self._adapter.read_observation(node_id)
     obs[node_id] = self._transform.transform(raw, node_id, step=0, STATIC_BASELINE_CONFIG)
7. Initialise AccordisState, episode_txn_pool, episode_rewards = []
8. Return obs dict
```

#### step() Internal Sequence

```
1. For each action in actions: clamped = self._clamp_action(action)
2. For each honest node: self._adapter.apply_config(node_id, clamped_config)
3. strategy = self._bfa.select_strategy(level, step, agent_configs)
   params   = self._bfa.get_disruption_parameters(strategy, agent_configs)
4. For each Byzantine node:
     self._adapter.inject_byzantine_action(node_id, strategy, targets, params)
5. self._adapter.advance_one_step()
6. For each honest node:
     raw = self._adapter.read_observation(node_id)
     obs[node_id] = self._transform.transform(raw, node_id, step, config)
7. Update AccordisState:
     for each honest node: pull committed_log from self._adapter.get_committed_log(node_id)
     update view_change_count, node roles, step counter
8. verifier_results = self._oracle.run_all(self._state)
9. liveness         = self._oracle.check_liveness(self._state)
10. reward          = self._calculator.compute(prev_state, self._state,
                          verifier_results, baseline, liveness)
11. done = (step >= max_steps) or (liveness.pending_count == 0)
12. If done: apply episode-end reward components; oracle.compute_baseline_comparison()
13. Return (obs, rewards, done, info)
```

#### _clamp_action() Contract

```
For every field in SAFE_BFT_TUNING_BOUNDS: clamp to [lo, hi]
Enforce: vote_aggregation_timeout_ms < view_timeout_ms // 2
The adapter always receives in-bounds values. No agent output can violate BFT safety.
```

---

### ByzantineFailureAgent (server/adversary/bfa.py)

Shared between both versions. Operates entirely on `BFAStrategy` enums and `BFTConfig` values.
Has **zero knowledge** of how strategies are implemented at the adapter level.

```
STRATEGY_POOL: Dict[int, List[BFAStrategy]] = {
    1: [NONE],
    2: [RANDOM_DELAY],
    3: [SELECTIVE_DELAY],
    4: [EQUIVOCATION],
    5: [ADAPTIVE_MIRROR],
    6: [LEADER_SUPPRESS, SELECTIVE_DELAY],
    7: [CASCADE_TIMING, EQUIVOCATION],
    8: all BFAStrategy values,
}

select_strategy(curriculum_level, step, agent_configs, seed) → BFAStrategy
  - Sample from STRATEGY_POOL[curriculum_level] using seeded PRNG
  - For ADAPTIVE_MIRROR: compute delta from min(agent_configs.view_timeout_ms)
  - Same seed → same strategy sequence across any adapter

get_disruption_parameters(strategy, agent_configs) → dict
  - Returns parameters dict matching inject_byzantine_action() contract exactly
  - Both adapters receive the same dict and are responsible for execution
```

---

### CorrectnessOracle (server/oracle/verifier.py)

Shared between both versions. Operates only on `AccordisState`. **Never calls the adapter.**
Reads from `AccordisState.node_states[nid].committed_log` which the environment populates
from adapter output after each step.

```
verify_agreement(state: AccordisState) → VerificationResult
  For every log slot, all honest nodes that have committed must share the same block hash.
  Failure = environment bug, not policy failure.

verify_validity(state: AccordisState) → VerificationResult
  Every committed transaction must appear in proposal_registry.honest_proposals.

check_liveness(state: AccordisState) → LivenessResult
  liveness_rate = committed_count / max(submitted_count, 1)

compute_baseline_comparison(episode_log, static_config, bfa_strategy_seed) → BaselineComparison
  Re-simulate episode with static_config and the same seed.
  Deterministic: same inputs → same result always.
  Policy-independent: does not access agent actions.

run_all(state: AccordisState) → VerifierResults
  Calls verify_agreement() and verify_validity(). Returns combined VerifierResults.
```

---

### RewardCalculator (server/rewards/reward_calculator.py)

Shared between both versions. Operates only on `AccordisState` transitions and oracle outputs.
**Never calls the adapter.**

```
Reward constants:
  LIVENESS_COST_PER_STEP  = -1.0
  UNNECESSARY_VIEW_CHANGE = -150.0
  LIVENESS_STALL          = -300.0
  BLOCK_COMMIT            = +60.0
  FAST_LEADER_RECOVERY    = +120.0
  FALSE_POSITIVE_AVOIDED  = +40.0
  PIPELINE_EFFICIENCY     = +15.0   (episode-end only)
  THROUGHPUT_IMPROVEMENT  = +20.0   (episode-end only)
  LATENCY_IMPROVEMENT     = +10.0   (episode-end only)

compute(prev_state, current_state, verifier_results, baseline, liveness) → AccordisReward
  Apply each component conditionally based on state transition signals.
  Episode-end components are set only when AccordisEnvironment.step() signals done=True.
  total = sum of all components.
```

---

### CurriculumManager (server/curriculum/manager.py)

Shared between both versions. Operates on liveness scores only.

```
record_episode(liveness_rate: float) → None
  Append to rolling 50-episode window.

should_advance() → bool
  True if window is full (50 episodes) AND rolling average > 0.85.

advance() → int
  Increment level to max 8. Return new level.

level: int  (read-only property)
```

---

### FaultProfiles (server/network/fault_profiles.py)

Shared data definitions used by both adapter versions. The `FaultProfile` is a plain
data object. Both adapters read it and are responsible for applying it in their own way.

```
FaultProfile fields:
  latency_p50_ms:  float
  latency_p99_ms:  float
  jitter_ms:       float
  packet_loss_pct: float
  distribution:    str = "pareto"

FAULT_PROFILES:
  Level 1: FaultProfile(2,    5,    1,   0.0)   Stable
  Level 2: FaultProfile(5,    50,   5,   0.0)   Variable
  Level 3: FaultProfile(10,   200,  20,  0.0)   Burst p99
  Level 4: FaultProfile(10,   100,  10,  10.0)  Packet loss
  Level 5: FaultProfile(5,    150,  15,  2.0)   Regime switches
  Level 6: FaultProfile(20,   300,  30,  5.0)   Crash + adversarial
  Level 7: FaultProfile(80,   500,  50,  0.0)   Geo-distributed
  Level 8: FaultProfile(30,   400,  40,  8.0)   Non-stationary

get_fault_profile(curriculum_level: int) → FaultProfile
```

---

### Task Definitions (server/tasks/)

All tasks are shared between both versions. Tasks define initial conditions and grading logic
only. They never reference an adapter.

#### BaseTask (server/tasks/base_task.py)

```
BaseTask (abstract):
  task_id:           str
  curriculum_levels: List[int]
  n_nodes:           int
  f_byzantine:       int
  leader_rotation:   LeaderRotation
  max_steps:         int

  get_initial_conditions() → dict
    Returns kwargs for AccordisEnvironment.reset()

  get_rubric() → AccordisRubric

  grade(episode_log: EpisodeLog) → float
    Deterministic score in [0.0, 1.0]. No adapter dependency.
```

#### EasyTask — Level 1–2 (server/tasks/task_easy.py)

```
Goal: Achieve and maintain consensus under stable conditions.

Initial conditions:
  curriculum_level: 1 or 2
  n_nodes:          4
  f_byzantine:      0 (level 1) / 1 (level 2, RANDOM_DELAY only)
  leader_rotation:  round_robin
  network_profile:  FaultProfile(p50=2ms, p99=5ms, jitter=1ms, loss=0%)
  bfa_strategy:     NONE / RANDOM_DELAY
  max_steps:        200

Success criteria:
  liveness_rate >= 0.95, view_change_count <= 3, no oracle violations

Grader:
  score = 0.5 × liveness_rate
        + 0.3 × max(0, 1 - view_change_count / 5)
        + 0.2 × (1.0 if agreement_ok and validity_ok else 0.0)
  Clamp to [0.0, 1.0]
```

#### MediumTask — Level 3–5 (server/tasks/task_medium.py)

```
Goal: Recover from crash failures and maintain consistency under adaptive delays.

Initial conditions:
  curriculum_level: 3–5
  n_nodes:          7
  f_byzantine:      2 (1 crash + 1 Byzantine)
  leader_rotation:  round_robin
  network_profile:  FaultProfile per level
  bfa_strategy:     SELECTIVE_DELAY / EQUIVOCATION / ADAPTIVE_MIRROR
  max_steps:        300

Success criteria:
  liveness_rate >= 0.80, fast_leader_recovery >= 1, no oracle violations

Grader:
  recovery_bonus = 0.2 if fast_leader_recovery_count >= 1 else 0.0
  score = 0.4 × liveness_rate
        + 0.2 × max(0, 1 - view_change_count / 10)
        + recovery_bonus
        + 0.2 × (1.0 if agreement_ok and validity_ok else 0.0)
  Clamp to [0.0, 1.0]
```

#### HardTask — Level 6–8 (server/tasks/task_hard.py)

```
Goal: Maintain correctness and liveness under coordinated full-coalition Byzantine attacks.

Initial conditions:
  curriculum_level: 6–8
  n_nodes:          10
  f_byzantine:      3 (maximum BFT tolerance at n=10)
  leader_rotation:  round_robin (level 6) → vrf / reputation_weighted (levels 7–8)
  network_profile:  FaultProfile per level (geo-distributed, non-stationary)
  bfa_strategy:     LEADER_SUPPRESS + CASCADE_TIMING + ADAPTIVE_MIRROR (coalition)
  max_steps:        500

Success criteria:
  liveness_rate >= 0.70, throughput_improvement > 0, no oracle violations

Grader:
  baseline_delta   = relative_tps_improvement from BaselineComparison
  throughput_score = min(1.0, max(0.0, baseline_delta + 0.5))
  score = 0.35 × liveness_rate
        + 0.25 × throughput_score
        + 0.25 × max(0, 1 - view_change_count / 20)
        + 0.15 × (1.0 if agreement_ok and validity_ok else 0.0)
  Clamp to [0.0, 1.0]
```

---

## Section 3 — Version 1: Simulated Environment

**Implementation target:** `server/adapters/simulated/`

Version 1 is a fully self-contained, single-process, synchronous implementation of
`BaseConsensusAdapter`. It has zero external dependencies. No build steps, no subprocesses,
no sockets. It starts instantly and is the default. All development, CI testing, and OpenEnv
validation run against Version 1.

### Design Principles

- Every node is an in-memory Python object, not a process.
- Every message is an entry in a Python dict/list, not a TCP packet.
- Every tick is a synchronous function call, not a real-time clock advance.
- Latency is a per-message integer representing the delivery tick, not a wall-clock measurement.
- Byzantine behavior is a per-node flag changing message-sending logic, not an RPC.
- The `BFAStrategy` enum values are identical to those used in Version 2.
- `advance_one_step()` processes the full HotStuff phase sequence for one round and returns.
- Determinism: all random sampling uses a seeded PRNG initialised from `bfa_strategy_seed`.

---

### hotstuff_sim.py — In-Memory HotStuff Phase Engine

#### Node State

Each simulated node holds the following:

```
node_id:                  NodeID
is_byzantine:             bool
current_view:             int
current_role:             NodeRole           (derived from view + rotation strategy)
committed_log:            List[Block]        (append-only)
config:                   BFTConfig
inbound_queue:            List[Message]      (messages delivered up to current_tick)
phase_latency_history:    Dict[Phase, deque] (rolling window; p50/p99 computed from this)
view_change_count:        int
no_commit_streak:         int                (steps since last block commit; for stall detection)
pending_byzantine_action: Optional[dict]     (set by inject_byzantine_action; cleared after tick)
```

#### Message Types

```
PROPOSAL  — (view: int, block: Block, proposer_id: NodeID)
VOTE      — (view: int, block_hash: str, voter_id: NodeID)
NEW_VIEW  — (from_view: int, new_view: int, sender_id: NodeID)
QC        — (view: int, block_hash: str)
```

#### HotStuff Phase Progression (one call to advance_one_step())

Each call to `advance_one_step()` processes exactly one logical round for all nodes:

```
Tick 1 — PREPARE:
  Determine leader for current_view via rotation strategy.
  Leader creates a new Block and sends PROPOSAL to all peers.
  PROPOSAL enters network_sim pending-delivery queue with computed delivery_tick.

Tick 2 — PRE-COMMIT (vote collection):
  NetworkSim flushes all messages with delivery_tick <= current_tick to inbound_queues.
  Each honest replica that received PROPOSAL sends a VOTE back to the leader.
  Byzantine replicas execute pending_byzantine_action instead of normal voting.
  Votes enter pending-delivery queue with computed delivery_tick.

Tick 3 — COMMIT (QC formation):
  Leader collects votes from inbound_queue.
  If 2f+1 VOTE messages for the same block_hash received: QC is formed.
    QC broadcast to all nodes (with delivery delay).
    All nodes that receive QC append the block to committed_log.
    Record QC formation latency (ticks elapsed since PROPOSAL sent).
  If QC not formed within view_timeout_ms ticks:
    All nodes send NEW_VIEW.
    Increment view_change_count.
    Elect new leader by rotation strategy.
    Reset pipeline (discard in-flight blocks for this view).
    Record view change.

Tick 4 — Metrics update:
  Update phase_latency_history for all four phases.
  Update pipeline_utilisation = (pipeline_depth - stalled_slots) / pipeline_depth.
  Increment no_commit_streak if no block committed this tick; else reset to 0.

Increment current_tick.
```

#### Leader Rotation

```
round_robin:         leader = node_ids[current_view % len(honest_nodes)]
vrf:                 leader = node_ids[seeded_hash(seed, current_view) % len(honest_nodes)]
reputation_weighted: leader = node_ids weighted by (1 / (1 + recent_view_change_count))
```

---

### network_sim.py — Synthetic Latency Model

The network simulator applies a `FaultProfile` to every message dispatched in the simulation.
It does not use real sockets or OS networking.

#### Delivery Model

When a node sends a message, the message is not placed in the recipient's `inbound_queue`
immediately. A delivery tick is computed and the message is placed in a global
pending-delivery queue sorted by delivery tick:

```
delivery_tick = current_tick + base_delay_ticks + jitter_ticks

base_delay_ticks: Pareto-sampled from (profile.latency_p50_ms, profile.latency_p99_ms)
                  Pareto shape parameter α computed from p50/p99 ratio.
jitter_ticks:     uniform random in [-profile.jitter_ms/2, +profile.jitter_ms/2]
loss:             if random() < profile.packet_loss_pct / 100: message is discarded.
                  Discarded messages never reach the delivery queue.
```

On each call to `advance_one_step()`, `NetworkSim.flush(current_tick)` is called first.
It moves all messages with `delivery_tick <= current_tick` into their target node's
`inbound_queue`.

#### Determinism Guarantee

All random sampling uses `random.Random(bfa_strategy_seed)` initialised at `reset()`.
The same seed always produces the same delivery sequence for the same fault profile.

#### Regime Switching (Levels 5 and 8)

`configure_network()` may be called mid-episode to replace the active `FaultProfile`.
The new profile takes effect on the next message dispatched. Already-queued messages
retain their original delivery_tick.

---

### bfa_sim.py — In-Process Byzantine Behavior Injector

The injector translates `BFAStrategy` values into node-level message-sending behavior.
It is called by `SimulatedConsensusAdapter.inject_byzantine_action()` and stores the
instruction on the target node's `pending_byzantine_action` field. The instruction is
consumed and cleared during the next `advance_one_step()` tick.

#### Strategy-to-Behavior Mapping

```
NONE:
  pending_byzantine_action = None
  Node behaves honestly on this tick.

RANDOM_DELAY:
  Mark all outbound messages from this node with: extra_delay_ticks = parameters["delay_ms"]
  NetworkSim adds this to the computed delivery_tick for all messages from this node.

SELECTIVE_DELAY:
  Same as RANDOM_DELAY but only for messages addressed to nodes in parameters["targets"].

EQUIVOCATION:
  Node sends two different VOTE messages this tick:
    vote_A → sent to parameters["targets_A"] with block_hash = hash_A
    vote_B → sent to parameters["targets_B"] with block_hash = hash_B (hash_B != hash_A)
  Both votes enter the delivery queue. This prevents QC formation if A ∩ B is sufficient.

LEADER_SUPPRESS:
  When this node is the leader: PROPOSAL messages to parameters["targets"] are discarded.
  They are never placed in the delivery queue.

CASCADE_TIMING:
  Functionally identical to SELECTIVE_DELAY applied to all Byzantine nodes simultaneously.
  AccordisEnvironment calls inject_byzantine_action() for each Byzantine node with
  the same parameters["delay_ms"]. The injector treats each node independently.

RECOVERY_DELAY:
  When this node sends NEW_VIEW: apply extra_delay_ticks = parameters["delay_ms"].

ADAPTIVE_MIRROR:
  Compute actual_delay = parameters["view_timeout_ms"] - parameters["delta_ms"]
  Apply as SELECTIVE_DELAY to all non-Byzantine nodes.
```

---

### SimulatedConsensusAdapter (server/adapters/simulated/adapter.py)

This class implements all methods of `BaseConsensusAdapter` using the three simulation
modules above. It is the single coordinator of the simulation tick.

#### Responsibilities

- Instantiate and hold `SimulatedNode` objects for each node
- Delegate message scheduling to `NetworkSimulator`
- Translate `inject_byzantine_action()` calls to `ByzantineInjector`
- Drive `HotStuffSimulator.tick()` on `advance_one_step()`
- Produce raw metrics dicts from node state on `read_observation()`
- Produce committed log dicts from node state on `get_committed_log()`

#### Raw Metrics Dict from `read_observation(node_id)`

Must contain exactly these keys (same contract as LibraBFTAdapter — enforced by parity test):

```
role:                    str (NodeRole value)
current_view:            int
phase_latency_p50:       Dict[str, float]  (Phase name → p50 ms from phase_latency_history)
phase_latency_p99:       Dict[str, float]  (Phase name → p99 ms from phase_latency_history)
qc_miss_streak:          int
view_changes_last_50:    int               (from rolling view_change_count window)
equivocation_counts:     Dict[NodeID, int] (inconsistent vote observations per peer)
inter_message_variance:  Dict[NodeID, float]
suspected_peers:         Dict[NodeID, bool]
committed_tps:           float             (committed blocks / elapsed time in ticks)
pending_count:           int               (len(episode_txn_pool) - len(committed transactions))
pipeline_utilisation:    float
```

---

## Section 4 — Version 2: LibraBFT Real System Integration

**Implementation target:** `server/adapters/librabft/`

Version 2 wraps a real running LibraBFT cluster. It implements the same `BaseConsensusAdapter`
interface as Version 1. The environment layer is identical. Only the adapter subtree differs.

### Prerequisites

```bash
# Build LibraBFT with crypto stub, Twins API, and simulation clock
cargo build --release --features twins,no-bls-verify,simulation-clock -p libra-bft

# Start Toxiproxy server
toxiproxy-server --port 8474 &
```

### Process and Port Layout (example: 4 nodes)

```
node_0: consensus_port=50050, mgmt_grpc_port=60050, metrics_http_port=9090
node_1: consensus_port=50051, mgmt_grpc_port=60051, metrics_http_port=9091
node_2: consensus_port=50052, mgmt_grpc_port=60052, metrics_http_port=9092
node_3: consensus_port=50053, mgmt_grpc_port=60053, metrics_http_port=9093 (Twins/Byzantine)

Toxiproxy directed link proxy example (node_1 → node_0):
  proxy:   listen=51010, upstream=localhost:50050
  toxics:  latency(latency=p50ms, jitter=jitter_ms), bandwidth(rate=...)
  result:  node_1's config says peer node_0 is at localhost:51010 (not 50050)
           node_0 never knows traffic passes through a proxy
```

For N nodes: N×(N-1) directed proxies are created. Port allocation is deterministic:
`proxy_port(src, dst) = 51000 + src_index * N + dst_index`

### start_cluster() Data Flow

```
1. Generate per-node TOML config files at /tmp/accordis/node_{i}.toml
   Each config includes: own consensus port, all peer addresses (Toxiproxy ports),
   leader rotation strategy, no-bls-verify flag, Twins flag for Byzantine nodes,
   gRPC management port.

2. Call toxiproxy_client.create_all_proxies(node_ids, consensus_ports)
   Toxiproxy proxies are created BEFORE nodes start — nodes must find them on startup.

3. For each node i: subprocess.Popen(["libra-bft", "--config", "/tmp/accordis/node_i.toml"])
   Track subprocess handles in self._processes.

4. Poll grpc_client.health_check(node_id) for all nodes in retry loop (max 30s).
   Raise RuntimeError if any node fails to respond.

5. Return list of all node_ids (honest first, Byzantine last).
```

### apply_config() Data Flow

```
grpc_client.set_round_timeout(node_id,    config.view_timeout_ms)
grpc_client.set_pipeline_depth(node_id,   config.pipeline_depth)
grpc_client.set_batch_size(node_id,       config.replication_batch_size)
grpc_client.set_equiv_threshold(node_id,  config.equivocation_threshold)
grpc_client.set_vote_agg_timeout(node_id, config.vote_aggregation_timeout_ms)

All calls synchronous. Config changes take effect on the immediately following engine tick.
The gRPC management server listens on mgmt_grpc_port, separate from the consensus port.
```

### read_observation() Data Flow

```
LibraBFT exports Prometheus-format metrics at http://localhost:{metrics_http_port}/metrics

grpc_client.scrape(node_id) → HTTP GET /metrics → parse Prometheus text format → raw dict

Prometheus metric → raw dict key mapping:
  libra_bft_role                         → role
  libra_bft_current_view                 → current_view
  libra_bft_phase_latency_p50{phase=X}   → phase_latency_p50[X]
  libra_bft_phase_latency_p99{phase=X}   → phase_latency_p99[X]
  libra_bft_qc_miss_streak               → qc_miss_streak
  libra_bft_view_changes_total (delta)   → view_changes_last_50
  libra_bft_equivocation_count{peer=X}   → equivocation_counts[X]
  libra_bft_msg_arrival_variance{peer=X} → inter_message_variance[X]
  libra_bft_suspected_peers              → suspected_peers
  libra_bft_committed_tps                → committed_tps
  libra_bft_pending_txn_count            → pending_count
  libra_bft_pipeline_utilisation         → pipeline_utilisation

The resulting dict schema is IDENTICAL to SimulatedConsensusAdapter.read_observation().
AccordisTransform receives the same structure regardless of version. (Parity test enforces this.)
```

### advance_one_step() Data Flow

```
grpc_client.advance_clock(ticks=1)

The LibraBFT engine (built with --features simulation-clock):
  - Processes its current message queue
  - Fires any pending view timeouts
  - Attempts QC formation (accepted on 2f+1 votes, no BLS check)
  - Returns when the tick is fully complete

The gRPC call is synchronous. Python does not proceed until the engine tick returns.
This guarantees that read_observation() always reflects post-tick state.
```

### inject_byzantine_action() Data Flow

```
twins_client.inject(
    node_id=byzantine_node_id,
    action_type=strategy (mapped to Twins protobuf enum),
    targets=parameters["targets"],
    delay_ms=parameters.get("delay_ms", 0),
    equivocation_targets_A=parameters.get("targets_A", []),
    equivocation_targets_B=parameters.get("targets_B", []),
)

The Twins node receives this instruction on its Twins gRPC endpoint (on mgmt_grpc_port).
On the next advance_clock(1) call, the Twins node executes genuine LibraBFT protocol paths
with the strategic deviation applied — e.g., sending real equivocating PREPARE messages
using the actual HotStuff message format, to different replica subsets.

Effect is consumed after one tick. Does not persist to the following step.
```

### configure_network() Data Flow

```
toxiproxy_client.delete_all()

For each directed link (src → dst):
  toxiproxy_client.create_proxy(
      name    = f"link_{src}_{dst}",
      listen  = f"localhost:{proxy_port(src, dst)}",
      upstream= f"localhost:{consensus_port[dst]}"
  )
  toxiproxy_client.add_toxic(
      proxy_name = f"link_{src}_{dst}",
      type       = "latency",
      attributes = {latency: profile.latency_p50_ms, jitter: profile.jitter_ms}
  )
  toxiproxy_client.add_toxic(
      proxy_name = f"link_{src}_{dst}",
      type       = "bandwidth",
      attributes = {rate: computed_from(profile.packet_loss_pct)}
  )

Uses Toxiproxy REST API at http://localhost:8474. No Toxiproxy Python SDK required —
HTTP calls via requests library.
```

---

## Section 5 — Swappability Guarantees

### The Guarantee

Replacing `SimulatedConsensusAdapter` with `LibraBFTAdapter` requires **zero changes**
to any of the following files:

```
accordis_environment.py   models.py           reward_calculator.py
verifier.py               bfa.py              manager.py
fault_profiles.py         base_task.py        task_easy.py
task_medium.py            task_hard.py        app.py (except create_adapter call)
inference.py              openenv.yaml
```

The only change required is at the **construction site** — the single `create_adapter()` call.

### Three Enforcement Mechanisms

**Mechanism 1 — Import firewall on `accordis_environment.py`.**

Permitted imports in `accordis_environment.py`:

```
models
server/adapters/base.py   (interface only, never implementations)
server/adversary/bfa.py
server/oracle/verifier.py
server/rewards/reward_calculator.py
server/curriculum/manager.py
standard library
```

Any import from `server/adapters/simulated/` or `server/adapters/librabft/` is a
**violation** that breaks the swappability guarantee. CI must check this.

**Mechanism 2 — Identical raw metrics dict schema.**

Both adapters must produce exactly the same set of dict keys from `read_observation()`.
`AccordisTransform` is tested against both adapters independently. If either adapter's
dict is missing a key or has a different value type, the parity test fails (Phase 15).

**Mechanism 3 — Constructor injection.**

`AccordisEnvironment.__init__` accepts `adapter: BaseConsensusAdapter` as a parameter.
It never calls `SimulatedConsensusAdapter()` or `LibraBFTAdapter()` internally.
A type annotation check confirms the parameter type is `BaseConsensusAdapter`.

---

## Section 6 — Dependency Injection Strategy

### Construction Pattern

```python
# Version 1 startup (default)
from server.adapters.simulated.adapter import SimulatedConsensusAdapter
from server.accordis_environment import AccordisEnvironment

adapter = SimulatedConsensusAdapter()
env     = AccordisEnvironment(adapter=adapter)

# Version 2 startup (identical environment, different adapter)
from server.adapters.librabft.adapter import LibraBFTAdapter
from server.accordis_environment import AccordisEnvironment

adapter = LibraBFTAdapter(
    librabft_binary="/usr/local/bin/libra-bft",
    toxiproxy_url="http://localhost:8474",
)
env = AccordisEnvironment(adapter=adapter)
```

### create_adapter() Factory (server/adapters/__init__.py)

`app.py` and `inference.py` call `create_adapter()`. The environment class itself never
reads the `ACCORDIS_ADAPTER` environment variable.

```
create_adapter(version: Optional[str] = None) → BaseConsensusAdapter

  version = version or os.environ.get("ACCORDIS_ADAPTER", "simulated")

  if version == "librabft":
      from server.adapters.librabft.adapter import LibraBFTAdapter
      return LibraBFTAdapter(
          librabft_binary=os.environ.get("LIBRABFT_BINARY", "/usr/local/bin/libra-bft"),
          toxiproxy_url=os.environ.get("TOXIPROXY_URL", "http://localhost:8474"),
      )

  from server.adapters.simulated.adapter import SimulatedConsensusAdapter
  return SimulatedConsensusAdapter()
```

`app.py` usage:

```python
from server.adapters import create_adapter
from server.accordis_environment import AccordisEnvironment

adapter = create_adapter()
env     = AccordisEnvironment(adapter=adapter)
app     = create_app(AccordisEnvironment, AccordisObservation, AccordisAction, ...)
```

---

## Section 7 — OpenEnv Compliance Layer

### openenv.yaml

```yaml
name: AccordisEnvironment
description: >
  Self-Adaptive BFT Consensus Tuning Environment. An RL meta-controller
  adaptively tunes HotStuff BFT configuration parameters under dynamic
  network conditions and active Byzantine disruption. Pluggable consensus
  engine: Version 1 uses an in-memory HotStuff simulator (zero deps);
  Version 2 wraps LibraBFT (production Rust engine). RLVE-compliant.

environment_id: AccordisEnvironment-v1
track: Real-World Sequential Decision-Making
formulation: POMDP
version: 1.0.0

tasks:
  - id: easy
    name: Stable Consensus
    description: No Byzantine nodes, stable conditions. Learn safe timeout range.
    curriculum_levels: [1, 2]
    max_steps: 200

  - id: medium
    name: Fault-Tolerant Recovery
    description: Crash failures plus adaptive Byzantine delays. Maintain consistency.
    curriculum_levels: [3, 4, 5]
    max_steps: 300

  - id: hard
    name: Adversarial Resilience
    description: Coordinated Byzantine coalition attacks. Resist liveness attacks.
    curriculum_levels: [6, 7, 8]
    max_steps: 500

action_space:
  type: dict
  keys: honest_node_ids
  per_node:
    view_timeout_ms:             {type: int, min: 200,  max: 10000}
    pipeline_depth:              {type: int, min: 1,    max: 8}
    replication_batch_size:      {type: int, min: 1,    max: 512}
    equivocation_threshold:      {type: int, min: 1,    max: 15}
    vote_aggregation_timeout_ms: {type: int, min: 50,   max: 1000}
    suspect_node:    {type: str, nullable: true}
    clear_suspicion: {type: str, nullable: true}

observation_space:
  type: dict
  keys: honest_node_ids
  per_node:
    node_id, current_role, current_view,
    per_phase_latency_p50, per_phase_latency_p99,
    qc_formation_miss_streak, view_change_count_recent,
    equivocation_miss_streak, message_arrival_variance, suspected_byzantine,
    commit_throughput_tps, pending_txn_count, pipeline_utilisation,
    current_config, step

reward_range: [0.0, 1.0]
verifiable: true
```

### Validation

```bash
# Version 1 (default, zero deps)
ACCORDIS_ADAPTER=simulated openenv validate accordis/

# Version 2 (requires LibraBFT + Toxiproxy running)
ACCORDIS_ADAPTER=librabft openenv validate accordis/
```

Both must pass all checks. The interface is identical — no changes to `openenv.yaml` between versions.

---

## Section 8 — Inference Compatibility

### llm_factory.py

```
BaseLLMClient (abstract):
  complete(system: str, user: str) → str

OpenAIClient  → OPENAI_API_KEY, model gpt-4o
GeminiClient  → GEMINI_API_KEY, model gemini-1.5-pro

LLMClientFactory.create() → BaseLLMClient
  Priority: OPENAI_API_KEY > GEMINI_API_KEY
  Raise EnvironmentError if neither key is set
```

### inference.py

Runs one episode using an LLM agent. The adapter version is selected via
`ACCORDIS_ADAPTER` — `inference.py` does not branch on this. The episode loop is
identical regardless of version.

Required log format:

```
[START] task=easy adapter=simulated
[STEP]  step=0 reward=-1.0 total=-1.0 done=False
[STEP]  step=1 reward=59.0 total=58.0 done=False
...
[END]   steps=87 total_reward=340.0 score=0.73
```

---

## Section 9 — Testing Strategy

### Test Organisation

```
tests/
  test_models.py                ← No adapter. Validates model schemas only.
  test_environment.py           ← Parameterised. Runs against BOTH adapters.
  test_verifier.py              ← No adapter. Constructs AccordisState directly.
  test_tasks.py                 ← No adapter. Tests grader functions directly.
  test_reward.py                ← No adapter. Tests RewardCalculator directly.
  test_simulated_adapter.py     ← Version 1 internals only.
  test_librabft_adapter.py      ← Version 2 only. Auto-skipped if binary absent.
```

### Parameterised Environment Tests (test_environment.py)

```python
@pytest.fixture(params=["simulated", "librabft"])
def env(request):
    if request.param == "librabft" and not librabft_available():
        pytest.skip("LibraBFT binary not found")
    return AccordisEnvironment(adapter=create_adapter(request.param))

Tests (run against both fixtures):
  reset() returns Dict[NodeID, AccordisObservation]
  step(valid_actions) returns (obs, rewards, bool, dict)
  state() raises RuntimeError before reset()
  _clamp_action() enforces all SAFE_BFT_TUNING_BOUNDS
  vote_aggregation_timeout_ms < view_timeout_ms // 2 after clamp
  Agreement oracle never fires during honest-only easy-task episodes
  openenv validate passes (subprocess call; simulated fixture only)
```

### Version 1 Adapter Tests (test_simulated_adapter.py)

```
Tests the simulation internals independently of the environment:
  HotStuffSimulator forms valid QC on 2f+1 votes
  HotStuffSimulator triggers view change when view_timeout_ms ticks elapsed
  NetworkSimulator drops messages at configured packet_loss_pct (probabilistic)
  NetworkSimulator delivers messages with correct Pareto-distributed ticks
  ByzantineInjector EQUIVOCATION causes two distinct vote hashes to be sent
  ByzantineInjector LEADER_SUPPRESS drops proposals to specified targets only
  ByzantineInjector ADAPTIVE_MIRROR delays messages to view_timeout_ms - delta
  Determinism: same bfa_strategy_seed produces identical episode tick sequence
  Regime switch: configure_network() mid-episode changes delivery distribution
  read_observation() dict contains exactly the required 12 keys
```

### Version 2 Adapter Tests (test_librabft_adapter.py)

```
Auto-skipped if LIBRABFT_BINARY not set. Requires Toxiproxy running on port 8474.
  start_cluster() spawns n processes; all gRPC health checks return OK
  apply_config() change is reflected in read_observation() on next tick
  Toxiproxy proxy count equals n×(n-1) after configure_network()
  inject_byzantine_action(EQUIVOCATION) causes observable vote divergence
  advance_one_step() is synchronous; observation reflects post-tick state
  stop_cluster() terminates all subprocess handles; no zombies remain
  read_observation() dict contains exactly the same 12 keys as simulated adapter
```

### Adapter-Agnostic Tests

```
test_verifier.py:
  verify_agreement() passes on identical committed logs across honest nodes
  verify_agreement() fails on divergent block hashes at same slot
  verify_validity() passes for honest proposals only
  verify_validity() fails when committed txn not in proposal_registry
  check_liveness() liveness_rate = 0.0 when nothing committed
  check_liveness() liveness_rate = 1.0 when all txns committed

test_reward.py:
  liveness_cost = -1.0 per step when pending_count > 0
  liveness_stall = -300.0 when stall condition met (>20 steps no commit, quorum present)
  block_commit = +60.0 × new_commits
  unnecessary_view_change = -150.0 when triggered
  total = sum of all non-zero components

test_tasks.py:
  EasyTask.grade() = 1.0 on perfect liveness, 0 view changes
  MediumTask.grade() = 0.0 on no commits
  HardTask.grade() uses baseline_delta correctly
  All three graders clamp output to [0.0, 1.0]

test_models.py:
  Every model instantiates with valid data without validation error
  BFTConfig defaults match STATIC_BASELINE_CONFIG
  AccordisRubric.grade() returns 0.0 on agreement_violated=True
  AccordisRubric.grade() clamps to [0.0, 1.0]
```

---

## Section 10 — Migration Path: Version 1 → Version 2

This defines the exact steps to migrate. No shared component is modified during migration.

### Pre-Migration Gate

Before beginning Version 2 implementation, all of the following must be true:

- All `test_environment.py` tests pass with `ACCORDIS_ADAPTER=simulated`
- All adapter-agnostic tests pass (`test_verifier.py`, `test_reward.py`, `test_tasks.py`)
- `openenv validate accordis/` passes with `ACCORDIS_ADAPTER=simulated`
- `inference.py` completes a full episode with `ACCORDIS_ADAPTER=simulated`

### Step 1 — Build LibraBFT

```bash
git clone --depth 1 https://github.com/diem/diem /tmp/diem
cd /tmp/diem
cargo build --release \
    --features twins,no-bls-verify,simulation-clock \
    -p libra-bft
# Binary at: /tmp/diem/target/release/libra-bft
```

Features used:

- `no-bls-verify`: stubs BLS signature checks → vote-count quorum only
- `twins`: enables Twins injection gRPC endpoint on management port
- `simulation-clock`: enables `AdvanceClock` gRPC method for deterministic ticking

### Step 2 — Start Toxiproxy

```bash
toxiproxy-server --port 8474 &
```

### Step 3 — Implement server/adapters/librabft/

Implement `LibraBFTAdapter` satisfying `BaseConsensusAdapter` exactly as specified in
Section 4. The raw metrics dict from `read_observation()` must be key-for-key identical
to `SimulatedConsensusAdapter.read_observation()`.

### Step 4 — Metrics Dict Parity Test

Write (or run existing) parity test that confirms:

- Both adapters return dicts with exactly the same 12 keys
- Value types match for each key
- `AccordisTransform.transform()` produces a valid `AccordisObservation` from both

This test is the gating criterion for Step 5.

### Step 5 — Run Parameterised Environment Tests

```bash
LIBRABFT_BINARY=/tmp/diem/target/release/libra-bft \
pytest tests/test_environment.py
```

Both `simulated` and `librabft` fixture variants must pass all tests.
If any test passes for `simulated` but fails for `librabft`, the adapter implementation
has a defect — the environment and oracle are not changed.

### Step 6 — Run Version 2 Inference

```bash
ACCORDIS_ADAPTER=librabft OPENAI_API_KEY=... python inference.py
```

Confirm `[START]` / `[STEP]` / `[END]` output is structurally identical to Version 1.

### Step 7 — Update Dockerfile for Version 2

```dockerfile
FROM rust:1.76 AS libra-builder
WORKDIR /diem
RUN git clone --depth 1 https://github.com/diem/diem .
RUN cargo build --release \
    --features twins,no-bls-verify,simulation-clock \
    -p libra-bft

FROM python:3.11-slim
WORKDIR /app
COPY --from=libra-builder /diem/target/release/libra-bft /usr/local/bin/libra-bft
RUN apt-get update && apt-get install -y wget && \
    wget -O /usr/local/bin/toxiproxy-server \
      https://github.com/Shopify/toxiproxy/releases/latest/download/toxiproxy-server-linux-amd64 && \
    chmod +x /usr/local/bin/toxiproxy-server
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY . .
EXPOSE 8000 8474
CMD ["bash", "-c", \
     "toxiproxy-server --port 8474 & \
      ACCORDIS_ADAPTER=librabft uvicorn server.app:app --host 0.0.0.0 --port 8000"]
```

Version 1 Dockerfile does not include the LibraBFT build stage and does not set
`ACCORDIS_ADAPTER` (defaults to `simulated`).

### Step 8 — Final Validation

```bash
docker build -t accordis-env:v2 .
docker run -p 8000:8000 -p 8474:8474 accordis-env:v2
curl http://localhost:8000/health
ACCORDIS_ADAPTER=librabft openenv validate accordis/
```

---

## Implementation Sequence (Strict Order)

```
Phase 0    Read OPENENV_REFERENCE.md. Confirm toolchain. Analyse inference.py.
           ─────────────────────────────────────────────────────────────────────
Phase 1    Write models.py completely. All models. No stubs. Zero adapter imports.
           ─────────────────────────────────────────────────────────────────────
Phase 2    Write server/adapters/base.py — BaseConsensusAdapter (interface only).
           Write AccordisEnvironment skeleton — constructor accepts adapter parameter.
           Write openenv.yaml.
           Run: ACCORDIS_ADAPTER=simulated openenv validate (fix schema errors only).
           ─────────────────────────────────────────────────────────────────────
Phase 3    Implement SimulatedConsensusAdapter — Version 1.
             3a. network_sim.py  — Pareto delivery queue, loss, regime switch
             3b. bfa_sim.py      — all 8 BFAStrategy mappings to node behavior
             3c. hotstuff_sim.py — phase engine, QC logic, view-change, rotation
             3d. adapter.py      — wire together; implement all interface methods
           ─────────────────────────────────────────────────────────────────────
Phase 4    Implement shared adversary and curriculum.
             4a. bfa.py     — ByzantineFailureAgent strategy selection only
             4b. manager.py — CurriculumManager
           ─────────────────────────────────────────────────────────────────────
Phase 5    Implement CorrectnessOracle (verifier.py). All methods complete.
           ─────────────────────────────────────────────────────────────────────
Phase 6    Implement RewardCalculator (reward_calculator.py). All 9 components.
           ─────────────────────────────────────────────────────────────────────
Phase 7    Complete AccordisEnvironment.reset() and step().
           Wire all shared components. Pass openenv validate.
           ─────────────────────────────────────────────────────────────────────
Phase 8    Implement AccordisTransform. All rolling computations complete.
           ─────────────────────────────────────────────────────────────────────
Phase 9    Implement task classes (base_task, task_easy, task_medium, task_hard).
           ─────────────────────────────────────────────────────────────────────
Phase 10   Implement llm_factory.py and inference.py.
           Run full episode: ACCORDIS_ADAPTER=simulated python inference.py
           ─────────────────────────────────────────────────────────────────────
Phase 11   Implement server/app.py with create_adapter() factory.
           ─────────────────────────────────────────────────────────────────────
Phase 12   Write all tests. All adapter-agnostic and simulated-adapter tests must pass.
           ─────────────────────────────────────────────────────────────────────
Phase 13   Write README.md. Write Version 1 Dockerfile (no LibraBFT build stage).
           ─────────────────────────────────────────────────────────────────────
──────── VERSION 1 COMPLETE AND FULLY FUNCTIONAL. STOP HERE FOR SUBMISSION. ────────
           ─────────────────────────────────────────────────────────────────────
Phase 14   Implement LibraBFTAdapter — Version 2.
             14a. grpc_client.py      — gRPC management stubs + Prometheus scraper
             14b. toxiproxy_client.py — Toxiproxy REST API wrapper
             14c. twins_client.py     — Twins injection gRPC wrapper
             14d. adapter.py          — wire together; implement all interface methods
           ─────────────────────────────────────────────────────────────────────
Phase 15   Verify metrics dict parity between adapters.
           Run parameterised environment tests — both adapters must pass all tests.
           ─────────────────────────────────────────────────────────────────────
Phase 16   Write Version 2 Dockerfile. Build. Validate. Deploy to Hugging Face Spaces.
```

---

## Invariants the Agent Must Never Violate

| Invariant                                                                                | Enforcement Point                       |
| ---------------------------------------------------------------------------------------- | --------------------------------------- |
| All models defined only in `models.py`                                                 | No model defined under `server/`      |
| All tests in `accordis/tests/`                                                         | Directory structure                     |
| `accordis_environment.py` imports only `BaseConsensusAdapter`, never implementations | Import firewall                         |
| `AccordisEnvironment.__init__` accepts adapter as parameter, never constructs one      | Constructor signature                   |
| Both adapters produce identical raw metrics dict key set                                 | Parity test, Phase 15                   |
| `quorum_size = 2f+1` enforced by adapter, never configurable from environment          | Adapter invariant                       |
| BFAStrategy-to-behavior translation happens inside the adapter, not in `bfa.py`        | Responsibility boundary                 |
| `CorrectnessOracle` never calls the adapter                                            | Oracle reads `AccordisState` only     |
| `RewardCalculator` never calls the adapter                                             | Calculator reads `AccordisState` only |
| BFA seed guarantees reproducible trajectories in both versions                           | Seeded PRNG in adapter and bfa.py       |
| `inject_byzantine_action()` effect is consumed after exactly one tick                  | Adapter invariant                       |
| `ACCORDIS_ADAPTER=simulated` requires zero external dependencies                       | Simulated adapter constraint            |
| Swapping adapters requires zero changes outside `create_adapter()`                     | Design constraint                       |
| `[START]` / `[STEP]` / `[END]` log format in `inference.py`                      | Output format                           |

---
