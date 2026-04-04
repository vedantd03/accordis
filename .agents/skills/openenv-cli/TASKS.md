# AccordisEnvironment V1 — Task Tracking

## Status Legend
- `TODO` — Not started
- `IN_PROGRESS` — Being implemented
- `DONE` — Complete and verified

---

## Core Models

| # | Task | File | Status |
|---|------|------|--------|
| 1.1 | Import OpenEnv base classes | models.py | DONE |
| 1.2 | Primitive types and enums (Phase, NodeRole, etc.) | models.py | DONE |
| 1.3 | BFTConfig + SAFE_BFT_TUNING_BOUNDS + STATIC_BASELINE_CONFIG | models.py | DONE |
| 1.4 | AccordisObservation | models.py | DONE |
| 1.5 | AccordisAction | models.py | DONE |
| 1.6 | AccordisReward | models.py | DONE |
| 1.7 | AccordisTransform | models.py | DONE |
| 1.8 | AccordisRubric | models.py | DONE |
| 1.9 | State and supporting models (NodeState, Block, Transaction, etc.) | models.py | DONE |

---

## Adapter Layer

| # | Task | File | Status |
|---|------|------|--------|
| 2.1 | BaseConsensusAdapter ABC | server/adapters/base.py | DONE |
| 2.2 | create_adapter() factory | server/adapters/__init__.py | DONE |
| 2.3 | NetworkSimulator (Pareto latency model) | server/adapters/simulated/network_sim.py | DONE |
| 2.4 | HotStuffSimulator (4-phase engine) | server/adapters/simulated/hotstuff_sim.py | DONE |
| 2.5 | ByzantineInjector (BFA strategy translator) | server/adapters/simulated/bfa_sim.py | DONE |
| 2.6 | SimulatedConsensusAdapter | server/adapters/simulated/adapter.py | DONE |

---

## Shared Components

| # | Task | File | Status |
|---|------|------|--------|
| 3.1 | FaultProfile definitions (8 levels) | server/network/fault_profiles.py | DONE |
| 3.2 | ByzantineFailureAgent + STRATEGY_POOL | server/adversary/bfa.py | DONE |
| 3.3 | CorrectnessOracle (verify_agreement, validity, liveness) | server/oracle/verifier.py | DONE |
| 3.4 | RewardCalculator (9 reward components) | server/rewards/reward_calculator.py | DONE |
| 3.5 | CurriculumManager (rolling 50-episode window) | server/curriculum/manager.py | DONE |

---

## Task Definitions

| # | Task | File | Status |
|---|------|------|--------|
| 4.1 | BaseTask ABC | server/tasks/base_task.py | DONE |
| 4.2 | EasyTask (levels 1–2) | server/tasks/task_easy.py | DONE |
| 4.3 | MediumTask (levels 3–5) | server/tasks/task_medium.py | DONE |
| 4.4 | HardTask (levels 6–8) | server/tasks/task_hard.py | DONE |

---

## Environment + App

| # | Task | File | Status |
|---|------|------|--------|
| 5.1 | AccordisEnvironment (full reset/step impl) | server/accordis_environment.py | DONE |
| 5.2 | app.py using create_adapter() pattern | server/app.py | DONE |

---

## Tests

| # | Task | File | Status |
|---|------|------|--------|
| 6.1 | Model validation tests | tests/test_models.py | DONE |
| 6.2 | Environment reset/step tests | tests/test_environment.py | DONE |
| 6.3 | CorrectnessOracle tests | tests/test_verifier.py | DONE |
| 6.4 | Task grading tests | tests/test_tasks.py | DONE |
| 6.5 | RewardCalculator tests | tests/test_reward.py | DONE |
| 6.6 | SimulatedConsensusAdapter tests | tests/test_simulated_adapter.py | DONE |
