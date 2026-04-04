# OpenEnv-Core Reference & Accordis Integration Audit

> Last updated: 2026-03-30
> Source: `openenv-core >= 0.2.1` installed in `base` conda environment
> Module path: `C:\Users\vedan\miniconda3\Lib\site-packages\openenv`
>
> Execute conda activate conda for access to openenv utilities
>
> ---
>
> name: openenv-cli
> description: "OpenEnv CLI (`openenv`) for scaffolding, validating, building, and pushing OpenEnv environments."
>
> ---
>
> The OpenEnv CLI command `openenv` is available.
> Use `openenv --help` to view available commands.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Types (`openenv.core.env_server.types`)](#2-core-types)
3. [Environment Interface (`openenv.core.env_server.interfaces`)](#3-environment-interface)
4. [Transform System](#4-transform-system)
5. [Rubric System (`openenv.core.rubrics`)](#5-rubric-system)
6. [Client System](#6-client-system)
7. [Auto-Discovery System (`openenv.auto`)](#7-auto-discovery-system)
8. [HTTP Server &amp; Serialization](#8-http-server--serialization)
9. [MCP Support](#9-mcp-support)
10. [Urban Lens Integration Audit](#10-urban-lens-integration-audit)
11. [Transform vs `_to_observation` Analysis](#11-transform-vs-_to_observation-analysis)

---

## 1. Architecture Overview

OpenEnv follows a **client-server** architecture with three conceptual layers:

```
┌──────────────────────────────────────────────────┐
│                 Auto Layer                        │
│  AutoEnv.from_env() / AutoAction.from_env()      │
│  HF Hub discovery, package install, class lookup  │
├──────────────────────────────────────────────────┤
│                 Client Layer                      │
│  EnvClient (async WebSocket)                      │
│  SyncEnvClient (sync wrapper)                     │
│  GenericEnvClient (untyped dict-based)            │
├──────────────────────────────────────────────────┤
│                 Server Layer                      │
│  Environment (abstract, Gym-style API)            │
│  HTTPEnvServer (FastAPI, HTTP + WebSocket)         │
│  Transform (observation pipeline)                 │
│  Rubric (reward computation, nn.Module-like)      │
├──────────────────────────────────────────────────┤
│                 Type Layer                         │
│  Action, Observation, State (Pydantic BaseModel)  │
│  WS message types, serialization utilities        │
└──────────────────────────────────────────────────┘
```

**Key Design Principles:**

- **Gym/Gymnasium API**: `reset() → Observation`, `step(Action) → Observation`
- **Pydantic everything**: All types are Pydantic models for validation & serialization
- **Server-side environment**: Environment runs on a server, clients connect via WebSocket
- **Pluggable reward**: Rubric system (like PyTorch's `nn.Module`) for composable rewards
- **Pluggable transform**: TorchRL-style observation transforms

---

## 2. Core Types

> Module: `openenv.core.env_server.types`

### `Action` (Base Class)

```python
class Action(BaseModel):
    """Base class for all environment actions."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Usage**: Subclass this to define your environment's action space. Fields get auto-validated by Pydantic.

### `Observation` (Base Class)

```python
class Observation(BaseModel):
    """Base class for all environment observations."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Usage**: Subclass this to define your environment's observation space. The `done`, `reward`, and `metadata` fields are provided by the base class — your subclass adds domain-specific fields.

### `State` (Base Class)

```python
class State(BaseModel):
    """Internal environment state, separate from observations."""
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
```

**Usage**: Represents internal environment bookkeeping (episode ID, step count). Uses `extra="allow"` so you can add arbitrary fields. This is NOT the hidden true state — it's the env's metadata.

### `ResetRequest` / `ResetResponse`

HTTP request/response models for the `/reset` endpoint. `ResetRequest` has `extra="allow"` to accept custom kwargs.

### `StepRequest` / `StepResponse`

HTTP request/response models for the `/step` endpoint.

### `StepResult` (Client-Side)

```python
@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
```

**Usage**: Returned by `EnvClient.step()` and `EnvClient.reset()`.

### WebSocket Message Types

| Type                      | Direction        | Purpose           |
| ------------------------- | ---------------- | ----------------- |
| `WSResetMessage`        | Client → Server | Reset environment |
| `WSStepMessage`         | Client → Server | Execute action    |
| `WSStateMessage`        | Client → Server | Request state     |
| `WSCloseMessage`        | Client → Server | Close session     |
| `WSObservationResponse` | Server → Client | Observation data  |
| `WSStateResponse`       | Server → Client | State data        |
| `WSErrorResponse`       | Server → Client | Error details     |

### Enums

| Enum             | Values                                                         | Purpose               |
| ---------------- | -------------------------------------------------------------- | --------------------- |
| `ServerMode`   | `SIMULATION`, `PRODUCTION`                                 | Server operation mode |
| `HealthStatus` | `HEALTHY`, `UNHEALTHY`, `DEGRADED`                       | Health check          |
| `WSErrorCode`  | `INVALID_JSON`, `UNKNOWN_TYPE`, `VALIDATION_ERROR`, etc. | WS error codes        |

### Concurrency Types

| Type                     | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `ConcurrencyConfig`    | Max concurrent envs + session timeout        |
| `ServerCapacityStatus` | Active vs max sessions                       |
| `SessionInfo`          | Session ID, timestamps, step count, env type |

---

## 3. Environment Interface

> Module: `openenv.core.env_server.interfaces`

### `Environment` (Abstract Base Class)

```python
class Environment(ABC, Generic[ActT, ObsT, StateT]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    rubric: Optional[Rubric]

    def __init__(self, transform=None, rubric=None):
        self.transform = transform
        self.rubric = rubric

    # REQUIRED to implement:
    @abstractmethod
    def reset(self, seed=None, episode_id=None, **kwargs) -> ObsT: ...
  
    @abstractmethod
    def step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT: ...
  
    @property
    @abstractmethod
    def state(self) -> StateT: ...

    # OPTIONAL to override:
    async def reset_async(self, ...) -> ObsT: ...   # defaults to calling sync reset
    async def step_async(self, ...) -> ObsT: ...    # defaults to calling sync step
    def get_metadata(self) -> EnvironmentMetadata: ...
    def close(self) -> None: ...

    # BUILT-IN helpers (call these, don't override):
    def _apply_transform(self, observation: ObsT) -> ObsT: ...
    def _apply_rubric(self, action: ActT, observation: ObsT) -> float: ...
    def _reset_rubric(self) -> None: ...
```

**Key Methods:**

| Method                 | Purpose                                   | When to Use                  |
| ---------------------- | ----------------------------------------- | ---------------------------- |
| `reset()`            | Reset env, return initial observation     | **Must implement**     |
| `step()`             | Execute action, return observation        | **Must implement**     |
| `state` (property)   | Get env state (episode_id, step_count)    | **Must implement**     |
| `reset_async()`      | Async version of reset                    | Override for true async      |
| `step_async()`       | Async version of step                     | Override for true async      |
| `_apply_transform()` | Apply registered Transform to observation | Call in reset/step           |
| `_apply_rubric()`    | Compute reward via Rubric                 | Call in step                 |
| `_reset_rubric()`    | Reset rubric state                        | Call in reset                |
| `get_metadata()`     | Return env name/description/version       | Override for custom metadata |
| `close()`            | Clean up resources                        | Override for cleanup         |

**Class Attributes:**

| Attribute                        | Default   | Purpose                                    |
| -------------------------------- | --------- | ------------------------------------------ |
| `SUPPORTS_CONCURRENT_SESSIONS` | `False` | Set `True` if env isolates session state |

---

## 4. Transform System

> Module: `openenv.core.env_server.interfaces` + `openenv.core.env_server.base_transforms`

### `Transform` (Abstract Base Class)

```python
class Transform(ABC, Generic[ObsT]):
    """TorchRL-style transform: takes Observation, returns (modified) Observation."""
    @abstractmethod
    def __call__(self, observation: ObsT) -> ObsT: ...
```

**Usage**: Create a Transform subclass to post-process observations (add computed fields, format data, augment info, etc.). Register it by passing to `Environment.__init__(transform=...)`.

### `CompositeTransform`

```python
class CompositeTransform(Transform):
    """Chains multiple transforms in sequence."""
    def __init__(self, transforms: list[Transform]): ...
    def __call__(self, observation: Observation) -> Observation: ...
```

**Usage**: Combine multiple transforms into a pipeline.

### `NullTransform`

```python
class NullTransform(Transform):
    """Pass-through, no modification."""
    def __call__(self, observation: Observation) -> Observation:
        return observation
```

### How Transforms are Applied

The `Environment._apply_transform()` method checks if `self.transform is not None` and calls it. **The environment must explicitly call `self._apply_transform(obs)` in its `reset()` and `step()` methods** — it's not automatic.

---

## 5. Rubric System

> Module: `openenv.core.rubrics`

### `Rubric` (Abstract Base Class)

```python
class Rubric(ABC):
    """nn.Module-like reward computation. Implement forward()."""
    last_score: Optional[float]
  
    @abstractmethod
    def forward(self, action, observation) -> float: ...
  
    # Auto-registered children, hooks, traversal:
    def children() -> Iterator[Rubric]: ...
    def named_rubrics(prefix="") -> Iterator[Tuple[str, Rubric]]: ...
    def get_rubric(path: str) -> Rubric: ...
    def register_forward_hook(hook): ...
    def register_forward_pre_hook(hook): ...
    def reset() -> None: ...
    def state_dict() -> Dict: ...
    def load_state_dict(state): ...
```

**Usage**: Implement `forward(action, observation) -> float` to compute step reward. Child rubrics are auto-registered when assigned as attributes.

### Container Rubrics

| Class           | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `Sequential`  | Chain rubrics in order                              |
| `Gate`        | Conditional: only apply inner rubric if gate passes |
| `WeightedSum` | Weighted combination of named rubrics               |
| `RubricList`  | List container (like `nn.ModuleList`)             |
| `RubricDict`  | Dict container (like `nn.ModuleDict`)             |

### `TrajectoryRubric`

Computes rewards based on full action-observation trajectory. Stores history internally.

### `ExponentialDiscountingTrajectoryRubric`

Applies exponential discounting to trajectory rewards.

### `LLMJudge`

Uses an LLM to evaluate action-observation pairs for reward.

---

## 6. Client System

### `EnvClient` (Async, Abstract)

> Module: `openenv.core.env_client`

```python
class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    def __init__(self, base_url, connect_timeout_s=10.0, message_timeout_s=60.0, ...): ...
  
    # WebSocket lifecycle
    async def connect() -> EnvClient: ...
    async def disconnect() -> None: ...
    async def close() -> None: ...
  
    # Gym-style API (async)
    async def reset(**kwargs) -> StepResult[ObsT]: ...
    async def step(action: ActT) -> StepResult[ObsT]: ...
    async def state() -> StateT: ...
  
    # Factory methods
    @classmethod
    async def from_docker_image(cls, image, provider=None, **kwargs) -> EnvClient: ...
    @classmethod
    async def from_env(cls, repo_id, use_docker=True, ...) -> EnvClient: ...
  
    # Sync wrapper
    def sync() -> SyncEnvClient: ...
  
    # REQUIRED to implement:
    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict: ...
    @abstractmethod
    def _parse_result(self, payload: Dict) -> StepResult[ObsT]: ...
    @abstractmethod
    def _parse_state(self, payload: Dict) -> StateT: ...
```

**Usage**: Subclass `EnvClient` to create a typed client for your environment. Implement `_step_payload`, `_parse_result`, `_parse_state`.

### `SyncEnvClient`

> Module: `openenv.core.sync_client`

Wraps an async `EnvClient` for synchronous usage. Created via `client.sync()`.

```python
sync_client = async_client.sync()
with sync_client:
    result = sync_client.reset()
    result = sync_client.step(action)
```

### `GenericEnvClient`

> Module: `openenv.core.generic_client`

Works with raw dicts — no custom Action/Observation classes needed.

```python
with GenericEnvClient(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step({"code": "print('hello')"})
```

### `GenericAction`

A dict subclass for semantic clarity when using `GenericEnvClient`.

---

## 7. Auto-Discovery System

> Module: `openenv.auto`

### `AutoEnv`

Factory class for automatic environment client selection. **Not instantiable directly**.

| Method                          | Purpose                                |
| ------------------------------- | -------------------------------------- |
| `AutoEnv.from_env(name, ...)` | Create client from name or HF Hub repo |
| `AutoEnv.list_environments()` | Print all available environments       |

Supports: installed packages, HF Hub repos, Docker images, UV runtime.

### `AutoAction`

Factory class for automatic Action class lookup. **Not instantiable directly**.

| Method                               | Purpose                         |
| ------------------------------------ | ------------------------------- |
| `AutoAction.from_env(name, ...)`   | Get Action class from env name  |
| `AutoAction.list_actions()`        | Print available Action classes  |
| `AutoAction.get_action_info(name)` | Get metadata about action class |

---

## 8. HTTP Server & Serialization

### `create_app()` / `create_fastapi_app()`

> Module: `openenv.core.env_server.http_server`

```python
app = create_app(
    EnvironmentClass,       # Environment factory (class)
    ActionClass,            # Action subclass
    ObservationClass,       # Observation subclass
    env_name="my_env",
    max_concurrent_envs=1,  # WebSocket sessions
)
```

Creates a FastAPI app with:

- `POST /reset` — Reset environment
- `POST /step` — Execute action
- `GET /state` — Get state
- `GET /schema` — Get action/observation JSON schemas
- `GET /health` — Health check
- `GET /metadata` — Environment metadata
- `WS /ws` — WebSocket for persistent sessions

### `HTTPEnvServer`

Lower-level class that wraps any `Environment` subclass. Manages sessions, thread pools, and WebSocket connections.

### Serialization Utilities

| Function                                             | Purpose                                                                      |
| ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `serialize_observation(obs)`                       | Convert Observation →`{"observation": {...}, "reward": ..., "done": ...}` |
| `deserialize_action(data, cls)`                    | Convert dict → Action subclass via Pydantic validation                      |
| `deserialize_action_with_preprocessing(data, cls)` | Same + tensor/type conversions for web UI                                    |

---

## 9. MCP Support

OpenEnv supports MCP (Model Context Protocol) for tool-based environments:

| Class                                              | Purpose                              |
| -------------------------------------------------- | ------------------------------------ |
| `MCPEnvironment`                                 | Environment that wraps an MCP server |
| `MCPClientBase`                                  | Base MCP client                      |
| `MCPToolClient`                                  | MCP tool-calling client              |
| `ListToolsAction` / `CallToolAction`           | MCP action types                     |
| `ListToolsObservation` / `CallToolObservation` | MCP observation types                |

---

## Quick Reference: Import Paths# Server-side (Environment implementation)
