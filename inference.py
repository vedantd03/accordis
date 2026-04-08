"""
Accordis Inference Script
===================================
MANDATORY environment variables:
    ACCORDIS_ADAPTER   Adapter to use: "simulated" (default) or "librabft".

LLM selection:
    API_KEY / HF_TOKEN     → OpenAI compatible model

Optional:
    ACCORDIS_TASKS      Task difficulty: "easy" (default), "medium", "hard".
    ACCORDIS_MAX_STEPS Override maximum steps per episode (default from task).

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> adapter=<adapter>
    [STEP]  step=<n> reward=<r> total=<cumulative> done=<True|False>
    [END]   steps=<n> total_reward=<r> score=<s>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and total_reward are formatted to 1 decimal place.
    - score is formatted to 2 decimal places.
    - done is a Python bool string (True or False).
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=easy adapter=simulated
    [STEP]  step=1 reward=-1.0 total=-1.0 done=False
    [STEP]  step=2 reward=59.0 total=58.0 done=False
    ...
    [END]   steps=87 total_reward=340.0 score=0.73
"""
from __future__ import annotations

import json
import os
import asyncio
import argparse
import textwrap
import subprocess
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Optional
from openai import OpenAI

from models import (
    AccordisAction,
    AccordisObservation,
    MultiNodeAction,
    MultiNodeObservation,
    NodeID,
    STATIC_BASELINE_CONFIG,
)

from client import AccordisEnvironment
from server.accordis_environment import AccordisEnvironment as ServerAccordisEnvironment
from server.tasks.task_easy import EasyTask
from server.tasks.task_medium import MediumTask
from server.tasks.task_hard import HardTask

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a control policy that tunes Byzantine Fault-Tolerant (BFT) consensus
    parameters for a cluster of honest nodes. At each step you receive JSON
    observations for every honest node and must return a JSON config for each
    node. Your goal: maximise transaction throughput, keep view changes low,
    and stay stable.

    ════════════════════════════════════════════════════════════════════════════
    FIRST PRINCIPLE — STABILITY BEATS CLEVERNESS
    ════════════════════════════════════════════════════════════════════════════
    Oscillating parameters step-to-step DESTROYS throughput. The cluster needs
    several consecutive steps with the SAME config to build a commit pipeline.
    Default behaviour: REPEAT THE PREVIOUS STEP'S CONFIG. Only change a value
    when a specific decision rule below tells you to. Never change more than
    TWO parameters in a single step.

    ════════════════════════════════════════════════════════════════════════════
    DEFAULT STARTING CONFIG (use this on step 0 for every node)
    ════════════════════════════════════════════════════════════════════════════
      view_timeout_ms             = 1000
      pipeline_depth              = 4
      replication_batch_size      = 256
      equivocation_threshold      = 5
      vote_aggregation_timeout_ms = 800

    These defaults are SAFE under both clean and adversarial conditions —
    start here on step 0 and only adjust based on the rules below.

    ════════════════════════════════════════════════════════════════════════════
    TIMING MODEL — INTERNALISE THIS
    ════════════════════════════════════════════════════════════════════════════
    1 environment step ≈ 50 ms of simulated wall-clock time. Episodes have a
    bounded step budget. view_timeout_ms is the wall-clock time the cluster
    waits for a leader before triggering a view change. If view_timeout_ms
    is set close to the remaining step budget, NO view change can fire and
    a Byzantine leader stall deadlocks the rest of the episode.

    HARD CEILING: never set view_timeout_ms above 1500 ms. The bound allows
    up to 3000 ms but using it is almost always a mistake — it leaves no
    room for the pacemaker to recover from an unresponsive leader.

    view_stuck_ms reports how long THIS NODE has been waiting in its current
    view. It is in the same unit as view_timeout_ms — compare them directly.

    ════════════════════════════════════════════════════════════════════════════
    DECISION RULES — apply in this order, top to bottom, at most one fires
    ════════════════════════════════════════════════════════════════════════════

    RULE 1 — LEADER STALL (highest priority)
      IF any node has view_stuck_ms > 0.6 × view_timeout_ms
         AND qc_miss_streak > 5
      THEN halve view_timeout_ms (floor at 400 ms) on EVERY node.
      WHY: the current leader is unresponsive; rotating faster lets the
      cluster pick a non-Byzantine leader.

    RULE 2 — DELAY ATTACK
      IF qc_miss_streak ≥ 3 on any node
         AND view_stuck_ms is NOT growing fast (no leader stall)
      THEN raise vote_aggregation_timeout_ms by 200 ms (cap at 1000) on
      EVERY node. Do NOT touch view_timeout_ms.
      WHY: the leader is alive but votes are arriving late under
      SELECTIVE_DELAY / ADAPTIVE_MIRROR. Bigger vote window = more QCs.

    RULE 3 — EQUIVOCATION DETECTED
      IF any peer in suspected_byzantine is true
      THEN keep replication_batch_size ≥ 128, and lower equivocation_threshold
      by 1 (floor at 2). Do NOT lower batch_size below 128 even under attack.
      WHY: small batches throttle throughput; the right defence is to detect
      attackers earlier, not to ship less data per round.

    RULE 4 — THROUGHPUT RAMP (when no rule above fired)
      IF cluster commit_tps is stable and > 0
         AND no rule above fired
      THEN raise replication_batch_size by 64 (cap at 512) on EVERY node.
      WHY: in a healthy cluster the only way to drain the pool faster is to
      ship more txns per block.

    RULE 5 — DEFAULT
      IF none of the above fired, REPEAT the previous step's config exactly.
      Stability is the default action, not a fallback.

    ════════════════════════════════════════════════════════════════════════════
    HARD CONSTRAINTS — NEVER VIOLATE
    ════════════════════════════════════════════════════════════════════════════
    - replication_batch_size      ≥ 64    (lower values throttle throughput)
    - vote_aggregation_timeout_ms < view_timeout_ms / 2   (env will clamp)
    - view_timeout_ms             ≤ 1500  (soft cap; bound allows 3000 but don't)
    - Apply the SAME config to every node unless a node-specific rule fires
      (no current rule is node-specific — use uniform configs)

    ════════════════════════════════════════════════════════════════════════════
    PARAMETER RANGES (env clamps to these)
    ════════════════════════════════════════════════════════════════════════════
      view_timeout_ms             : 200 – 3000   (target ≤ 1500)
      pipeline_depth              : 1   – 8      (target 4)
      replication_batch_size      : 1   – 512    (target 256–512)
      equivocation_threshold      : 1   – 15     (target 3–5)
      vote_aggregation_timeout_ms : 50  – 1000   (target 600–1000, must be < view_timeout_ms / 2)

    ════════════════════════════════════════════════════════════════════════════
    OBSERVATION FORMAT
    ════════════════════════════════════════════════════════════════════════════
    Each step's observation is a JSON object with two top-level keys:
      - "cluster_min_pending": int — minimum pending_txns across all honest
        nodes. The episode is close to ending when this approaches 0.
      - "nodes": object keyed by node_id, each containing local metrics:
        role, view, commit_tps, pending_txns, pipeline_utilisation,
        qc_miss_streak, view_changes_recent, view_stuck_ms,
        suspected_byzantine, current_config.

    PARTIAL OBSERVABILITY: nodes' pending_txns values diverge because QC
    messages propagate with latency. Use cluster_min_pending as the
    cluster-wide progress signal. The leader has the freshest view.

    ════════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ════════════════════════════════════════════════════════════════════════════
    Return a FLAT JSON object keyed by node_id — do NOT nest under "nodes"
    or any other wrapper. Include EVERY node_id present in the observation.
    Respond with ONLY valid JSON — no prose, no code fences, no markdown.

    Example (4-node cluster, step 0 with the default config):
    {
      "node_0": {"view_timeout_ms": 1000, "pipeline_depth": 4, "replication_batch_size": 256, "equivocation_threshold": 5, "vote_aggregation_timeout_ms": 800},
      "node_1": {"view_timeout_ms": 1000, "pipeline_depth": 4, "replication_batch_size": 256, "equivocation_threshold": 5, "vote_aggregation_timeout_ms": 800},
      "node_2": {"view_timeout_ms": 1000, "pipeline_depth": 4, "replication_batch_size": 256, "equivocation_threshold": 5, "vote_aggregation_timeout_ms": 800},
      "node_3": {"view_timeout_ms": 1000, "pipeline_depth": 4, "replication_batch_size": 256, "equivocation_threshold": 5, "vote_aggregation_timeout_ms": 800}
    }
    """
).strip()

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Docker image name for OpenEnv runtime
ACCORDIS_BASE_URL = os.getenv("ACCORDIS_BASE_URL") or os.getenv("BASE_URL")

class OpenAIClient():
    """OpenAI chat-completion client with Hugging Face model compatibility."""

    def __init__(self, model: str) -> None:
        self._BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self._MODEL = model
        self._API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self._client = OpenAI(
            base_url=self._BASE_URL,
            api_key=self._API_KEY
        )

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if not callable(close):
            return

        result = close()
        if asyncio.iscoroutine(result):
            await result

# ── Logging ────────────────────────────────────────────────────────────────────

def log_start(task: str, adapter: str) -> None:
    print(f"[START] task={task} adapter={adapter}")


def log_step(step: int, action: any, obs: any, reward: float, total: float, done: bool) -> None:
    print(
        f"[STEP]  step={step} action={action} observation={obs} reward={reward:.2f} total={total:.1f} done={done}"
    )

def log_end(steps: int, total_reward: float, score: float) -> None:
    print(
        f"[END]   steps={steps} total_reward={total_reward:.1f} score={score:.2f}"
    )


def _select_task(name: str):
    name = name.lower()
    if name == "medium":
        return MediumTask()
    if name == "hard":
        return HardTask()
    return EasyTask(curriculum_level=1)


def _obs_to_dict(obs_nodes: Dict[NodeID, AccordisObservation]) -> str:
    """Compact JSON summary of all honest-node observations for the LLM.

    Includes a top-level cluster_min_pending field: the minimum pending_txns
    across all honest nodes. This is derived entirely from per-node observable
    data and gives the best lower bound on remaining cluster work.
    """
    node_summary = {}
    for nid, o in obs_nodes.items():
        node_summary[nid] = {
            "role":                   o.current_role.value,
            "view":                   o.current_view,
            "commit_tps":             round(o.commit_throughput_tps, 2),
            "pending_txns":           o.pending_txn_count,
            "pipeline_utilisation":   round(o.pipeline_utilisation, 2),
            "qc_miss_streak":         o.qc_formation_miss_streak,
            "view_changes_recent":    o.view_change_count_recent,
            "view_stuck_ms":          o.view_stuck_ms,
            "suspected_byzantine":    o.suspected_byzantine,
            "current_config": {
                "view_timeout_ms":             o.current_config.view_timeout_ms,
                "pipeline_depth":              o.current_config.pipeline_depth,
                "replication_batch_size":      o.current_config.replication_batch_size,
                "equivocation_threshold":      o.current_config.equivocation_threshold,
                "vote_aggregation_timeout_ms": o.current_config.vote_aggregation_timeout_ms,
            },
        }

    cluster_min_pending = min(o.pending_txn_count for o in obs_nodes.values())

    summary = {"cluster_min_pending": cluster_min_pending, "nodes": node_summary}
    return json.dumps(summary, indent=2)


def _get_static_action(obs: MultiNodeObservation) -> MultiNodeAction:
    """Return STATIC_BASELINE_CONFIG for every honest node — no changes."""
    node_actions: Dict[NodeID, AccordisAction] = {}
    for nid in obs.nodes:
        node_actions[nid] = AccordisAction(
            node_id=nid,
            view_timeout_ms=STATIC_BASELINE_CONFIG.view_timeout_ms,
            pipeline_depth=STATIC_BASELINE_CONFIG.pipeline_depth,
            replication_batch_size=STATIC_BASELINE_CONFIG.replication_batch_size,
            equivocation_threshold=STATIC_BASELINE_CONFIG.equivocation_threshold,
            vote_aggregation_timeout_ms=STATIC_BASELINE_CONFIG.vote_aggregation_timeout_ms,
        )
    return MultiNodeAction(nodes=node_actions)


async def _get_llm_action(
    llm: OpenAIClient,
    step: int,
    obs: MultiNodeObservation,
    last_reward: float,
) -> MultiNodeAction:
    node_ids = list(obs.nodes.keys())

    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Current observations:
        {_obs_to_dict(obs.nodes)}

        Return your BFT configuration for every node_id found under "nodes" above.
        """
    ).strip()

    raw_configs: Dict = {}
    try:
        text = llm.complete(SYSTEM_PROMPT, user_prompt)
        raw_configs = json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM request or JSON parse failed: {exc}")

    node_actions: Dict[NodeID, AccordisAction] = {}
    for nid in node_ids:
        cfg_raw   = raw_configs.get(nid, {})
        prev_cfg  = obs.nodes[nid].current_config
        node_actions[nid] = AccordisAction(
            node_id=nid,
            view_timeout_ms=int(
                cfg_raw.get("view_timeout_ms", prev_cfg.view_timeout_ms)
            ),
            pipeline_depth=int(
                cfg_raw.get("pipeline_depth", prev_cfg.pipeline_depth)
            ),
            replication_batch_size=int(
                cfg_raw.get("replication_batch_size", prev_cfg.replication_batch_size)
            ),
            equivocation_threshold=int(
                cfg_raw.get("equivocation_threshold", prev_cfg.equivocation_threshold)
            ),
            vote_aggregation_timeout_ms=int(
                cfg_raw.get("vote_aggregation_timeout_ms", prev_cfg.vote_aggregation_timeout_ms)
            ),
        )

    return MultiNodeAction(nodes=node_actions)


async def _run_single_task(
    task_name: str,
    provider: str,
    model: Optional[str],
) -> dict:
    """Run one full episode for the given task and return a result dict."""
    task  = _select_task(task_name)
    conds = task.get_initial_conditions()

    try:
        env = await _create_environment_client()
    except Exception as exc:
        print(f"[WARN] Client environment unavailable; using in-process server fallback: {exc}")
        return await _run_single_task_server(task_name, provider, model, llm=llm)

    llm: Optional[OpenAIClient] = None
    if provider != "static":
        llm = _create_llm_client(provider=provider, model=model)

    log_start(task_name, os.getenv("ACCORDIS_ADAPTER", "simulated"))

    total_reward = 0.0
    steps_taken  = 0
    score        = 0.0

    try:
        reset_result = await env.reset(**conds)
        obs: MultiNodeObservation = reset_result.observation
        last_reward  = 0.0

        episode_max_steps = conds.get("max_steps", int(os.getenv("ACCORDIS_MAX_STEPS", "100")))
        for step in range(1, episode_max_steps + 1):
            if provider == "static":
                action = _get_static_action(obs)
            else:
                action = await _get_llm_action(llm, step, obs, last_reward)
            
            step_result = await env.step(action)

            obs = step_result.observation
            reward       = float(step_result.reward) if step_result.reward is not None else 0.0
            done         = bool(step_result.done)
            last_reward  = reward
            total_reward += reward
            steps_taken   = step

            log_step(step=step, action=action, obs=obs, reward=reward, total=total_reward, done=done)

            if done:
                break

        state = await env.state()
        if state.episode_log is not None:
            score = task.grade(state.episode_log)

    except Exception as exc:
        print(f"[ERROR] Episode error: {exc}")

    finally:
        if llm is not None:
            try:
                await llm.close()
            except Exception:
                pass
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}")
        log_end(steps=steps_taken, total_reward=total_reward, score=score)

    return {
        "task":         task_name,
        "steps":        steps_taken,
        "total_reward": round(total_reward, 2),
        "score":        round(score, 4),
    }


async def _run_single_task_server(
    task_name: str,
    provider: str,
    model: Optional[str],
    llm: Optional[OpenAIClient] = None,
) -> dict:
    """Run one episode using the in-process server environment as a fallback."""
    task = _select_task(task_name)
    conds = task.get_initial_conditions()

    env = ServerAccordisEnvironment()
    owns_llm = llm is None and provider != "static"
    if llm is None and provider != "static":
        llm = _create_llm_client(provider=provider, model=model)

    log_start(task_name, os.getenv("ACCORDIS_ADAPTER", "simulated"))

    total_reward = 0.0
    steps_taken = 0
    score = 0.0

    try:
        obs: MultiNodeObservation = env.reset(**conds)
        last_reward = 0.0

        episode_max_steps = conds.get("max_steps", int(os.getenv("ACCORDIS_MAX_STEPS", "100")))
        for step in range(1, episode_max_steps + 1):
            if provider == "static":
                action = _get_static_action(obs)
            else:
                action = await _get_llm_action(llm, step, obs, last_reward)

            obs = env.step(action)

            reward = float(obs.reward) if obs.reward is not None else 0.0
            done = bool(obs.done)
            last_reward = reward
            total_reward += reward
            steps_taken = step

            log_step(step=step, action=action, obs=obs, reward=reward, total=total_reward, done=done)

            if done:
                break

        if env._episode_log is not None:
            score = task.grade(env._episode_log)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}")

    finally:
        if owns_llm and llm is not None:
            try:
                await llm.close()
            except Exception:
                pass
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        log_end(steps=steps_taken, total_reward=total_reward, score=score)

    return {
        "task": task_name,
        "steps": steps_taken,
        "total_reward": round(total_reward, 2),
        "score": round(score, 4),
    }


async def _create_environment_client() -> AccordisEnvironment:
    """Create an environment client from either Docker or a running server."""
    if IMAGE_NAME:
        print(f"Using Docker image: {IMAGE_NAME}")
        try:
            return await AccordisEnvironment.from_docker_image(image=IMAGE_NAME)
        except Exception as exc:
            if ACCORDIS_BASE_URL:
                print(
                    "Docker runtime unavailable; falling back to "
                    f"ACCORDIS_BASE_URL={ACCORDIS_BASE_URL}"
                )
                return AccordisEnvironment(base_url=ACCORDIS_BASE_URL)

            message = (
                "Failed to start the Accordis environment from Docker image "
                f"{IMAGE_NAME!r}. Make sure Docker is installed and the daemon "
                "is running, or set ACCORDIS_BASE_URL=http://localhost:8000 to "
                "use an already running server."
            )
            if isinstance(exc, subprocess.CalledProcessError):
                detail = f" docker command failed: {' '.join(exc.cmd)} (exit {exc.returncode})."
            else:
                detail = f" original error: {exc}"
            raise RuntimeError(message + detail) from exc

    if ACCORDIS_BASE_URL:
        print(f"Using running Accordis server: {ACCORDIS_BASE_URL}")
        return AccordisEnvironment(base_url=ACCORDIS_BASE_URL)

    raise ValueError(
        "No environment target configured. Set IMAGE_NAME to use Docker, or "
        "set ACCORDIS_BASE_URL=http://localhost:8000 to use a running Accordis server."
    )


def _create_llm_client(provider: str, model: Optional[str]) -> OpenAIClient:
    """Create the right LLM client for the selected provider."""
    if model is None:
        raise ValueError("model must be set when provider is not 'static'")

    if provider == "huggingface" or provider == "openai":
        return OpenAIClient(model=model)

    return OpenAIClient.create(provider=provider, model=model)


async def inference(
    provider: str,
    model: Optional[str],
    tasks: Optional[list[str]] = None,
) -> dict:
    """Run baseline evaluation across one or more task difficulties.

    Args:
        tasks:    List of task names to run. Defaults to all three if None.
        provider: Inference provider — "static" or "openai".
        model:    LLM model name (required when provider is not "static").

    Returns:
        A dict with status and per-task results.
    """
    if tasks is None:
        tasks = ["easy", "medium", "hard"]

    results = {}
    for task_name in tasks:
        results[task_name] = await _run_single_task(task_name, provider, model)

    inference_result = {
        "status":   "success",
        "provider": provider,
        "tasks":    tasks,
        "data":     results,
    }
    print(f"Baseline result: {json.dumps(inference_result, indent=2)}")
    return inference_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Accordis baseline evaluation")
    parser.add_argument(
        "--provider", default="huggingface",
        choices=["static", "huggingface", "openai"],
        help="LLM provider (default: huggingface)"
    )
    parser.add_argument(
        "--model", default=os.getenv("MODEL_NAME", None),
        help="LLM model to use (default: provider-specific default)"
    )
    parser.add_argument(
        "--tasks", nargs="*", default=os.getenv("ACCORDIS_TASKS", "easy,medium,hard").split(","),
        help="Difficulty levels to evaluate (default: all(easy,medium,hard). Ignored when --scenario is set."
    )
    args = parser.parse_args()
    if args.provider == "huggingface" and args.model is None:
        args.model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    if args.provider == "openai" and args.model is None:
        raise ValueError("Model must be specified for OpenAI provider")
    
    asyncio.run(inference(
        provider=args.provider,
        model=args.model,
        tasks=args.tasks
    ))
