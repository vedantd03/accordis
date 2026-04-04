"""
Accordis Inference Script
===================================
MANDATORY environment variables:
    ACCORDIS_ADAPTER   Adapter to use: "simulated" (default) or "librabft".

LLM selection (LLMClientFactory, priority order):
    OPENAI_API_KEY     → OpenAI gpt-4o
    GEMINI_API_KEY     → Google gemini-1.5-pro

Optional:
    ACCORDIS_TASK      Task difficulty: "easy" (default), "medium", "hard".
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

import json
import os
import sys
import textwrap
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

from accordis.llm_factory import BaseLLMClient, LLMClientFactory
from accordis.models import (
    AccordisAction,
    AccordisObservation,
    MultiNodeAction,
    MultiNodeObservation,
    NodeID,
)
from accordis.server.accordis_environment import AccordisEnvironment
from accordis.server.adapters import create_adapter
from accordis.server.tasks.task_easy import EasyTask
from accordis.server.tasks.task_medium import MediumTask
from accordis.server.tasks.task_hard import HardTask

# ── Configuration ──────────────────────────────────────────────────────────────

ADAPTER_NAME = os.environ.get("ACCORDIS_ADAPTER", "simulated")
TASK_NAME    = os.environ.get("ACCORDIS_TASK", "easy")
MAX_STEPS    = int(os.environ.get("ACCORDIS_MAX_STEPS", "200"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an RL agent tuning Byzantine Fault-Tolerant (BFT) consensus parameters
    for a cluster of honest nodes. Your goal is to maximise transaction throughput
    and liveness while keeping the number of view changes low.

    At each step you receive JSON observations for every honest node and must return
    a JSON object mapping each node_id to its new BFT configuration.

    PARTIAL OBSERVABILITY — read this carefully:
      Each node's observation reflects only its own local log. Because QC messages
      propagate over a simulated network with variable latency, nodes' pending_txns
      values will diverge: the current leader commits its block immediately on QC
      formation, while replicas apply the same block one or more ticks later.

      The episode ends when all transactions have been finalized by QC — which
      happens before every replica's local log has fully caught up. This means
      done=True can arrive while some nodes still show pending_txns > 0.

      To track cluster-wide progress, use cluster_min_pending in the observation
      summary: this is the lowest pending_txns across all honest nodes and gives
      the best observable lower bound on remaining work. When cluster_min_pending
      approaches 0, the episode is close to ending. The node with role="leader"
      has the freshest view of the current batch — its pending_txns reflects
      its own block having been committed by QC this tick.

    Parameters and their safe ranges:
      view_timeout_ms:             200 - 10000  (ms before triggering a view change)
      pipeline_depth:              1 - 8        (concurrent in-flight rounds)
      replication_batch_size:      1 - 512      (txns per proposal)
      equivocation_threshold:      1 - 15       (votes before declaring equivocation)
      vote_aggregation_timeout_ms: 50 - 1000    (must be strictly less than view_timeout_ms / 2)

    Respond with ONLY valid JSON — no prose, no code fences, no markdown. Example:
    {
      "node_0": {
        "view_timeout_ms": 2000,
        "pipeline_depth": 2,
        "replication_batch_size": 64,
        "equivocation_threshold": 5,
        "vote_aggregation_timeout_ms": 500
      }
    }
    """
).strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _get_llm_action(
    llm: BaseLLMClient,
    step: int,
    obs: MultiNodeObservation,
    last_reward: float,
) -> MultiNodeAction:
    """Ask the LLM for a new BFT configuration and build a MultiNodeAction.

    Falls back to the node's current config if the LLM response is not valid JSON
    or is missing a node's entry.
    """
    node_ids = list(obs.nodes.keys())

    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Current observations:
        {_obs_to_dict(obs.nodes)}

        Return your BFT configuration for every node listed above.
        """
    ).strip()

    raw_configs: Dict = {}
    try:
        text = llm.complete(SYSTEM_PROMPT, user_prompt)
        raw_configs = json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM request or JSON parse failed: {exc}", flush=True)

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


# ── Logging ────────────────────────────────────────────────────────────────────

def log_start(task: str, adapter: str) -> None:
    print(f"[START] task={task} adapter={adapter}", flush=True)


def log_step(step: int, reward: float, total: float, done: bool) -> None:
    print(
        f"[STEP]  step={step} reward={reward:.1f} total={total:.1f} done={done}",
        flush=True,
    )

def log_action(action: MultiNodeAction) -> None:
    action_dict = {nid: {
        "view_timeout_ms": a.view_timeout_ms,
        "pipeline_depth": a.pipeline_depth,
        "replication_batch_size": a.replication_batch_size,
        "equivocation_threshold": a.equivocation_threshold,
        "vote_aggregation_timeout_ms": a.vote_aggregation_timeout_ms,
    } for nid, a in action.nodes.items()}
    print(f"[ACTION] {json.dumps(action_dict)}", flush=True)

def log_observation(obs: MultiNodeObservation) -> None:
    obs_dict = {nid: {
        "role": o.current_role.value,
        "view": o.current_view,
        "commit_tps": round(o.commit_throughput_tps, 2),
        "pending_txns": o.pending_txn_count,
        "pipeline_utilisation": round(o.pipeline_utilisation, 2),
        "qc_miss_streak": o.qc_formation_miss_streak,
        "view_changes_recent": o.view_change_count_recent,
        "current_config": {
            "view_timeout_ms": o.current_config.view_timeout_ms,
            "pipeline_depth": o.current_config.pipeline_depth,
            "replication_batch_size": o.current_config.replication_batch_size,
            "equivocation_threshold": o.current_config.equivocation_threshold,
            "vote_aggregation_timeout_ms": o.current_config.vote_aggregation_timeout_ms,
        },
    } for nid, o in obs.nodes.items()}
    print(f"[OBSERVATION] {json.dumps(obs_dict)}", flush=True)

def log_end(steps: int, total_reward: float, score: float) -> None:
    print(
        f"[END]   steps={steps} total_reward={total_reward:.1f} score={score:.2f}",
        flush=True,
    )


# ── Episode loop ───────────────────────────────────────────────────────────────

def main() -> None:
    task  = _select_task(TASK_NAME)
    conds = task.get_initial_conditions()

    env = AccordisEnvironment(adapter=create_adapter())
    llm = LLMClientFactory.create()

    log_start(TASK_NAME, ADAPTER_NAME)

    total_reward = 0.0
    steps_taken  = 0
    score        = 0.0

    try:
        obs: MultiNodeObservation = env.reset(**conds)
        log_observation(obs)
        last_reward = 0.0

        episode_max_steps = conds.get("max_steps", MAX_STEPS)
        for step in range(1, episode_max_steps + 1):
            action = _get_llm_action(llm, step, obs, last_reward)
            log_action(action)
            obs    = env.step(action)
            log_observation(obs)

            reward       = float(obs.reward) if obs.reward is not None else 0.0
            done         = bool(obs.done)
            last_reward  = reward
            total_reward += reward
            steps_taken   = step

            log_step(step=step, reward=reward, total=total_reward, done=done)

            if done:
                break

        if env._episode_log:
            score = task.grade(env._episode_log)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True, file=sys.stderr)

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(steps=steps_taken, total_reward=total_reward, score=score)


if __name__ == "__main__":
    main()
